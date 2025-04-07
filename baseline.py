import pandas as pd
import gym
import numpy as np
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class BTCTradingEnv(gym.Env):
    """自定义BTC永续合约交易环境"""
    def __init__(self, data: pd.DataFrame, window_size: int = 10, 
                 initial_balance: float = 10000,  # 初始资金，假设以 USDT 计价
                 taker_fee: float = 0.00075,      # Bybit 吃单手续费率 0.075%
                 maker_fee: float = -0.00025,     # Bybit 挂单手续费率 -0.025%（返佣），本例未使用
                 slippage_rate: float = 0.001,    # 滑点最大百分比0.1%（这里用0.001表示千分之一，实际0.1%）
                 max_drawdown: float = 0.2,       # 最大容忍回撤20%，超出则终止
                 funding_rate_col: str = None,    # 数据中资金费率列名（如果提供）
                 funding_interval: int = None     # 资金费结算间隔（分钟），如8小时=480分钟
                 ):
        super(BTCTradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.slippage_rate = slippage_rate
        self.max_drawdown = max_drawdown
        # 提取价格序列用于计算盈亏，假设使用收盘价作为交易价基准
        assert 'close' in data.columns, "数据缺少 'close' 收盘价列"
        self.close_prices = self.data['close'].values
        # 若有资金费率数据
        if funding_rate_col and funding_rate_col in data.columns:
            self.funding_rates = self.data[funding_rate_col].values
        else:
            self.funding_rates = None
        # Funding结算步长（以分钟计），如果未提供但有funding_rate数据，我们默认为480分钟(8小时)
        if funding_interval:
            self.funding_interval = funding_interval
        else:
            self.funding_interval = 480  # 缺省8小时结算一次
        
        # 计算价格变化作为状态特征：这里用相对收益率（对数或百分比变化）
        # 我们计算对数收益率，避免尺度问题： r_t = log(p_t / p_{t-1})
        self.returns = np.zeros_like(self.close_prices)
        self.returns[1:] = np.diff(np.log(self.close_prices))
        # 定义Gym空间
        # 状态维度: window_size 个最近对数收益率 + 当前仓位比例 (共 window_size+1 维)
        state_dim = window_size + 1
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        # 动作空间: 连续 [-1,1] 区间，只有1维（目标仓位比例）
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # 初始化内部状态变量
        self.reset()  # 在构造时先初始化
    
    def reset(self):
        """环境重置，开始新一个 episode"""
        # 初始化资金和持仓
        self.cash_balance = self.initial_balance  # 可用资金（USDT）
        self.btc_position = 0.0                  # 持有BTC数量（多头正，空头负）
        self.total_equity = self.initial_balance  # 账户总净值 = 现金 + 持仓价值
        self.peak_equity = self.total_equity      # 记录历史峰值净值用于计算回撤
        
        # 随机选择起始位置开始一个episode，以增加训练多样性
        # 起始索引需预留window_size长度的历史用于状态，且后面留至少1步
        max_start = len(self.data) - self.window_size - 1
        self.current_step = np.random.randint(self.window_size, max_start)  # 当前索引（将作为第一个状态的末尾）
        # 重置上一步目标仓位（用于计算交易量），初始为当前仓位
        self.prev_action = 0.0
        
        # 构建初始观测状态: 最近 window_size 个对数收益率 + 当前持仓比例
        obs = self._get_observation()
        return obs
    
    def _get_observation(self):
        """构建当前状态观测，包括近期价格变动和当前仓位"""
        # 提取最近 window_size 个对数收益率作为状态特征
        start = self.current_step - self.window_size
        if start < 0:
            # 边界情况（不太会发生，因为current_step>=window_size），如不足窗口则用开头数据
            start = 0
        recent_returns = self.returns[start:self.current_step]
        # 若实际长度不足window_size（episode序列末尾），则用零填充
        if len(recent_returns) < self.window_size:
            pad_len = self.window_size - len(recent_returns)
            recent_returns = np.concatenate([np.zeros(pad_len), recent_returns])
        # 当前仓位占比（仓位名义价值/总净值），注意仓位名义价值 = btc_position * 当前价格
        current_price = self.close_prices[self.current_step]
        position_value = self.btc_position * current_price
        # 仓位比例 = 持仓名义价值 / 总净值
        position_pct = 0.0 if self.total_equity == 0 else position_value / self.total_equity
        # 将状态组成数组: [近期对数收益..., 仓位比例]
        obs = np.append(recent_returns, position_pct).astype(np.float32)
        return obs
    
    def step(self, action):
        """执行给定动作(调整仓位)，推进一个时间步"""
        # 动作为numpy数组，提取其中的目标仓位比例
        target_position_pct = float(action[0])
        # 限制动作范围在 [-1, 1]
        if target_position_pct > 1: 
            target_position_pct = 1.0
        if target_position_pct < -1:
            target_position_pct = -1.0
        
        # 当前价格（用于交易执行和之后的盈亏计算）
        current_price = self.close_prices[self.current_step]
        done = False
        info = {}
        
        # 计算当前总净值用于确定目标仓位的BTC数量
        self.total_equity = self.cash_balance + self.btc_position * current_price
        if self.total_equity <= 0:
            # 账户净值为0或负，爆仓，结束
            done = True
            # 巨额负奖励以惩罚爆仓
            reward = -1.0
            info['equity'] = 0.0
            return self._get_observation(), reward, done, info
        
        # 计算目标BTC持仓数量 (目标仓位比例 * 当前净值 / 当前价格)
        target_btc_position = target_position_pct * (self.total_equity / current_price)
        # 计算需要交易的BTC数量差额
        trade_size = target_btc_position - self.btc_position
        
        # 如果 trade_size != 0，执行交易
        if abs(trade_size) > 1e-6:
            # 模拟滑点: 调整执行价
            slip_pct = np.random.uniform(0, self.slippage_rate)  # 随机滑点百分比
            if trade_size > 0:
                # 买入增加仓位: 用更高价格成交
                exec_price = current_price * (1 + slip_pct)
            else:
                # 卖出减少/做空仓位: 用更低价格成交
                exec_price = current_price * (1 - slip_pct)
            # 计算交易价值（USDT）
            trade_value = abs(trade_size) * exec_price
            # 计算手续费成本
            fee_cost = trade_value * self.taker_fee  # 这里假设都是taker单
            # 更新现金余额和仓位
            if trade_size > 0:
                # 买入: 减少现金，增加BTC仓位
                self.cash_balance -= trade_value  # 支出trade_value购买BTC
                self.btc_position += trade_size   # 增加持仓
                # 扣除手续费（以USDT计）
                self.cash_balance -= fee_cost
            else:
                # 卖出: 增加现金，减少BTC仓位
                self.cash_balance += trade_value  # 卖出BTC获得现金
                self.btc_position += trade_size   # trade_size为负，实际上是减少持仓或者转为空头
                # 扣除手续费
                self.cash_balance -= fee_cost
            # 确保现金不为负（若现金不足支付手续费，可能借贷，本例简单处理）
            if self.cash_balance < 0:
                # 若出现现金为负（杠杆借款情况），仍允许，但记录债务？本例简单置0避免负数
                self.cash_balance = 0
            
            # 更新净值并检查可能的回撤
            self.total_equity = self.cash_balance + self.btc_position * current_price
            # 记录此次交易信息
            info['trade_size'] = trade_size
            info['trade_price'] = exec_price
            info['fee_paid'] = fee_cost
        else:
            # 没有交易，fee_cost=0
            fee_cost = 0.0
        
        # **资金费处理**: 检查该步是否有资金费结算
        if self.funding_rates is not None:
            # 判断当前步是否是资金费结算点。
            # 若提供funding_rate序列，则直接取当前步的费率（假设在结算步，该费率非零）。
            # 或根据时间间隔判断（例如每隔 funding_interval 分钟）
            fund_rate = 0.0
            if self.funding_rates is not None:
                fund_rate = self.funding_rates[self.current_step]
            # 如果没有直接给定fund_rate序列，可以用当前步索引判断
            elif self.current_step % self.funding_interval == 0:
                # 若恰逢结算间隔，假设一个资金费率值，比如从另一个来源获取。
                # 这里简单假设fund_rate=0（没有提供就不收付资金费）
                fund_rate = 0.0
            # 如果有资金费需要结算
            if fund_rate != 0:
                # 计算资金费支出/收入
                # 多头付费，空头收取（fund_rate为正）；反之空头付费（fund_rate负则空头付）
                # 持仓名义价值
                position_value = self.btc_position * current_price
                # 资金费变化的资金
                funding_pnl = - position_value * fund_rate
                # 更新现金余额和净值
                self.cash_balance += funding_pnl
                self.total_equity = self.cash_balance + self.btc_position * current_price
                info['funding_rate'] = fund_rate
                info['funding_pnl'] = funding_pnl
        
        # 推进到下一个时间步（价格变动）
        # 保存当前净值用于计算收益
        prev_equity = self.total_equity
        prev_price = current_price
        # 前进一步
        self.current_step += 1
        # 若到达数据结尾，则终止episode
        if self.current_step >= len(self.data) - 1:
            done = True
        
        # 当前步新的价格，用于计算本步收益
        new_price = self.close_prices[self.current_step]
        # 根据仓位计算未经惩罚的收益变化（含未实现盈亏）
        # 注意：持仓未变，净值变化主要来自价格变动的浮动盈亏
        price_change = new_price - prev_price
        unrealized_pnl = self.btc_position * price_change  # 持仓产生的盈亏（未实时兑现，但记入净值）
        # 计算新的净值
        self.total_equity = self.cash_balance + self.btc_position * new_price
        # 更新历史最高净值
        if self.total_equity > self.peak_equity:
            self.peak_equity = self.total_equity
        # 计算当前回撤比例
        drawdown = (self.peak_equity - self.total_equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        # **奖励计算**：
        # 基础收益率（净值相对变化率）
        reward_return = (self.total_equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
        reward = reward_return
        # 如果收益为负，则加大惩罚（使负reward更加负，增强风险厌恶）
        if reward < 0:
            # 惩罚系数，例如2倍放大亏损
            reward *= 2.0
        # 交易频率惩罚：若本步执行了交易，则减去一个固定小惩罚项
        if abs(trade_size) > 1e-6:
            reward -= 0.001  # 每次交易额外惩罚，值可调，过大会抑制交易&#8203;:contentReference[oaicite:13]{index=13}
        # 最大回撤惩罚：如果当前回撤超过阈值，则给予强惩罚并结束episode
        if drawdown > self.max_drawdown:
            # 触发强制止损事件
            done = True
            # 给一个较大的负奖励，惩罚超过最大回撤风险
            reward -= 0.5
            info['max_drawdown_exceeded'] = True
        
        # 准备下一步观测
        obs = self._get_observation()
        # 在info中记录额外信息，例如当前净值和回撤等
        info['equity'] = float(self.total_equity)
        info['drawdown'] = float(drawdown)
        
        return obs, float(reward), done, info


# 准备数据
data_df = pd.read_csv("BTCUSDT_5m_2023-01-01_to_2025-04-01.csv")
# 简单的训练集/测试集拆分，例如按照日期或索引拆分
train_size = int(len(data_df) * 0.8)
train_data = data_df.iloc[:train_size].copy()
test_data = data_df.iloc[train_size:].copy()

# 创建训练环境和测试环境
train_env = BTCTradingEnv(data=train_data, window_size=10, initial_balance=10000, 
                           taker_fee=0.00075, slippage_rate=0.001, max_drawdown=0.2)
test_env = BTCTradingEnv(data=test_data, window_size=10, initial_balance=10000, 
                          taker_fee=0.00075, slippage_rate=0.001, max_drawdown=0.2)

# 将环境包装为向量化环境（Stable-Baselines3需要），可以使用DummyVecEnv
train_env = DummyVecEnv([lambda: train_env])


def optimize_agent(trial):
    """Optuna调优目标函数：给定一组超参数训练并评估模型收益"""
    # 建议的超参数
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)       # 学习率在1e-5到1e-3对数均匀采样
    gamma = trial.suggest_uniform('gamma', 0.95, 0.999)   # 折扣因子在0.95~0.999之间
    ent_coef = trial.suggest_loguniform('ent_coef', 1e-4, 1e-2)  # 探索熵损失系数
    
    # 使用建议超参数创建新环境和模型（每个trial都从头训练，为了快速示例，我们用小步数）
    model = PPO("MlpPolicy", train_env, learning_rate=lr, gamma=gamma, ent_coef=ent_coef, verbose=0, device="cpu")
    # 训练一定步数，例如50000步，然后评估在验证集上的表现
    model.learn(total_timesteps=50000, progress_bar=True)
    # 在训练集最后一部分或专门的验证集评估策略收益
    # 这里简化直接用训练环境评估最终总净值收益
    obs = train_env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = train_env.step(action)
        total_reward += reward
    # Optuna优化目标：最大化平均每步奖励或者最终净值。我们返回每个episode总奖励作为指标
    return total_reward

# 设置Optuna研究并优化
study = optuna.create_study(direction="maximize")
study.optimize(optimize_agent, n_trials=20)
print("最佳超参数:", study.best_params)


# 初始化 PPO 模型
# model = PPO(policy="MlpPolicy", env=train_env, 
#             learning_rate=3e-4,    # 初始学习率，可调
#             gamma=0.99,            # 折扣因子
#             verbose=1,
#             tensorboard_log="./tensorboard_logs/")  # TensorBoard 日志路径（可选）

# # 开始训练模型
# total_timesteps = 100000  # 训练步数，可根据数据量和收敛情况调整
# model.learn(total_timesteps=total_timesteps, progress_bar=True)
