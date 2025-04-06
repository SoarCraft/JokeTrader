import gym
import numpy as np

class BybitTradingEnv(gym.Env):
    def __init__(self, features: np.ndarray, prices: np.ndarray, window_size: int = 50, 
                 initial_balance: float = 10000.0, maintenance_margin_rate: float = 0.005):
        """
        features: 预处理后的特征序列数组 (shape: [T, feature_dim]) 
        prices: 收盘价序列 (shape: [T])
        window_size: 状态包含的历史序列长度
        initial_balance: 初始账户余额 (USDT)
        maintenance_margin_rate: 维持保证金率 (用于计算爆仓价)
        """
        super(BybitTradingEnv, self).__init__()
        self.features = features
        self.prices = prices
        self.window_size = window_size
        self.maintenance_margin_rate = maintenance_margin_rate
        self.data_length = len(prices)
        # 动作空间: 连续2维 [position_fraction, leverage_factor]
        # position_fraction范围[-1,1], leverage_factor范围[-1,1]对应实际1x到25x
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        # 状态空间: 包括最近window_size步的特征序列 + 账户指标（我们用一个固定长度的一维数组表示账户状态）
        # 行情序列可以作为高维输入，由我们自定义的神经网络处理，因此这里Observation空间用object或Dict
        obs_space_low  = np.finfo(np.float32).min
        obs_space_high = np.finfo(np.float32).max
        # 我们将状态定义为包含两部分: "history"序列特征 和 "account"账户状态
        self.observation_space = gym.spaces.Dict({
            "history": gym.spaces.Box(low=obs_space_low, high=obs_space_high, shape=(window_size, features.shape[1]), dtype=np.float32),
            "account": gym.spaces.Box(low=obs_space_low, high=obs_space_high, shape=(10,), dtype=np.float32)
        })
        # 初始化账户状态
        self.initial_balance = initial_balance
        self.reset()
        
    def reset(self):
        # 随机选择episode开始的索引，以增加训练多样性 (确保有window_size历史)
        self.current_step = np.random.randint(self.window_size, self.data_length - 1)
        # 账户初始状态
        self.balance = self.initial_balance    # 可用余额（初始时=总权益）
        self.equity = self.initial_balance     # 总权益（balance + 当前持仓未实现盈亏）
        self.position_size = 0.0               # 持仓数量（正表示多头BTC数量，负表示空头）
        self.position_entry_price = None       # 持仓开仓价
        self.position_margin = 0.0             # 持仓占用保证金
        self.leverage = 1.0                    # 当前杠杆（默认为1倍无杠杆）
        self.unrealized_pnl = 0.0              # 当前未实现盈亏
        # 日结算跟踪
        self.day_step_count = 0
        self.equity_start_of_day = self.equity
        self.max_daily_utilization = 0.0   # 当日最大资金利用率（持仓保证金/权益）
        # 构建初始观察
        return self._get_observation()
    
    def _get_observation(self):
        # 提取最近window_size步的特征作为history
        hist_start = self.current_step - self.window_size + 1
        history = self.features[hist_start: self.current_step+1]
        # 如果长度不够(window_size可能在episode开头)，则用开头数据填充
        if len(history) < self.window_size:
            pad_len = self.window_size - len(history)
            history = np.vstack((np.tile(history[0], (pad_len, 1)), history))
        # 账户状态信息 account_features 列举如下（共10个指标作为示例）:
        # [当前仓位方向(1多/-1空/0无), 当前持仓规模(仓位价值/账户权益), 当前杠杆, 未实现盈亏比例(未实现盈亏/权益),
        #  当日盈亏比例(当前权益/日初权益 - 1), 可用余额比例(可用余额/权益), 爆仓价相对当前价的比率(如(当前价-爆仓价)/当前价),
        #  保证金率(持仓保证金/权益), 当前时间步(当日0~1之间), ... 保留额外槽位以扩展]
        pos_dir = 0.0
        pos_value_ratio = 0.0
        liq_price_ratio = 0.0
        if self.position_size != 0:
            pos_dir = 1.0 if self.position_size > 0 else -1.0
            # 仓位名义价值 = 持仓数量 * 当前价
            pos_notional = abs(self.position_size) * self.prices[self.current_step]
            pos_value_ratio = pos_notional / self.equity  # 仓位名义价值与权益比
            # 计算爆仓价与当前价距离比率
            # 维持保证金
            maintenance_margin = pos_notional * self.maintenance_margin_rate
            if pos_dir == 1:  # 多头
                # LiquidationPrice = entry_price - (margin - maintenance_margin)/position_size
                liq_price = self.position_entry_price - (self.position_margin - maintenance_margin) / self.position_size
            else:  # 空头
                # LiquidationPrice = entry_price + (margin - maintenance_margin)/|position_size|
                liq_price = self.position_entry_price + (self.position_margin - maintenance_margin) / abs(self.position_size)
            # 若当前价低于爆仓价(多头)或高于爆仓价(空头)，说明已经爆仓，liq_price_ratio可以设置为0或负
            if pos_dir == 1:
                liq_price_ratio = (self.prices[self.current_step] - liq_price) / self.prices[self.current_step]
            else:
                liq_price_ratio = (liq_price - self.prices[self.current_step]) / self.prices[self.current_step]
            # 防止出现极端值
            liq_price_ratio = max(-1.0, liq_price_ratio)
        # 未实现盈亏比例
        unrealized_pnl_ratio = 0.0
        if self.equity > 0:
            unrealized_pnl_ratio = self.unrealized_pnl / self.equity
        # 当日盈亏比例
        daily_return = 0.0
        if self.equity_start_of_day > 0:
            daily_return = (self.equity / self.equity_start_of_day) - 1.0
        # 可用余额比例
        available_balance_ratio = self.balance / self.equity if self.equity > 0 else 0.0
        # 保证金率（持仓保证金 / 当前权益）
        margin_ratio = self.position_margin / self.equity if self.equity > 0 else 0.0
        # 时间（归一化到0-1表示一天中的位置）
        time_of_day = self.day_step_count / 288.0  # 288个5min步长为一整天
        account_features = np.array([
            pos_dir,
            pos_value_ratio,
            self.leverage,
            unrealized_pnl_ratio,
            daily_return,
            available_balance_ratio,
            liq_price_ratio,
            margin_ratio,
            time_of_day,
            0.0  # 预留扩展
        ], dtype=np.float32)
        return {"history": history.astype(np.float32), "account": account_features}
    
    def step(self, action):
        # 将动作值转换为目标仓位比例和杠杆
        frac = float(np.clip(action[0], -1.0, 1.0))   # 目标仓位比例 (-1~1)
        lever_factor = float(np.clip(action[1], -1.0, 1.0))
        # 将lever_factor从[-1,1]映射到实际杠杆[1,25]
        target_leverage = 1.0 + (lever_factor + 1.0) * 12.0  # -1->1x, +1->25x (区间长度25-1=24，对应lever_factor 2)
        # 四舍五入到最近的0.1倍(比如可以限定杠杆增量)
        target_leverage = round(target_leverage, 1)
        if target_leverage < 1.0: 
            target_leverage = 1.0
        if target_leverage > 25.0:
            target_leverage = 25.0
        # 记录最终将应用的目标仓位和杠杆
        target_fraction = frac
        target_dir = 0 if abs(target_fraction) < 1e-6 else (1 if target_fraction > 0 else -1)
        
        reward = 0.0
        info = {}
        # 计算当前价格
        price = self.prices[self.current_step]
        
        # 更新未实现盈亏（基于上一步持仓，行情推进一个时间步后的盈亏计算）
        if self.position_size != 0:
            # 根据最新价格更新未实现盈亏
            # PnL = 持仓数量 * (当前价 - 开仓价) （多头正，空头负方向在数量的符号中体现）
            self.unrealized_pnl = self.position_size * (price - self.position_entry_price)
            # 如果触发爆仓条件（亏损超出保证金）
            if (self.position_size > 0 and price <= info.get("liq_price", -np.inf)) or \
               (self.position_size < 0 and price >= info.get("liq_price", np.inf)):
                # 触发爆仓
                loss_amount = self.position_margin  # 爆仓损失相当于损失掉初始保证金（忽略额外扣除）
                self.equity -= loss_amount
                reward -= loss_amount * 2  # 爆仓惩罚：亏损金额两倍
                # 清空仓位
                self.position_size = 0.0
                self.position_entry_price = None
                self.position_margin = 0.0
                self.leverage = 1.0
                self.balance = self.equity  # 剩余权益全部为可用余额
                done = True
                info['event'] = 'liquidation'
                return self._get_observation(), reward, done, info
        
        # **应用智能体的动作调整仓位**
        realized_pnl = 0.0
        # 当前是否持仓
        if self.position_size != 0:
            current_dir = 1 if self.position_size > 0 else -1
        else:
            current_dir = 0
        
        # 决定行动：
        if target_dir == 0:
            # 目标无仓位 -> 平仓
            if current_dir != 0:
                # 平掉当前仓位
                realized_pnl = self.unrealized_pnl  # 平仓实现盈亏等于未实现盈亏
                self.balance += self.position_margin  # 释放保证金回到可用余额
                # 更新权益：加上实现的盈亏
                self.equity += self.unrealized_pnl
                # 计算止盈止损奖励
                if realized_pnl > 0:
                    reward += realized_pnl  # 盈利部分奖励
                elif realized_pnl < 0:
                    reward += realized_pnl  # 亏损部分直接作为负奖励（扣分）
                # 清空仓位状态
                self.position_size = 0.0
                self.position_entry_price = None
                self.position_margin = 0.0
                self.leverage = 1.0
                self.unrealized_pnl = 0.0
        else:
            # 目标持仓不为空
            if current_dir == 0:
                # 当前无仓位 -> 直接开新仓
                # 计算可用资金和将用保证金
                margin_to_use = abs(target_fraction) * self.equity
                if margin_to_use > self.balance:
                    margin_to_use = self.balance  # 不能超出可用余额
                # 计算仓位数量 (USDT本位合约: 仓位名义价值 = margin * leverage, BTC数量 = 名义价值 / 当前价)
                pos_notional = margin_to_use * target_leverage
                pos_size = pos_notional / price  # BTC数量
                if target_dir == -1:
                    pos_size = -pos_size
                # 设置仓位
                self.position_size = pos_size
                self.position_entry_price = price
                self.position_margin = margin_to_use
                self.leverage = target_leverage
                self.balance -= margin_to_use  # 扣除用于仓位的保证金
                # 未实现盈亏初始为0
                self.unrealized_pnl = 0.0
            else:
                # 当前有仓位
                if target_dir == current_dir:
                    # 新目标与当前方向相同 -> 调整仓位规模或杠杆
                    # 计算当前仓位占权益比例
                    current_frac = self.position_margin / self.equity if self.equity > 0 else 0.0
                    # 计算目标占用保证金
                    target_margin = abs(target_fraction) * self.equity
                    if target_margin > self.balance + self.position_margin:
                        # 如果目标保证金超过当前保证金+可用余额（即想增加仓位但没有足够资金）
                        target_margin = self.position_margin + self.balance  # 最大只能用完全部资金
                    # 计算新的名义价值和仓位大小
                    new_notional = target_margin * target_leverage
                    new_pos_size = new_notional / price
                    if target_dir == -1:
                        new_pos_size = - new_pos_size
                    # 实现部分平仓或加仓的盈亏
                    if abs(new_pos_size) < abs(self.position_size):
                        # 缩减仓位（部分平仓）
                        reduce_size = self.position_size - new_pos_size
                        # 对应平仓盈亏
                        realized_pnl = reduce_size * (price - self.position_entry_price)
                        # 更新账户
                        self.balance += (self.position_margin - target_margin)  # 释放部分保证金
                        self.equity += realized_pnl
                        # 计算止盈止损奖励
                        if realized_pnl > 0:
                            reward += realized_pnl
                        elif realized_pnl < 0:
                            reward += realized_pnl
                    elif abs(new_pos_size) > abs(self.position_size):
                        # 扩大仓位（加仓），需要从余额划拨保证金
                        additional_margin = target_margin - self.position_margin
                        if additional_margin > 0:
                            self.balance -= additional_margin
                    # 更新仓位参数
                    self.position_size = new_pos_size
                    self.position_margin = target_margin
                    # 调整开仓价: 对于部分平仓或加仓，我们简单处理为采用加权平均或当前价作为新的entry（这里为了简化，直接将当前价设为新的entry）
                    self.position_entry_price = price
                    self.leverage = target_leverage
                    self.unrealized_pnl = 0.0  # 刚调整后按新entry价未实现盈亏归零
                else:
                    # 目标方向与当前相反 -> 反手操作
                    # 先平掉当前仓位
                    realized_pnl = self.unrealized_pnl
                    self.balance += self.position_margin  # 返还当前仓位保证金
                    self.equity += realized_pnl
                    if realized_pnl > 0:
                        reward += realized_pnl
                    elif realized_pnl < 0:
                        reward += realized_pnl
                    # 然后按照目标开仓
                    margin_to_use = abs(target_fraction) * self.equity
                    if margin_to_use > self.balance:
                        margin_to_use = self.balance
                    pos_notional = margin_to_use * target_leverage
                    pos_size = pos_notional / price
                    if target_dir == -1:
                        pos_size = -pos_size
                    self.position_size = pos_size
                    self.position_entry_price = price
                    self.position_margin = margin_to_use
                    self.leverage = target_leverage
                    self.balance -= margin_to_use
                    self.unrealized_pnl = 0.0
        
        # **行情推进到下一步**
        self.current_step += 1
        self.day_step_count += 1
        done = False
        # 如果到达数据末尾，则episode结束
        if self.current_step >= self.data_length - 1:
            done = True
        
        # 每日结算检查：若经过24小时(288步)或episode结束触发日结算
        if self.day_step_count >= 288 or done:
            # 计算当日收益并给与额外奖惩
            daily_return = 0.0
            if self.equity_start_of_day > 0:
                daily_return = (self.equity / self.equity_start_of_day) - 1.0
            # 根据当日收益率奖励或惩罚（额外部分）
            # 例如直接以当日收益金额作为奖励
            daily_profit = self.equity - self.equity_start_of_day
            if daily_profit > 0:
                reward += daily_profit  # 正收益再奖励同等金额
            elif daily_profit < 0:
                reward += daily_profit  # 亏损再扣除同等金额
            # 资金利用率惩罚：如果当日最大保证金率过低，进行惩罚
            if self.max_daily_utilization < 0.1:  # 阈值例如10%
                penalty = (0.1 - self.max_daily_utilization) * self.equity_start_of_day
                reward -= penalty  # 利用率不足部分按资金规模惩罚
            # 重置每日统计
            self.equity_start_of_day = self.equity
            self.day_step_count = 0
            self.max_daily_utilization = 0.0
        
        # 更新未实现盈亏（应用完动作后的新仓位基于当前价计算UPnL初始值）
        if self.position_size != 0:
            self.unrealized_pnl = self.position_size * (self.prices[self.current_step] - self.position_entry_price)
        else:
            self.unrealized_pnl = 0.0
        
        # 更新当日最大资金利用率
        current_util = self.position_margin / self.equity if self.equity > 0 else 0.0
        if current_util > self.max_daily_utilization:
            self.max_daily_utilization = current_util
        
        # 组装新的状态观测
        obs = self._get_observation()
        return obs, reward, done, info, daily_return
