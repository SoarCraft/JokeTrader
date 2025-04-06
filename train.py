from torch.distributions import Normal
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from actor_critic import ActorCritic
from trading_env import BybitTradingEnv
from tqdm import tqdm

# 加载预处理数据
data = np.load("BTCUSDT_5m_features_2023-01-01_to_2025-04-01.npz")
features = data["features"]
prices = data["prices"]
# 创建环境实例
env = BybitTradingEnv(features, prices)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActorCritic(seq_feature_dim=features.shape[1], account_dim=10, window_size=env.window_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)
writer = SummaryWriter(log_dir="logs")

# PPO超参数
gamma = 0.99
lam = 0.95
clip_param = 0.2
value_coef = 0.5
entropy_coef = 0.01
batch_size = 2048   # 每次收集2048个时间步数据
mini_batch_size = 256
ppo_epochs = 10
max_train_steps = 50000000

# 日志辅助
episode_rewards = []  # 存储每回合总奖励（可选）

state = env.reset()
# 将状态中的各部分转为张量
def state_to_tensor(state):
    hist = torch.tensor(state["history"], dtype=torch.float32, device=device).unsqueeze(0)
    acc = torch.tensor(state["account"], dtype=torch.float32, device=device).unsqueeze(0)
    return hist, acc

# 主训练循环
global_step = 0
episode = 0
while global_step < max_train_steps:
    # 初始化存储buffers
    states_history = []
    states_account = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    dones = []
    
    # 采样 rollouts
    for t in tqdm(range(batch_size), desc=f"Sampling Rollouts (Episode: {episode}, Step: {global_step})"):
        hist_tensor = torch.tensor(state["history"], dtype=torch.float32, device=device).unsqueeze(0)
        acc_tensor = torch.tensor(state["account"], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            policy_mean, value = model(hist_tensor, acc_tensor)
            value = value.cpu().item()
            # 依据高斯策略采样动作
            std = model.log_std.exp().cpu().numpy()
            mean = policy_mean.cpu().numpy().squeeze()
            dist = Normal(loc=torch.tensor(mean), scale=torch.tensor(std))
            action_tensor = dist.sample()
            log_prob_tensor = dist.log_prob(action_tensor)
            # 因为连续动作各维独立，合成总log_prob
            log_prob = log_prob_tensor.sum().cpu().item()
            action = action_tensor.cpu().numpy().squeeze()
            # Tanh限制动作范围(-1,1)
            action = np.tanh(action)
        # 与环境交互
        next_state, reward, done, info, daily_return = env.step(action)
        # 存储转移数据
        states_history.append(state["history"])
        states_account.append(state["account"])
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)
        dones.append(done)
        
        state = next_state
        global_step += 1
        if done:
            # 回合结束，记录累计奖励并重置环境
            episode_total_reward = sum(rewards[-env.day_step_count:])  # or track separately
            episode_rewards.append(episode_total_reward)
            writer.add_scalar("Episode/TotalReward", episode_total_reward, episode)
            episode += 1
            state = env.reset()
            
        if daily_return is not None:
            # 记录每日收益率
            writer.add_scalar("Metrics/DailyReturn", daily_return, global_step)

    # 将最后状态的值计算用于优势估计（bootstrap）
    hist_tensor = torch.tensor(state["history"], dtype=torch.float32, device=device).unsqueeze(0)
    acc_tensor = torch.tensor(state["account"], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _, last_value = model(hist_tensor, acc_tensor)
        last_value = last_value.cpu().item()
    # 计算GAE优势
    adv_buffer = []
    gae = 0.0
    for t in reversed(range(batch_size)):
        delta = rewards[t] + (0 if dones[t] else gamma * (last_value if t == batch_size-1 else values[t+1])) - values[t]
        gae = delta + (0 if dones[t] else gamma * lam) * gae
        adv_buffer.insert(0, gae)
        if dones[t]:
            gae = 0.0  # 如果episode结束，重置GAE累计
    
    advantages = np.array(adv_buffer, dtype=np.float32)
    # 归一化优势
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # 计算目标价值（回报）
    returns = advantages + np.array(values, dtype=np.float32)
    
    # 转换所有数据为tensor
    states_history = torch.tensor(np.array(states_history), dtype=torch.float32, device=device)
    states_account = torch.tensor(np.array(states_account), dtype=torch.float32, device=device)
    actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32, device=device)
    old_log_probs_tensor = torch.tensor(np.array(log_probs), dtype=torch.float32, device=device)
    returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
    adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
    
    # PPO多epochs更新
    for epoch in range(ppo_epochs):
        # 将batch打乱组成minibatch
        indices = np.arange(batch_size)
        np.random.shuffle(indices)
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            mb_idx = indices[start:end]
            # 取出小批量数据
            mb_hist = states_history[mb_idx].to(device)
            mb_acc = states_account[mb_idx].to(device)
            mb_actions = actions_tensor[mb_idx].to(device)
            mb_old_log_probs = old_log_probs_tensor[mb_idx].to(device)
            mb_returns = returns_tensor[mb_idx].to(device)
            mb_adv = adv_tensor[mb_idx].to(device)
            # 前向通过网络获取新策略和价值
            policy_mean, value_pred = model(mb_hist, mb_acc)
            # 计算新策略下动作的log_prob
            # 构建分布
            dist = Normal(loc=policy_mean, scale=model.log_std.exp().expand_as(policy_mean))
            mb_new_log_probs = dist.log_prob(mb_actions).sum(axis=-1)
            entropy = dist.entropy().sum(axis=-1).mean()
            # PPO策略损失
            ratio = torch.exp(mb_new_log_probs - mb_old_log_probs)  # 重要性采样比值
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            # 值函数损失
            value_loss = nn.functional.mse_loss(value_pred, mb_returns)
            # 总损失
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            # 优化更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # （可选）在每个epoch结束后，可以记录loss或在外层记录
    # 记录每次更新后的策略损失、价值损失等
    writer.add_scalar("Loss/Policy", policy_loss.item(), global_step)
    writer.add_scalar("Loss/Value", value_loss.item(), global_step)
    writer.add_scalar("Loss/Entropy", entropy.item(), global_step)
    
    writer.add_scalar("Metrics/Equity", env.equity, global_step)
    writer.add_scalar("Metrics/Leverage", env.leverage, global_step)
    writer.add_scalar("Metrics/PositionFraction", abs(env.position_margin)/env.equity if env.equity>0 else 0, global_step)

    # 记录当前训练的平均累计奖励
    if episode_rewards:
        avg_reward = np.mean(episode_rewards[-10:])  # 平均最近10回合奖励
        writer.add_scalar("Episode/AvgReward", avg_reward, global_step)

writer.close()
