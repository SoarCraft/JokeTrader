import numpy as np

from trading_env import BybitTradingEnv

# 加载预处理数据
data = np.load("BTCUSDT_5m_features_2024-01-01_to_2025-04-01.npz")
features = data["features"]
prices = data["prices"]
# 创建环境实例
env = BybitTradingEnv(features, prices)
