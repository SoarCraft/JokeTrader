import numpy as np
import pandas as pd

def preprocess_data(df: pd.DataFrame, window_size: int = 50):
    """
    对原始K线DataFrame进行预处理，生成用于训练的numpy数组。
    返回features数组和对应的价格序列等。
    """
    # 计算价格变化百分比特征，例如收盘价的对数收益率或百分比变化
    df = df.copy()
    df['pct_change'] = df['close'].pct_change().fillna(0)  # 第一行无前值，用0填充
    # 也可以计算高低价波动百分比、成交量相对变化等特征
    df['vol_change'] = df['volume'].pct_change().fillna(0)
    # 提取需要的列作为特征，例如：pct_change, vol_change, 以及其他你想加入的指标
    feature_cols = ['pct_change', 'vol_change']
    features = df[feature_cols].to_numpy(dtype=np.float32)
    # 此外保留原始价格以便环境计算盈亏
    prices = df['close'].to_numpy(dtype=np.float32)
    # 为方便，返回一个滑动窗口特征数组，稍后环境可按索引提取
    # 例如，我们可以预先构建每个时间点对应之前window_size步的特征序列
    # 这里简单返回整个特征矩阵和价格数组，由环境在运行时索引window slice。
    return features, prices

# 使用数据文件进行预处理并保存缓存
if __name__ == "__main__":
    df = pd.read_csv("BTCUSDT_5m_2023-01-01_to_2025-04-01.csv")
    features, prices = preprocess_data(df)
    np.savez("BTCUSDT_5m_features_2023-01-01_to_2025-04-01.npz", features=features, prices=prices)
