import pandas as pd
import time
from datetime import datetime
from pybit.unified_trading import HTTP

# Bybit API会话（不传api_key和api_secret表示使用公开数据）
session = HTTP(testnet=False)

def fetch_historical_klines(symbol: str, interval: int, start_time: str, end_time: str):
    """
    获取指定时间范围的历史K线数据（间隔为interval分钟）。
    返回Pandas DataFrame。
    """
    # 将日期字符串解析为UTC时间戳（毫秒）
    start_ts = int(datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
    end_ts   = int(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
    
    all_data = []
    current_ts = start_ts
    # Bybit每次最多返回1000条K线
    limit = 1000
    while current_ts < end_ts:
        print(f"Fetching data from {datetime.utcfromtimestamp(current_ts/1000.0)}")
        resp = session.get_kline(category="linear", symbol=symbol, interval=str(interval), start=current_ts, limit=limit)
        result = resp.get("result")
        if result is None or len(result.get("list", [])) == 0:
            break  # 没有更多数据
        candles = result["list"]
        # Bybit返回的list按时间倒序排列，我们将其按时间顺序排序
        candles.sort(key=lambda x: int(x[0]))
        for entry in candles:
            ts = int(entry[0])
            if ts > end_ts:
                break  # 超出结束时间
            # 提取所需字段
            open_price  = float(entry[1]); high_price = float(entry[2])
            low_price   = float(entry[3]); close_price= float(entry[4])
            volume      = float(entry[5])
            # 转换时间戳为日期时间
            dt = datetime.utcfromtimestamp(ts/1000.0)
            all_data.append([dt, open_price, high_price, low_price, close_price, volume])
        # 更新下一次抓取的开始时间为最后一根K线的时间再加一个间隔
        last_ts = int(candles[-1][0])
        current_ts = last_ts + interval*60*1000  # interval分钟后的时间戳（加上5分钟的毫秒数）
        # 控制请求频率，稍作延迟以避免触发API限频
        time.sleep(0.1)
        if current_ts >= end_ts:
            break

    # 将列表转换为DataFrame
    df = pd.DataFrame(all_data, columns=["datetime", "open", "high", "low", "close", "volume"])
    return df

# 使用上述函数获取2023-01-01至2025-04-01的BTCUSDT 5m数据
if __name__ == "__main__":
    df = fetch_historical_klines("BTCUSDT", interval=5, 
                                 start_time="2023-01-01 00:00:00", 
                                 end_time="2025-04-01 00:00:00")
    # 将数据缓存保存为CSV
    df.to_csv("BTCUSDT_5m_2023-01-01_to_2025-04-01.csv", index=False)
    print(f"Fetched {len(df)} rows of data")
