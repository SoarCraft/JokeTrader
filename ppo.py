
import multiprocessing as mp
mp.set_start_method("forkserver", force=True)

import math
from typing import Any, Dict
from datetime import datetime
from pathlib import Path
from typing import Tuple
from tqdm.notebook import tqdm

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from pybit.unified_trading import HTTP

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)


TOTAL_TIMESTEPS  = 50_000_000          # ⇦ reduce for a faster test
ROLLOUT_STEPS    = 16384               # rollout length per update
N_EPOCHS         = 10                  # PPO optimisation epochs per update
GAMMA            = 0.999               # discount factor (bitcoin dataset is 1‑min bars)
GAE_LAMBDA       = 0.95                # GAE parameter λ
CLIP_RANGE       = 0.2
ENT_COEF         = 0.00
VF_COEF          = 0.5
MAX_GRAD_NORM    = 0.5
LEARNING_RATE    = 3e-4
TENSORBOARD_DIR  = "runs/ppo_bitcoin"
N_ENVS           = 80                  # number of parallel environments

# ---- early stopping ----
EVAL_FREQ                 = 50000          # env steps between evaluations
MAX_NO_IMPROVEMENT_EVALS  = 50             # patience: 50 consecutive evals
MIN_EVALS_BEFORE_STOP     = 50             # burn‑in (same as patience)


def _datetime_to_ms(dt: str | datetime) -> int:
    ts = pd.Timestamp(dt, tz="UTC") if isinstance(dt, str) else pd.Timestamp(dt).tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _fetch_ohlcv_minute(session: HTTP, symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> pd.DataFrame:
    rows: list[list] = []
    cur = start_ms
    step_ms = 60_000 # 1 minute in milliseconds
    total_minutes = (end_ms - start_ms) // step_ms

    with tqdm(total=total_minutes, desc=f"Fetching {symbol} OHLCV") as pbar:
        while cur < end_ms:
            start_time_current_call = cur
            resp = session.get_kline(category="linear", symbol=symbol, interval="1", start=cur, limit=limit)
            data = resp["result"]["list"]
            if not data:
                # If no data is returned, assume we reached the end for the requested period
                # Update progress bar to reflect the remaining time as processed
                remaining_minutes = (end_ms - cur) // step_ms
                pbar.update(remaining_minutes)
                break

            rows.extend(data)
            last_ts = int(data[-1][0])

            # Calculate minutes fetched in this call and update progress bar
            minutes_fetched = (last_ts - start_time_current_call + step_ms) // step_ms
            pbar.update(minutes_fetched)

            # Set cursor for the next iteration
            cur = last_ts + step_ms

            # Ensure progress doesn't exceed total if API returns data beyond end_ms
            if pbar.n > total_minutes:
                 pbar.n = total_minutes
                 pbar.refresh()

        # Ensure the progress bar completes if the loop finishes early
        if pbar.n < total_minutes:
             pbar.update(total_minutes - pbar.n)


    df = pd.DataFrame(rows, columns=["startTime", "open", "high", "low", "close", "volume", "turnover"])
    df["startTime"] = pd.to_datetime(df["startTime"], unit="ms", utc=True)
    df.set_index("startTime", inplace=True)
    # Filter data strictly within the requested range [start_ms, end_ms)
    df = df[(df.index >= pd.to_datetime(start_ms, unit='ms', utc=True)) & (df.index < pd.to_datetime(end_ms, unit='ms', utc=True))]
    df = df.astype(float)[["open", "high", "low", "close", "volume"]]
    return df.sort_index()


def _fetch_long_short_ratio(session: HTTP, symbol: str, start_ms: int, end_ms: int) -> pd.Series:
    rows = []
    limit = 500
    interval_ms = 5 * 60_000 # 5 minutes in milliseconds
    total_intervals = (end_ms - start_ms) // interval_ms
    last_fetched_ts = start_ms

    with tqdm(total=total_intervals, desc=f"Fetching {symbol} Long/Short Ratio") as pbar:
        while True:
            start_time_current_call = last_fetched_ts
            try:
                resp = session.get_long_short_ratio(
                    category="linear",
                    symbol=symbol,
                    period="5min",
                    startTime=start_time_current_call, # Use last fetched timestamp to avoid overlap issues
                    endTime=end_ms,
                    limit=limit
                )
                data = resp["result"]["list"]
                if not data:
                    # No more data in the range for this call
                    remaining_intervals = max(0, (end_ms - last_fetched_ts) // interval_ms)
                    pbar.update(remaining_intervals)
                    break # Exit loop if no data is returned

                rows.extend(data)
                current_last_ts = int(data[-1]["timestamp"])

                # Calculate intervals fetched based on time covered
                intervals_fetched = max(0, (current_last_ts - last_fetched_ts) // interval_ms)
                # Add 1 interval for the last timestamp itself if it wasn't fully covered by the division
                if (current_last_ts - last_fetched_ts) % interval_ms > 0 or intervals_fetched == 0:
                     intervals_fetched += 1


                pbar.update(intervals_fetched)
                last_fetched_ts = current_last_ts + interval_ms # Set start for next potential fetch

                # Check if we have fetched data beyond the requested end_ms
                if last_fetched_ts >= end_ms:
                     # Ensure progress bar completes if we fetched up to or beyond end_ms
                     if pbar.n < total_intervals:
                         pbar.update(total_intervals - pbar.n)
                     break

            except Exception as e:
                print(f"An error occurred: {e}")
                # Update progress bar to reflect the assumed end if an error occurs
                if pbar.n < total_intervals:
                    pbar.update(total_intervals - pbar.n)
                break # Exit loop on error

        # Ensure the progress bar completes fully if the loop finishes early
        if pbar.n < total_intervals:
             pbar.update(total_intervals - pbar.n)


    if not rows:
        # Return an empty series with the correct dtype if no data was fetched
        return pd.Series(dtype=float, name="ls_ratio")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df.sort_index()
    # Filter data strictly within the requested range [start_ms, end_ms)
    df = df[(df.index >= pd.to_datetime(start_ms, unit='ms', utc=True)) & (df.index < pd.to_datetime(end_ms, unit='ms', utc=True))]
    # Remove potential duplicates from overlapping API calls if cursor wasn't effective
    df = df[~df.index.duplicated(keep='first')]

    if df.empty:
        return pd.Series(dtype=float, name="ls_ratio")

    df[["buyRatio", "sellRatio"]] = df[["buyRatio", "sellRatio"]].astype(float)
    # Avoid division by zero if both buyRatio and sellRatio are 0
    total_ratio = df["buyRatio"] + df["sellRatio"]
    ls_ratio = df["buyRatio"].divide(total_ratio).fillna(0.5) # Fill NaN with 0.5 (neutral) or 0

    # Resample to 1 minute and forward fill
    ls_ratio = ls_ratio.resample("1min").ffill()
    # Ensure the resampled series covers the full requested range, padding with ffill/bfill
    full_range_index = pd.date_range(start=pd.to_datetime(start_ms, unit='ms', utc=True),
                                     end=pd.to_datetime(end_ms - 1, unit='ms', utc=True), # end is exclusive
                                     freq='1min')
    ls_ratio = ls_ratio.reindex(full_range_index).ffill().bfill() # Forward fill then backfill NaNs

    return ls_ratio.rename("ls_ratio")


def _fetch_funding_rate(session: HTTP, symbol: str, start_ms: int, end_ms: int) -> pd.Series:
    rows = []
    cursor = None
    limit = 200 # Max limit for funding rate history
    # Estimate total intervals for progress bar (funding typically every 8 hours)
    interval_ms = 8 * 60 * 60_000
    total_intervals = max(1, (end_ms - start_ms) // interval_ms)
    last_fetched_ts = start_ms # Track the timestamp of the last fetched record for progress update

    with tqdm(total=total_intervals, desc=f"Fetching {symbol} Funding Rate") as pbar:
        while True:
            try:
                resp = session.get_funding_rate_history(
                    category="linear",
                    symbol=symbol,
                    # Rely primarily on cursor for pagination, filter by time later
                    limit=limit,
                    cursor=cursor
                )

                data = resp["result"]["list"]
                if not data:
                    # No more data from API for this cursor
                    if pbar.n < total_intervals:
                        pbar.update(total_intervals - pbar.n) # Complete the bar
                    break

                rows.extend(data)
                current_last_ts = int(data[-1]["fundingRateTimestamp"])

                # Update progress based on time covered since last fetch
                if current_last_ts > last_fetched_ts:
                    intervals_covered = (current_last_ts - last_fetched_ts) // interval_ms
                    # Ensure at least 1 interval is credited if any time passed and data received
                    if intervals_covered == 0 and current_last_ts > last_fetched_ts:
                         intervals_covered = 1
                    # Cap update to not exceed total
                    update_amount = min(intervals_covered, total_intervals - pbar.n)
                    if update_amount > 0:
                        pbar.update(update_amount)
                    last_fetched_ts = current_last_ts # Update last fetched timestamp

                cursor = resp["result"].get("nextPageCursor")
                if not cursor:
                    # No next page cursor means we are done fetching
                    if pbar.n < total_intervals:
                        pbar.update(total_intervals - pbar.n) # Complete the bar
                    break

            except Exception as e:
                print(f"An error occurred during funding rate fetch: {e}")
                # Update progress bar to reflect the assumed end if an error occurs
                if pbar.n < total_intervals:
                    pbar.update(total_intervals - pbar.n)
                break # Exit loop on error

        # Ensure the progress bar completes fully if the loop finished early
        if pbar.n < total_intervals:
             pbar.update(total_intervals - pbar.n)

    if not rows:
        # Return an empty series with the correct dtype and name if no data was fetched
        return pd.Series(dtype=float, name="fundingRate")

    df = pd.DataFrame(rows)
    df["fundingRateTimestamp"] = pd.to_datetime(df["fundingRateTimestamp"].astype(int), unit="ms", utc=True)
    df.set_index("fundingRateTimestamp", inplace=True)
    df = df.sort_index()

    # Filter data strictly within the requested range [start_ms, end_ms) AFTER collecting all data
    df = df[(df.index >= pd.to_datetime(start_ms, unit='ms', utc=True)) & (df.index < pd.to_datetime(end_ms, unit='ms', utc=True))]

    if df.empty:
        return pd.Series(dtype=float, name="fundingRate")

    # Remove potential duplicates just in case (e.g., overlapping calls if cursor logic had issues)
    df = df[~df.index.duplicated(keep='first')]

    df["fundingRate"] = df["fundingRate"].astype(float)

    # Resample to 1 minute and interpolate linearly
    funding_series = df["fundingRate"].resample("1min").interpolate(method='linear')

    # Ensure the resampled series covers the full requested range, padding with ffill/bfill
    full_range_index = pd.date_range(start=pd.to_datetime(start_ms, unit='ms', utc=True),
                                     end=pd.to_datetime(end_ms - 1, unit='ms', utc=True), # end is exclusive
                                     freq='1min')
    # Reindex to the full range, then fill any remaining NaNs at the beginning/end
    # Interpolation handles NaNs between points, ffill/bfill handle edges.
    funding_series = funding_series.reindex(full_range_index).ffill().bfill()

    return funding_series.rename("fundingRate") # Ensure series name is set


def fetch_bybit_data(
    symbol: str = "BTCUSDT",
    start: str | datetime = "2025-03-01 00:00:00",
    end: str | datetime = "2025-04-01 00:00:00",
    save_csv: bool = False,
    out_dir: str | Path = "data",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    import numpy as np
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / f"{symbol}_bybit_data.npz"

    # ---------- Try loading from .npz ----------
    if npz_path.exists():
        print(f"Loading data from NPZ file: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)

        ohlcv_cols = data["ohlcv_columns"]
        ohlcv_idx = pd.to_datetime(data["ohlcv_index"])
        ohlcv = pd.DataFrame(data["ohlcv_values"], columns=ohlcv_cols)
        ohlcv.index = ohlcv_idx
        ohlcv.index.name = "startTime"
        ohlcv = ohlcv.astype(float).asfreq("1min")

        lsr = pd.Series(data["lsr_values"], index=pd.to_datetime(data["lsr_index"]), name="ls_ratio").asfreq("1min")
        funding = pd.Series(data["funding_values"], index=pd.to_datetime(data["funding_index"]), name="fundingRate").asfreq("1min")

        return ohlcv, lsr, funding

    # ---------- Fetch from API ----------
    print("Fetching data from Bybit API...")
    session = HTTP(testnet=False)
    start_ms, end_ms = _datetime_to_ms(start), _datetime_to_ms(end)
    ohlcv = _fetch_ohlcv_minute(session, symbol, start_ms, end_ms)
    lsr = _fetch_long_short_ratio(session, symbol, start_ms, end_ms)
    funding = _fetch_funding_rate(session, symbol, start_ms, end_ms)

    # ---------- Save to npz ----------
    print(f"Saving data to NPZ file: {npz_path}")
    np.savez_compressed(
        npz_path,
        ohlcv_values=ohlcv.to_numpy(),
        ohlcv_columns=np.array(ohlcv.columns),
        ohlcv_index=ohlcv.index.astype(np.int64),
        lsr_values=lsr.to_numpy(),
        lsr_index=lsr.index.astype(np.int64),
        funding_values=funding.to_numpy(),
        funding_index=funding.index.astype(np.int64),
    )

    # ---------- Optionally save to CSV ----------
    if save_csv:
        print(f"Also saving CSV to {out_dir}...")
        ohlcv.to_csv(out / f"{symbol}_ohlcv_1min.csv")
        lsr.to_csv(out / f"{symbol}_long_short_ratio.csv", header=True)
        funding.to_csv(out / f"{symbol}_funding_rate.csv", header=True)

    return ohlcv, lsr, funding


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = ema(df["close"], fast)
    ema_slow = ema(df["close"], slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


def connors_rsi(df: pd.DataFrame, rsi_period: int = 3, streak_rsi_period: int = 2, pct_rank_period: int = 100) -> pd.Series:
    close = df["close"]
    # (1) 价格 RSI
    rsi_cl = rsi(close, rsi_period)
    # (2) 连涨/跌天数
    streak = np.sign(close.diff()).fillna(0)
    streak = streak.groupby((streak != streak.shift()).cumsum()).cumsum()
    rsi_streak = rsi(streak, streak_rsi_period)
    # (3) 当日涨跌幅在过去 n 日百分位
    pct_change = close.pct_change().fillna(0)
    pct_rank = pct_change.rolling(pct_rank_period).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    # CRSI = 上述三者平均
    crsi = (rsi_cl + rsi_streak + pct_rank) / 3.0
    return crsi


def support_resistance(df: pd.DataFrame, lookback: int = 60) -> Tuple[pd.Series, pd.Series]:
    """返回 (support, resistance) 支撑 / 压力位"""
    rolling_low = df["low"].rolling(lookback).min()
    rolling_high = df["high"].rolling(lookback).max()
    return rolling_low, rolling_high


class BitcoinFuturesEnv(gym.Env):
    """BTC 永续合约环境（线性、USDT 计价）"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        ohlcv: pd.DataFrame,
        long_short_ratio: pd.Series,
        funding_rate: pd.Series,
        window_size: int = 60,
        initial_balance: float = 1_000_000.0,
        fee_rate: float = 0.00044,
        leverage: float = 10.0,
        maintenance_margin_ratio: float = 0.005,
        random_start: bool = True,
    ):
        super().__init__()
        # 输入长度检查
        assert ohlcv.index.freq == "1min", \
            f"OHLCV 必须是 1 分钟频率的 DataFrame, 当前 freq={ohlcv.index.freq}"
        assert len(ohlcv) == len(long_short_ratio) == len(funding_rate), \
            "OHLCV、long_short_ratio 和 funding_rate 必须等长"
        # 指标窗口大小检查
        assert window_size >= 50, f"window_size 必须 >= {50} 才能计算技术指标"

        self.ohlcv = ohlcv.reset_index(drop=False)
        self.long_short_ratio = long_short_ratio.reset_index(drop=True)
        self.funding_rate = funding_rate.reset_index(drop=True)

        self.window_size = window_size
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.leverage_setting = leverage
        self.maintenance_margin_ratio = maintenance_margin_ratio
        self.random_start = random_start
        # self.max_episode_minutes = 21 * 24 * 60  # days in minutes
        self._step_counter = 0

        # ===== 预先计算技术指标 =====
        self._precompute_indicators()

        # ===== Gym spaces =====
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        obs_dim = (
            window_size * 14  # OHLC + Volume + 9个技术指标
            + 5  # position info & 可用余额 etc.
            + 2  # 资金费率和多空比例
        )
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # 内部状态
        self._reset_account()
        self._ptr: int = self.window_size  # 数据指针

    def _precompute_indicators(self):
        """一次性计算并存储所有技术指标，填充缺失值"""
        df = self.ohlcv.copy()
        # 计算指标
        df['sma_fast'] = sma(df['close'], 20)
        df['sma_slow'] = sma(df['close'], 50)
        df['ema'] = ema(df['close'], 20)
        macd_df = macd(df)
        df = pd.concat([df, macd_df], axis=1)
        df['crsi'] = connors_rsi(df)
        support, resistance = support_resistance(df)
        df['support'] = support
        df['resistance'] = resistance

        features = [
            'open','high','low','close','volume',
            'sma_fast','sma_slow','ema',
            'macd','macd_signal','macd_hist',
            'crsi','support','resistance'
        ]
        df = df[features].ffill().fillna(0.0)
        # 转为 NumPy 加速切片
        self.tech_all = df.to_numpy(dtype=np.float32)

    # ---------------------------------
    # 重置 / 步进
    # ---------------------------------
    def reset(self, *, seed: int | None = None):
        super().reset(seed=seed)
        self._reset_account()
        self._step_counter = 0
        if self.random_start:
            self._ptr = self.np_random.integers(self.window_size, len(self.tech_all) / 2)
        else:
            self._ptr = self.window_size
        obs = self._get_observation()
        return obs, {}

    def step(self, action: np.ndarray):
        """执行一步，action ∈ [-1,1]."""
        action_val = float(action[0])
        price = self._current_price()

        # === 强平检查（先于资金费） ===
        self._check_liquidation(price)
        # === 资金费处理 ===
        self._apply_funding(price)

        # === 解析动作 ===
        if abs(action_val) > 0.01:
            if self.position_size == 0:
                # 开新仓
                self._open_position(action_val, price)
            else:
                same_direction = (self.position_size > 0 and action_val > 0) or (
                    self.position_size < 0 and action_val < 0
                )
                if same_direction:
                    # 加仓
                    self._add_position(action_val, price)
                else:
                    # 减仓或反向 → 先平部分 / 全平
                    self._reduce_or_close(action_val, price)

        # === 时间向前推进 ===
        self._ptr += 1
        self._step_counter += 1

        # 到数据末尾算作truncate（时间用尽）
        # truncated = self._ptr >= len(self.tech_all) - 1 or self._step_counter >= self.max_episode_minutes
        truncated = self._ptr >= len(self.tech_all) - 1
        # 余额为零或负数算作terminated（破产）
        terminated = self.balance <= 0

        obs = self._get_observation()
        reward = self.realized_pnl + (self._unrealized_pnl(price) - self._last_unrealized_pnl)  # 差分奖励
        self._last_unrealized_pnl = self._unrealized_pnl(price)
        self.realized_pnl = 0.0  # 清零，避免下轮重复

        info = (
            {
                "equity": self.balance + self._unrealized_pnl(price),
                "position_size": self.position_size,
                "entry_price": self.entry_price,
                "unrealized_pnl": self._unrealized_pnl(price),
            }
        )

        if terminated:
            info["termination_reason"] = "bankrupt"
        elif truncated:
            info["termination_reason"] = "time_limit"

        return obs, reward, terminated, truncated, info

    # ---------------------------------
    # 账户逻辑
    # ---------------------------------
    def _reset_account(self):
        self._last_unrealized_pnl = 0.0  # 初始化追踪浮动收益差分
        self.balance: float = self.initial_balance  # 可用余额 / Equity
        self.position_size: float = 0.0  # >0 long <0 short (张数 BTC)
        self.entry_price: float = 0.0
        self.realized_pnl: float = 0.0

    def _apply_fee(self, notional: float):
        fee = abs(notional) * self.fee_rate
        self.balance -= fee
        self.realized_pnl -= fee

    def _open_position(self, action_val: float, price: float):
        notional = self.balance * abs(action_val) * self.leverage_setting
        qty = notional / price
        self.position_size = qty if action_val > 0 else -qty
        self.entry_price = price
        margin = notional / self.leverage_setting
        self.balance -= margin
        self._apply_fee(notional)

    def _add_position(self, action_val: float, price: float):
        notional = self.balance * abs(action_val) * self.leverage_setting
        add_qty = notional / price
        total_notional = abs(self.position_size) * self.entry_price + notional
        new_size = self.position_size + (add_qty if action_val > 0 else -add_qty)
        self.entry_price = total_notional / abs(new_size)
        self.position_size = new_size
        self.balance -= notional / self.leverage_setting
        self._apply_fee(notional)

    def _reduce_or_close(self, action_val: float, price: float):
        """
        1. 计算目标 notional = 可用余额 * |action_val| * 杠杆
        2. 如果目标 notional 小于当前持仓 notional，则按比例平仓
        3. 如果目标 notional >= 当前持仓 notional，则先全平再按剩余 notional 反向开仓
        """
        # 目标 notional（USD）
        notional_to_close = self.balance * abs(action_val) * self.leverage_setting
        # 当前持仓 notional（USD）
        current_notional = abs(self.position_size) * price

        # 部分平仓
        if notional_to_close < current_notional:
            # 计算需平仓数量（BTC）
            close_qty = (notional_to_close / price)

            # 结算 PnL
            pnl = close_qty * (price - self.entry_price) * (1 if self.position_size > 0 else -1)
            self.realized_pnl += pnl

            # 退回保证金 + 盈亏
            self.balance += close_qty * self.entry_price / self.leverage_setting + pnl
            self._apply_fee(notional_to_close)

            # 更新剩余仓位
            remain_qty = abs(self.position_size) - close_qty
            if remain_qty <= 0:
                self.position_size = 0.0
                self.entry_price = 0.0
            else:
                self.position_size = math.copysign(remain_qty, self.position_size)

        # 全平并反向开仓
        else:
            # --- 1) 平掉所有现有仓位 ---
            # 既有 notional = current_notional
            pnl = abs(self.position_size) * (price - self.entry_price) * (1 if self.position_size > 0 else -1)
            self.realized_pnl += pnl
            initial_margin = abs(self.position_size) * self.entry_price / self.leverage_setting
            self.balance += initial_margin + pnl
            self._apply_fee(current_notional)
 
            # 清空仓位
            self.position_size = 0.0
            self.entry_price = 0.0

            # --- 2) 剩余 notional 用于反向开仓 ---
            reverse_notional = self.balance * abs(action_val) * self.leverage_setting
            if reverse_notional > 0:
                qty = reverse_notional / price
                self.position_size = qty if action_val > 0 else -qty
                self.entry_price = price
                self.balance -= reverse_notional / self.leverage_setting
                self._apply_fee(reverse_notional)

    def _apply_funding(self, price: float):
        """按分钟线性插值资金费，收取到/付出 Equity"""
        current_funding = self._current_funding()
        notional = abs(self.position_size) * price
        funding_payment = notional * current_funding / (8 * 60)  # 每分钟份额
        # long 支付正 funding，short 获得
        self.balance -= funding_payment * np.sign(self.position_size)
        self.realized_pnl -= funding_payment * np.sign(self.position_size)

    def _unrealized_pnl(self, price: float) -> float:
        return abs(self.position_size) * (price - self.entry_price) * (
            1 if self.position_size > 0 else -1
        )

    def _check_liquidation(self, price: float):
        if self.position_size == 0:
            return
        notional = abs(self.position_size) * price
        equity = self.balance + self._unrealized_pnl(price)
        # 维护保证金按名义价值比例计算
        if equity < notional * self.maintenance_margin_ratio:
            # 强平：损失所有保证金
            self.realized_pnl -= notional / self.leverage_setting
            self.balance = equity
            self.position_size = 0.0
            self.entry_price = 0.0

    # ---------------------------------
    # Observation & Helpers
    # ---------------------------------
    def _current_price(self) -> float:
        return float(self.ohlcv.iloc[self._ptr]["close"])

    def _current_funding(self) -> float:
        return float(self.funding_rate.iloc[self._ptr])

    def _current_long_short_ratio(self) -> float:
        prev_val = self.long_short_ratio.iloc[self._ptr - 1]
        next_val = self.long_short_ratio.iloc[self._ptr]
        return float(self.np_random.uniform(min(prev_val, next_val), max(prev_val, next_val)))

    def _get_observation(self) -> np.ndarray:
        start = self._ptr - self.window_size
        tech_np = self.tech_all[start:self._ptr].flatten()
        price = float(self.ohlcv.iloc[self._ptr]['close'])
        pos_dir = 0.0 if self.position_size == 0 else math.copysign(1, self.position_size)

        account_state = np.array([
            self.balance,
            self.position_size,
            self.entry_price,
            pos_dir,
            self._unrealized_pnl(price)
        ], dtype=np.float32)

        obs = np.concatenate([
            tech_np,
            account_state,
            np.array([
                float(self.funding_rate.iloc[self._ptr]),
                self._current_long_short_ratio()
            ], dtype=np.float32)
        ])

        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            raise ValueError("Observation contains inf or nan!")
        return obs

    # ---------------------------------
    # Render / Close
    # ---------------------------------
    def render(self):
        price = self._current_price()
        print(
            f"t={self._ptr} | price={price:.2f} | bal={self.balance:.2f} | pos={self.position_size:.4f} @ {self.entry_price:.2f} | unreal={self._unrealized_pnl(price):+.2f}"
        )

    def close(self):
        pass


def make_env() -> BitcoinFuturesEnv:  # type: ignore[name-defined]
    return BitcoinFuturesEnv(
            ohlcv_df,
            ls_series,
            funding_series,
            random_start=True,
            leverage=1
        )


def backtest_ppo(env: BitcoinFuturesEnv, agent: PPO) -> Dict[str, Any]:  # type: ignore[name-defined]
    obs, _ = env.reset(seed=0)
    equity_curve = [env.balance]
    trade_profits = []

    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        equity_curve.append(info["equity"])
        if reward != 0.0:
            trade_profits.append(reward)

    equity_series = pd.Series(equity_curve)
    init_eq, final_eq = equity_series.iloc[0], equity_series.iloc[-1]
    total_return = (final_eq - init_eq) / init_eq

    mins_per_year = 365 * 24 * 60
    annual_ret = (1 + total_return) ** (mins_per_year / len(equity_series)) - 1

    ret_series = equity_series.pct_change().fillna(0)
    sharpe = np.sqrt(mins_per_year) * ret_series.mean() / (ret_series.std() + 1e-12)

    profits = np.sum([p for p in trade_profits if p > 0])
    losses  = -np.sum([p for p in trade_profits if p < 0])
    win_rate = np.mean(np.array(trade_profits) > 0) if trade_profits else 0.0
    profit_factor = profits / losses if losses > 0 else float("inf")

    drawdown = (equity_series - equity_series.cummax()) / equity_series.cummax()
    max_dd = drawdown.min()

    return {
        "total_return":       total_return,
        "annualized_return":  annual_ret,
        "win_rate":           win_rate,
        "profit_factor":      profit_factor,
        "max_drawdown":       max_dd,
        "sharpe_ratio":       sharpe,
        "n_trades":           len(trade_profits),
        "n_steps":            len(equity_series) - 1,
    }

if __name__ == "__main__":
    ohlcv_df, ls_series, funding_series = fetch_bybit_data(save_csv=True)
    
    train_env = VecMonitor(SubprocVecEnv([make_env for _ in range(N_ENVS)]))
    eval_env  = VecMonitor(DummyVecEnv([make_env]))

    best_model_path = Path("./ppo_ckpts/best_model.zip")
    
    if best_model_path.exists():
        print("✅ Found previous best model – resuming training from it.")
        model = PPO.load(
            best_model_path,
            env=train_env,
            tensorboard_log=TENSORBOARD_DIR,
            # device="cpu",
        )
    else:
        print("🚀 No previous model found – starting fresh training.")
        policy_kwargs = dict(
            net_arch=dict(
                pi=[256, 64],
                vf=[256, 64]
            ),
            activation_fn=torch.nn.SiLU,        # swish-like activation (better for value flow)
            ortho_init=True,
            log_std_init=-0.5,
        )
    
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            n_steps=ROLLOUT_STEPS,
            batch_size=64,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,
            vf_coef=VF_COEF,
            max_grad_norm=MAX_GRAD_NORM,
            learning_rate=LEARNING_RATE,
            tensorboard_log=TENSORBOARD_DIR,
            verbose=1,
            # device="cpu",
            # policy_kwargs=policy_kwargs,
        )

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=MAX_NO_IMPROVEMENT_EVALS,
        min_evals=MIN_EVALS_BEFORE_STOP,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=max(EVAL_FREQ // N_ENVS, 100),
        best_model_save_path="./ppo_ckpts",  # only best model is saved
        log_path="./ppo_eval_logs",
        deterministic=False,
        render=False,
        callback_after_eval=stop_callback,
    )

    print("\n▶️  Start PPO training – early stopping (patience 50) …\n")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True,
    )
    model.save("ppo_bitcoin_final")
    print("\n✓ Training finished (steps exhausted or early‑stopped). Model saved as `ppo_bitcoin_final.zip`.\n")

    print("▶️  Running back‑test …")

    env_eval = BitcoinFuturesEnv(
        ohlcv_df,
        ls_series,
        funding_series,
        random_start=False,
    )
    metrics = backtest_ppo(env_eval, model)

    env_eval.close()

    print("\n===========  Back‑test metrics  ===========")
    for k, v in metrics.items():
        if k in {"win_rate", "sharpe_ratio"}:
            print(f"{k:18s}: {v:.4f}")
        else:
            print(f"{k:18s}: {v:.4%}")
    print("===========================================\n")
