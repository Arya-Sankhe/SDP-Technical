function Split-Lines {
    param([string]$Text)
    $normalized = $Text -replace "`r`n", "`n"
    $lines = $normalized -split "`n", -1
    return @($lines | ForEach-Object { $_ + "`n" })
}

function New-CodeCell {
    param([string]$Text)
    return @{
        cell_type = "code"
        execution_count = $null
        metadata = @{}
        outputs = @()
        source = Split-Lines $Text
    }
}

function New-MarkdownCell {
    param([string]$Text)
    return @{
        cell_type = "markdown"
        metadata = @{}
        source = Split-Lines $Text
    }
}

$cells = @()

$cells += New-MarkdownCell @'
# FINAL.ipynb

This notebook is the inference-only side of the final workflow. It expects `FinalTrain.ipynb` to have already created `output/final_artifacts/manifest.json` plus all forecast checkpoints, scaler stats, feature manifests, ensemble weights, and PPO artifacts.

The notebook loads those frozen artifacts, fetches the latest market data, runs the five forecast experts one at a time, shows each expert prediction, builds the weighted aggregate forecast, and then runs the saved `v9.6` decision layer on top.
'@

$cells += New-CodeCell @'
import importlib
import subprocess
import sys

REQUIRED_PACKAGES = {
    "alpaca.data": "alpaca-py",
    "pandas_market_calendars": "pandas-market-calendars",
    "pyarrow": "pyarrow",
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
}


def module_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


missing = [package for module_name, package in REQUIRED_PACKAGES.items() if not module_available(module_name)]
if missing:
    print(f"Installing missing packages: {missing}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
else:
    print("Required packages already available.")

if not module_available("torch"):
    print("Installing missing package: torch")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])
else:
    print("torch already available.")
'@

$cells += New-CodeCell @'
import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import torch
import torch.nn as nn
import torch.nn.functional as F
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
'@

$cells += New-CodeCell @'
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOW_VRAM_GPU = DEVICE.type == "cuda" and (torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)) <= 8.5
RUN_LABEL = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")

SYMBOL = "MSFT"
LOOKBACK_DAYS = 120
HORIZON = 50
SESSION_TZ = "America/New_York"
SESSION_OPEN_SKIP_BARS = 6
REQUEST_CHUNK_DAYS = 5
MAX_REQUESTS_PER_MINUTE = 120
MAX_RETRIES = 5
ACTION_NEUTRAL_BAND = 0.10
SAVE_RUN_OUTPUTS = True

TARGET_COLUMNS = ["rOpen", "rHigh", "rLow", "rClose"]
CORE_FEATURE_COLUMNS = [
    "rOpen",
    "rHigh",
    "rLow",
    "rClose",
    "logVolChange",
    "logTradeCountChange",
    "vwapDelta",
    "rangeFrac",
    "orderFlowProxy",
    "tickPressure",
]
TECHNICAL_FEATURE_COLUMNS = [
    "sma_5",
    "sma_10",
    "sma_20",
    "sma_50",
    "ema_12",
    "ema_26",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "macd_momentum",
    "rsi_14",
    "rsi_14_slope",
    "stoch_k",
    "stoch_d",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "bb_position",
    "atr_14",
    "atr_14_pct",
    "obv",
    "obv_slope",
    "vwap_20",
    "vwap_20_dev",
    "price_momentum_5",
    "price_momentum_10",
    "price_momentum_20",
    "body_size",
    "body_pct",
    "upper_shadow",
    "lower_shadow",
    "direction",
]
REGIME_FEATURE_COLUMNS = [
    "atr_14",
    "atr_14_pct",
    "returns",
    "turbulence_60",
    "regime_indicator",
]
MARKET_STATE_COLUMNS = [
    "rsi_14",
    "macd_histogram",
    "bb_position",
    "atr_14_pct",
    "price_momentum_5",
    "price_momentum_20",
    "vwap_20_dev",
    "obv_slope",
    "direction",
    "relative_volume",
]
EXPECTED_EXPERTS = ["v8_5", "v9_1", "v9_2", "v9_3", "v9_5"]

ARTIFACT_ROOT = Path("output/final_artifacts")
MANIFEST_PATH = ARTIFACT_ROOT / "manifest.json"
FINAL_RUN_DIR = Path("output/final_runs") / RUN_LABEL

PORTFOLIO_STATE = {
    "cash_norm": 1.0,
    "shares_norm": 0.0,
    "portfolio_value_norm": 1.0,
    "position_pct": 0.0,
}

print(
    {
        "device": str(DEVICE),
        "low_vram_gpu": LOW_VRAM_GPU,
        "manifest_path": str(MANIFEST_PATH),
        "run_label": RUN_LABEL,
        "save_run_outputs": SAVE_RUN_OUTPUTS,
    }
)
'@

$cells += New-CodeCell @'
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def production_timestamp_utc() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(to_serializable(payload), indent=2), encoding="utf-8")


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def fail_if_missing(path: Path, message: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{message}: {path}")


def feature_columns_for_mode(mode: str) -> List[str]:
    if mode == "core":
        return CORE_FEATURE_COLUMNS.copy()
    if mode == "technical":
        return CORE_FEATURE_COLUMNS + TECHNICAL_FEATURE_COLUMNS
    if mode == "regime":
        return CORE_FEATURE_COLUMNS + REGIME_FEATURE_COLUMNS
    raise ValueError(f"Unsupported feature mode: {mode}")


def apply_input_scaler(x: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
    return ((x - scaler["mean"]) / scaler["std"]).astype(np.float32)


def enforce_candle_validity(path: np.ndarray) -> np.ndarray:
    repaired = np.asarray(path, dtype=np.float32).copy()
    repaired[:, 1] = np.maximum(repaired[:, 1], np.maximum(repaired[:, 0], repaired[:, 3]))
    repaired[:, 2] = np.minimum(repaired[:, 2], np.minimum(repaired[:, 0], repaired[:, 3]))
    return repaired


def returns_to_prices_seq(anchor_prev_close: float, return_seq: np.ndarray) -> np.ndarray:
    prices = np.zeros_like(return_seq, dtype=np.float32)
    prev_close = float(anchor_prev_close)
    for step in range(return_seq.shape[0]):
        prices[step] = np.exp(return_seq[step]) * prev_close
        prev_close = float(prices[step, 3])
    return enforce_candle_validity(prices)
'@

$cells += New-CodeCell @'
fail_if_missing(MANIFEST_PATH, "Missing final artifact manifest. Run FinalTrain.ipynb first")
manifest = load_json(MANIFEST_PATH)

for key in ["shared", "forecast_models", "ensemble", "rl"]:
    if key not in manifest:
        raise KeyError(f"Manifest missing required section: {key}")

SYMBOL = manifest["runtime"].get("symbol", SYMBOL)
HORIZON = int(manifest["runtime"].get("horizon", HORIZON))

MODEL_ARTIFACTS = manifest["forecast_models"]
for expert_name in EXPECTED_EXPERTS:
    if expert_name not in MODEL_ARTIFACTS:
        raise KeyError(f"Manifest missing expert artifact for {expert_name}")

weights_path = Path(manifest["ensemble"]["weights_json"])
policy_path = Path(manifest["rl"]["policy_path"])
env_config_path = Path(manifest["rl"]["env_config_path"])
state_schema_path = Path(manifest["rl"]["state_schema_path"])

for path in [weights_path, policy_path, env_config_path, state_schema_path]:
    fail_if_missing(path, "Required inference artifact is missing")

ensemble_weights = load_json(weights_path)
if abs(sum(ensemble_weights.values()) - 1.0) >= 1e-6:
    raise ValueError("Saved ensemble weights do not sum to 1.0")

artifact_rows = []
for expert_name in EXPECTED_EXPERTS:
    artifact = MODEL_ARTIFACTS[expert_name]
    artifact_rows.append(
        {
            "expert": expert_name,
            "version": artifact["version"],
            "lookback": artifact["lookback"],
            "feature_mode": artifact["feature_mode"],
            "architecture": artifact["architecture"],
            "weight": ensemble_weights[expert_name],
        }
    )
artifact_summary_df = pd.DataFrame(artifact_rows)
display(artifact_summary_df)
'@

$cells += New-CodeCell @'
class RequestPacer:
    def __init__(self, max_requests_per_minute: int = 120) -> None:
        self.max_requests_per_minute = max_requests_per_minute
        self.request_times: List[float] = []

    def wait(self) -> None:
        now = time.time()
        self.request_times = [ts for ts in self.request_times if now - ts < 60.0]
        if len(self.request_times) >= self.max_requests_per_minute:
            sleep_time = 60.0 - (now - self.request_times[0]) + 0.05
            time.sleep(max(sleep_time, 0.05))
        self.request_times.append(time.time())


def _require_alpaca_credentials() -> Tuple[str, str]:
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError("Set ALPACA_API_KEY / ALPACA_API_SECRET before running FINAL.ipynb.")
    return api_key, api_secret


def _resolve_feed() -> DataFeed:
    preferred = (os.getenv("ALPACA_FEED") or "iex").strip().lower()
    return DataFeed.SIP if preferred == "sip" else DataFeed.IEX


def fetch_bars_alpaca(symbol: str, lookback_days: int) -> Tuple[pd.DataFrame, int]:
    api_key, api_secret = _require_alpaca_credentials()
    client = StockHistoricalDataClient(api_key, api_secret)
    feed = _resolve_feed()
    pacer = RequestPacer(MAX_REQUESTS_PER_MINUTE)

    end = pd.Timestamp.now(tz="UTC").floor("min")
    start = end - pd.Timedelta(days=lookback_days)
    cursor = start
    frames: List[pd.DataFrame] = []
    api_calls = 0

    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(days=REQUEST_CHUNK_DAYS), end)
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Minute,
            start=cursor.to_pydatetime(),
            end=chunk_end.to_pydatetime(),
            feed=feed,
            adjustment="raw",
        )
        success = False
        for attempt in range(MAX_RETRIES):
            try:
                pacer.wait()
                result = client.get_stock_bars(request)
                api_calls += 1
                bars = result.df
                success = True
                break
            except Exception:  # noqa: BLE001
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(0.5 * (attempt + 1))
        if not success:
            raise RuntimeError("Failed to fetch data from Alpaca.")

        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol, level=0)
        bars = bars.reset_index()
        if not bars.empty:
            frames.append(bars)
        cursor = chunk_end

    if not frames:
        raise RuntimeError("No bars returned from Alpaca.")

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    expected_columns = ["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]
    for column in expected_columns:
        if column not in df.columns:
            df[column] = np.nan if column not in ["volume", "trade_count"] else 0.0
    return df[expected_columns].reset_index(drop=True), api_calls


def sessionize_with_calendar(raw_df: pd.DataFrame, skip_open_bars: int = 6) -> pd.DataFrame:
    if raw_df.empty:
        raise ValueError("raw_df is empty")

    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(
        start_date=(df["timestamp"].min() - pd.Timedelta(days=3)).date(),
        end_date=(df["timestamp"].max() + pd.Timedelta(days=3)).date(),
    )

    frames: List[pd.DataFrame] = []
    for session_date, row in schedule.iterrows():
        open_ts = row["market_open"].tz_convert("UTC")
        close_ts = row["market_close"].tz_convert("UTC")
        mask = (df["timestamp"] >= open_ts) & (df["timestamp"] <= close_ts)
        session = df.loc[mask].copy()
        if session.empty:
            continue
        session = session.sort_values("timestamp").reset_index(drop=True)
        session["session_date"] = pd.Timestamp(session_date).strftime("%Y-%m-%d")
        session["bar_in_session"] = np.arange(len(session), dtype=np.int32)
        session["row_open_skip"] = session["bar_in_session"] < skip_open_bars
        session["row_imputed"] = session[["open", "high", "low", "close", "vwap"]].isna().any(axis=1)
        for price_col in ["open", "high", "low", "close", "vwap"]:
            session[price_col] = session[price_col].ffill().bfill()
        for count_col in ["volume", "trade_count"]:
            session[count_col] = session[count_col].fillna(0.0)
        frames.append(session)

    if not frames:
        raise RuntimeError("Sessionization removed every row.")

    output = pd.concat(frames, ignore_index=True)
    output["timestamp_ny"] = output["timestamp"].dt.tz_convert(SESSION_TZ)
    output["minute_of_day"] = output["timestamp_ny"].dt.hour * 60 + output["timestamp_ny"].dt.minute
    return output
'@

$cells += New-CodeCell @'
raw_df_utc, api_calls = fetch_bars_alpaca(SYMBOL, LOOKBACK_DAYS)
session_df = sessionize_with_calendar(raw_df_utc, skip_open_bars=SESSION_OPEN_SKIP_BARS)

print(
    {
        "symbol": SYMBOL,
        "rows_raw": len(raw_df_utc),
        "rows_sessionized": len(session_df),
        "sessions": int(session_df["session_date"].nunique()),
        "last_visible_timestamp": str(session_df["timestamp"].iloc[-1]),
        "api_calls": api_calls,
    }
)
display(session_df.tail())
'@

$cells += New-CodeCell @'
def safe_divide(numerator: pd.Series, denominator: pd.Series, eps: float = 1e-8) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan).fillna(eps)


def build_core_feature_frame(session_df: pd.DataFrame) -> pd.DataFrame:
    df = session_df.copy().sort_values("timestamp").reset_index(drop=True)
    df["prev_close"] = df["close"].shift(1)
    session_change = df["session_date"] != df["session_date"].shift(1)
    df.loc[session_change, "prev_close"] = df.loc[session_change, "open"]

    prev_volume = df["volume"].shift(1).fillna(df["volume"].median())
    prev_trade_count = df["trade_count"].shift(1).fillna(df["trade_count"].median())

    df["rOpen"] = np.log(safe_divide(df["open"], df["prev_close"]).clip(lower=1e-8))
    df["rHigh"] = np.log(safe_divide(df["high"], df["prev_close"]).clip(lower=1e-8))
    df["rLow"] = np.log(safe_divide(df["low"], df["prev_close"]).clip(lower=1e-8))
    df["rClose"] = np.log(safe_divide(df["close"], df["prev_close"]).clip(lower=1e-8))
    df["returns"] = df["rClose"]
    df["logVolChange"] = np.log1p(df["volume"]) - np.log1p(prev_volume)
    df["logTradeCountChange"] = np.log1p(df["trade_count"]) - np.log1p(prev_trade_count)
    df["vwapDelta"] = safe_divide(df["vwap"] - df["prev_close"], df["prev_close"])
    df["rangeFrac"] = safe_divide(df["high"] - df["low"], df["prev_close"]).clip(lower=0.0)
    candle_span = (df["high"] - df["low"]).replace(0.0, np.nan)
    df["orderFlowProxy"] = ((df["close"] - df["open"]) / candle_span).fillna(0.0) * np.log1p(df["volume"])
    close_position = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / candle_span
    df["tickPressure"] = close_position.fillna(0.0)
    return df


class TechnicalIndicatorCalculator:
    def ema(self, series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False, min_periods=span).mean()

    def sma(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=window).mean()

    def rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    def stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> Tuple[pd.Series, pd.Series]:
        rolling_low = low.rolling(window, min_periods=window).min()
        rolling_high = high.rolling(window, min_periods=window).max()
        stoch_k = 100.0 * (close - rolling_low) / (rolling_high - rolling_low).replace(0.0, np.nan)
        stoch_d = stoch_k.rolling(3, min_periods=3).mean()
        return stoch_k, stoch_d

    def bollinger(self, close: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        mid = close.rolling(window, min_periods=window).mean()
        std = close.rolling(window, min_periods=window).std()
        upper = mid + num_std * std
        lower = mid - num_std * std
        width = (upper - lower) / mid.replace(0.0, np.nan)
        position = (close - lower) / (upper - lower).replace(0.0, np.nan)
        return upper, lower, width, position

    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.sign(close.diff().fillna(0.0))
        return (direction * volume.fillna(0.0)).cumsum()


def calculate_technical_features(core_df: pd.DataFrame) -> pd.DataFrame:
    calc = TechnicalIndicatorCalculator()
    out = pd.DataFrame(index=core_df.index)
    close = core_df["close"]
    high = core_df["high"]
    low = core_df["low"]
    volume = core_df["volume"]

    out["sma_5"] = calc.sma(close, 5)
    out["sma_10"] = calc.sma(close, 10)
    out["sma_20"] = calc.sma(close, 20)
    out["sma_50"] = calc.sma(close, 50)
    out["ema_12"] = calc.ema(close, 12)
    out["ema_26"] = calc.ema(close, 26)
    out["macd_line"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd_line"].ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_histogram"] = out["macd_line"] - out["macd_signal"]
    out["macd_momentum"] = out["macd_histogram"].diff()
    out["rsi_14"] = calc.rsi(close, 14)
    out["rsi_14_slope"] = out["rsi_14"].diff()
    out["stoch_k"], out["stoch_d"] = calc.stochastic(high, low, close, 14)
    out["bb_upper"], out["bb_lower"], out["bb_width"], out["bb_position"] = calc.bollinger(close, 20, 2.0)
    out["atr_14"] = calc.atr(high, low, close, 14)
    out["atr_14_pct"] = out["atr_14"] / close.replace(0.0, np.nan)
    out["obv"] = calc.obv(close, volume)
    out["obv_slope"] = out["obv"].diff(5) / 5.0
    out["vwap_20"] = core_df["vwap"].rolling(20, min_periods=20).mean()
    out["vwap_20_dev"] = safe_divide(core_df["close"] - out["vwap_20"], out["vwap_20"])
    out["price_momentum_5"] = close.pct_change(5)
    out["price_momentum_10"] = close.pct_change(10)
    out["price_momentum_20"] = close.pct_change(20)
    body = core_df["close"] - core_df["open"]
    full_range = (core_df["high"] - core_df["low"]).replace(0.0, np.nan)
    out["body_size"] = body
    out["body_pct"] = body / full_range
    out["upper_shadow"] = core_df["high"] - core_df[["open", "close"]].max(axis=1)
    out["lower_shadow"] = core_df[["open", "close"]].min(axis=1) - core_df["low"]
    out["direction"] = np.sign(body)
    return out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)


def compute_turbulence_index(returns: pd.Series, lookback: int = 60) -> pd.Series:
    rolling_mean = returns.rolling(lookback, min_periods=lookback).mean()
    rolling_std = returns.rolling(lookback, min_periods=lookback).std().replace(0.0, np.nan)
    z = ((returns - rolling_mean) / rolling_std).abs()
    return z.fillna(0.0)


def build_dynamic_regime_indicator(atr_pct: pd.Series, turbulence: pd.Series, lookback: int = 390) -> pd.Series:
    atr_q75 = atr_pct.rolling(lookback, min_periods=60).quantile(0.75).shift(1)
    atr_q90 = atr_pct.rolling(lookback, min_periods=60).quantile(0.90).shift(1)
    turb_q75 = turbulence.rolling(lookback, min_periods=60).quantile(0.75).shift(1)
    turb_q90 = turbulence.rolling(lookback, min_periods=60).quantile(0.90).shift(1)
    crisis = (atr_pct >= atr_q90.fillna(np.inf)) & (turbulence >= turb_q90.fillna(np.inf))
    elevated = (atr_pct >= atr_q75.fillna(np.inf)) | (turbulence >= turb_q75.fillna(np.inf))
    return pd.Series(np.select([crisis, elevated], [1.0, 0.5], default=0.0), index=atr_pct.index)


def build_feature_frame_for_mode(session_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    core_df = build_core_feature_frame(session_df)
    technical_df = calculate_technical_features(core_df)
    feature_df = core_df.copy()
    if mode == "technical":
        feature_df = pd.concat([feature_df, technical_df], axis=1)
    elif mode == "regime":
        feature_df["atr_14"] = technical_df["atr_14"]
        feature_df["atr_14_pct"] = technical_df["atr_14_pct"]
        feature_df["turbulence_60"] = compute_turbulence_index(feature_df["returns"], 60)
        feature_df["regime_indicator"] = build_dynamic_regime_indicator(feature_df["atr_14_pct"], feature_df["turbulence_60"])
    elif mode != "core":
        raise ValueError(f"Unsupported mode: {mode}")

    if "turbulence_60" not in feature_df.columns:
        feature_df["turbulence_60"] = compute_turbulence_index(feature_df["returns"], 60)
    if "atr_14" not in feature_df.columns:
        feature_df["atr_14"] = technical_df["atr_14"]
    if "atr_14_pct" not in feature_df.columns:
        feature_df["atr_14_pct"] = technical_df["atr_14_pct"]
    if "regime_indicator" not in feature_df.columns:
        feature_df["regime_indicator"] = build_dynamic_regime_indicator(feature_df["atr_14_pct"], feature_df["turbulence_60"])

    feature_df["relative_volume"] = feature_df["volume"] / feature_df["volume"].rolling(20, min_periods=5).mean().replace(0.0, np.nan)
    feature_df["relative_volume"] = feature_df["relative_volume"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return feature_df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)


FEATURE_FRAMES = {
    "core": build_feature_frame_for_mode(session_df, "core"),
    "technical": build_feature_frame_for_mode(session_df, "technical"),
    "regime": build_feature_frame_for_mode(session_df, "regime"),
}

print({mode: frame.shape for mode, frame in FEATURE_FRAMES.items()})
'@

$cells += New-CodeCell @'
@dataclass
class LoadedExpertBundle:
    expert_name: str
    version: str
    lookback: int
    feature_mode: str
    architecture: str
    ensemble_size: int
    amp_enabled: bool
    model: nn.Module
    scaler: Dict[str, np.ndarray]
    feature_columns: List[str]
    inference_config: Dict[str, Any]
    retrieval_artifact: Optional[Dict[str, np.ndarray]]
    spec: Dict[str, Any]


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.energy = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        projected_query = self.query(query).unsqueeze(1)
        projected_memory = self.key(memory)
        scores = self.energy(torch.tanh(projected_query + projected_memory)).squeeze(-1)
        attn = torch.softmax(scores, dim=1)
        context = torch.bmm(attn.unsqueeze(1), memory).squeeze(1)
        return context, attn


class Seq2SeqAttnGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float, horizon: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention = AdditiveAttention(hidden_size)
        self.decoder = nn.GRUCell(4 + hidden_size, hidden_size)
        self.mu_head = nn.Linear(hidden_size * 2, 4)
        self.log_sigma_head = nn.Linear(hidden_size * 2, 4)

    def encode_sequence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_out, enc_hidden = self.encoder(x)
        return enc_out, enc_hidden[-1]

    def encode_context(self, x: torch.Tensor) -> torch.Tensor:
        enc_out, enc_hidden = self.encode_sequence(x)
        pooled = enc_out.mean(dim=1)
        return F.normalize(torch.cat([pooled, enc_hidden], dim=-1), dim=-1)

    def decode_step(self, decoder_input: torch.Tensor, decoder_hidden: torch.Tensor, encoder_memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context, _ = self.attention(decoder_hidden, encoder_memory)
        decoder_hidden = self.decoder(torch.cat([decoder_input, context], dim=-1), decoder_hidden)
        fused = torch.cat([decoder_hidden, context], dim=-1)
        mu = self.mu_head(fused)
        log_sigma = torch.clamp(self.log_sigma_head(fused), min=-5.0, max=3.0)
        return mu, log_sigma, decoder_hidden


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ITransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm_1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm_2(x + self.dropout(ffn_out))


class ITransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, lookback: int, d_model: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.token_projection = nn.Linear(lookback, d_model)
        self.positional = SinusoidalPositionalEncoding(d_model, max_len=input_dim + 8)
        self.layers = nn.ModuleList([ITransformerEncoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.transpose(1, 2)
        tokens = self.token_projection(tokens)
        tokens = self.positional(tokens)
        for layer in self.layers:
            tokens = layer(tokens)
        return self.norm(tokens).flatten(start_dim=1)


class HybridSeq2SeqForecaster(nn.Module):
    def __init__(self, input_dim: int, lookback: int, hidden_size: int, num_layers: int, dropout: float, horizon: int, d_model: int, n_heads: int, n_layers: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.itransformer = ITransformerEncoder(input_dim, lookback, d_model, n_heads, n_layers, dropout)
        self.fusion = nn.Sequential(
            nn.Linear((input_dim * d_model) + hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.attention = AdditiveAttention(hidden_size)
        self.decoder = nn.GRUCell(4 + hidden_size, hidden_size)
        self.mu_head = nn.Linear(hidden_size * 2, 4)
        self.log_sigma_head = nn.Linear(hidden_size * 2, 4)

    def encode_sequence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gru_out, gru_hidden = self.gru(x)
        transformer_flat = self.itransformer(x)
        fused = self.fusion(torch.cat([transformer_flat, gru_hidden[-1]], dim=-1))
        return gru_out, fused

    def encode_context(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, fused = self.encode_sequence(x)
        return F.normalize(torch.cat([gru_out.mean(dim=1), fused], dim=-1), dim=-1)

    def decode_step(self, decoder_input: torch.Tensor, decoder_hidden: torch.Tensor, encoder_memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context, _ = self.attention(decoder_hidden, encoder_memory)
        decoder_hidden = self.decoder(torch.cat([decoder_input, context], dim=-1), decoder_hidden)
        fused = torch.cat([decoder_hidden, context], dim=-1)
        mu = self.mu_head(fused)
        log_sigma = torch.clamp(self.log_sigma_head(fused), min=-5.0, max=3.0)
        return mu, log_sigma, decoder_hidden


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self.shared(x)
        mean = torch.tanh(self.actor_mean(shared))
        std = torch.exp(self.actor_log_std).expand_as(mean)
        value = self.critic(shared)
        return mean, std, value
'@

$cells += New-CodeCell @'
def build_model_from_checkpoint(checkpoint: Dict[str, Any]) -> nn.Module:
    spec = checkpoint["spec"]
    architecture = spec["architecture"]
    input_dim = int(checkpoint["input_dim"])
    hidden_size = checkpoint["training_defaults"]["hidden_size"]
    num_layers = checkpoint["training_defaults"]["num_layers"]
    dropout = checkpoint["training_defaults"]["dropout"]
    horizon = HORIZON

    if architecture in ["gru", "gru_rag"]:
        model = Seq2SeqAttnGRU(input_dim, hidden_size, num_layers, dropout, horizon)
    elif architecture == "hybrid_itransformer_gru":
        model = HybridSeq2SeqForecaster(
            input_dim=input_dim,
            lookback=int(spec["lookback"]),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
            d_model=int(spec.get("d_model", 128)),
            n_heads=int(spec.get("n_heads", 8)),
            n_layers=int(spec.get("n_layers", 2)),
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    model.load_state_dict(checkpoint["state_dict"])
    return model.to(DEVICE).eval()


def load_expert_bundle(expert_name: str) -> LoadedExpertBundle:
    artifact = MODEL_ARTIFACTS[expert_name]
    model_path = Path(artifact["model_path"])
    scaler_path = Path(artifact["scaler_path"])
    feature_manifest_path = Path(artifact["feature_manifest_path"])
    inference_config_path = Path(artifact["inference_config_path"])
    for path in [model_path, scaler_path, feature_manifest_path, inference_config_path]:
        fail_if_missing(path, f"Missing artifact for {expert_name}")

    checkpoint = torch.load(model_path, map_location="cpu")
    model = build_model_from_checkpoint(checkpoint)
    scaler_npz = np.load(scaler_path)
    scaler = {"mean": scaler_npz["mean"].astype(np.float32), "std": scaler_npz["std"].astype(np.float32)}
    feature_manifest = load_json(feature_manifest_path)
    inference_config = load_json(inference_config_path)

    retrieval_artifact = None
    rag_path = Path(artifact["dir"]) / "rag_database.npz"
    if rag_path.exists():
        rag_npz = np.load(rag_path)
        retrieval_artifact = {
            "embeddings": rag_npz["embeddings"].astype(np.float32),
            "future_returns": rag_npz["future_returns"].astype(np.float32),
            "indices": rag_npz["indices"].astype(np.int32),
        }

    return LoadedExpertBundle(
        expert_name=expert_name,
        version=artifact["version"],
        lookback=int(artifact["lookback"]),
        feature_mode=artifact["feature_mode"],
        architecture=artifact["architecture"],
        ensemble_size=int(inference_config["ensemble_size"]),
        amp_enabled=bool(checkpoint["spec"].get("amp_enabled", False)),
        model=model,
        scaler=scaler,
        feature_columns=list(feature_manifest["feature_columns"]),
        inference_config=inference_config,
        retrieval_artifact=retrieval_artifact,
        spec=checkpoint["spec"],
    )


def intraday_temperature(anchor_ts: pd.Timestamp) -> float:
    local_ts = pd.Timestamp(anchor_ts).tz_convert(SESSION_TZ)
    hhmm = local_ts.hour * 100 + local_ts.minute
    if hhmm < 1015:
        return 1.25
    if hhmm < 1400:
        return 1.45
    return 1.60


def calculate_trend_slope(close_seq: np.ndarray) -> float:
    if len(close_seq) < 2:
        return 0.0
    x_axis = np.arange(len(close_seq), dtype=np.float32)
    slope, _ = np.polyfit(x_axis, close_seq.astype(np.float32), 1)
    return float(slope / max(abs(close_seq[-1]), 1e-6))


def select_best_path_by_trend(historical_closes: np.ndarray, candidate_paths: np.ndarray, trend_lookback_bars: int, strong_trend_threshold: float) -> np.ndarray:
    history_slice = historical_closes[-trend_lookback_bars:]
    historical_slope = calculate_trend_slope(history_slice)
    candidate_slopes = np.asarray([calculate_trend_slope(path[:, 3]) for path in candidate_paths], dtype=np.float32)
    if abs(historical_slope) >= strong_trend_threshold:
        same_sign = np.sign(candidate_slopes) == np.sign(historical_slope)
        filtered_idx = np.where(same_sign)[0]
        if len(filtered_idx) > 0:
            candidate_paths = candidate_paths[filtered_idx]
            candidate_slopes = candidate_slopes[filtered_idx]
    best_idx = int(np.argmin(np.abs(candidate_slopes - historical_slope)))
    return candidate_paths[best_idx]


def detect_regime_multiplier(history_slice: pd.DataFrame) -> Tuple[str, float, float]:
    turbulence = float(history_slice["turbulence_60"].iloc[-1])
    atr_pct = float(history_slice["atr_14_pct"].iloc[-1])
    turb_q75 = float(history_slice["turbulence_60"].quantile(0.75))
    turb_q90 = float(history_slice["turbulence_60"].quantile(0.90))
    atr_q75 = float(history_slice["atr_14_pct"].quantile(0.75))
    atr_q90 = float(history_slice["atr_14_pct"].quantile(0.90))
    if turbulence >= turb_q90 and atr_pct >= atr_q90:
        return "CRISIS", 1.8, 1.0
    if turbulence >= turb_q75 or atr_pct >= atr_q75:
        return "ELEVATED", 1.3, 0.5
    return "NORMAL", 1.0, 0.0


def apply_retrieval_blend(model: nn.Module, x_scaled_single: np.ndarray, generated_returns: np.ndarray, retrieval_artifact: Optional[Dict[str, np.ndarray]], k_retrieve: int, blend_weight: float) -> np.ndarray:
    if retrieval_artifact is None or len(retrieval_artifact["embeddings"]) == 0:
        return generated_returns
    with torch.no_grad():
        query = torch.from_numpy(x_scaled_single[None, ...]).to(DEVICE)
        query_embedding = model.encode_context(query).detach().cpu().numpy()[0].astype(np.float32)
    query_embedding /= np.linalg.norm(query_embedding).clip(min=1e-8)
    similarities = retrieval_artifact["embeddings"] @ query_embedding
    top_idx = np.argsort(similarities)[-k_retrieve:][::-1]
    top_scores = similarities[top_idx]
    top_weights = np.exp(top_scores - top_scores.max())
    top_weights /= top_weights.sum().clip(min=1e-8)
    retrieved_future = np.tensordot(top_weights, retrieval_artifact["future_returns"][top_idx], axes=(0, 0))
    return ((1.0 - blend_weight) * generated_returns + blend_weight * retrieved_future).astype(np.float32)


def build_latest_input(feature_df: pd.DataFrame, bundle: LoadedExpertBundle) -> Dict[str, Any]:
    if len(feature_df) < bundle.lookback:
        raise RuntimeError(f"Not enough rows to build input for {bundle.expert_name}")
    context = feature_df.iloc[-bundle.lookback :].copy()
    feature_block = context.loc[:, bundle.feature_columns].to_numpy(dtype=np.float32)
    impute_frac = float(context["row_imputed"].mean())
    feature_block = np.concatenate([feature_block, np.full((bundle.lookback, 1), impute_frac, dtype=np.float32)], axis=1)
    scaled = apply_input_scaler(feature_block[None, ...], bundle.scaler)[0]
    anchor_prev_close = float(context["close"].iloc[-1])
    anchor_timestamp = pd.Timestamp(context["timestamp"].iloc[-1])
    return {
        "scaled_input": scaled,
        "anchor_prev_close": anchor_prev_close,
        "anchor_timestamp": anchor_timestamp,
        "historical_closes": context["close"].to_numpy(dtype=np.float32),
        "context": context,
    }


def generate_ensemble_with_trend_selection(bundle: LoadedExpertBundle, model_input: Dict[str, Any], temperature: float, regime_multiplier: float = 1.0) -> np.ndarray:
    model = bundle.model
    x_tensor = torch.from_numpy(model_input["scaled_input"][None, ...]).to(DEVICE)
    with torch.no_grad():
        encoder_memory, decoder_hidden_init = model.encode_sequence(x_tensor)
    decoder_input_init = x_tensor[:, -1, :4]
    candidate_paths = []

    for seed_offset in range(bundle.ensemble_size):
        torch.manual_seed(SEED + seed_offset)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + seed_offset)
        decoder_hidden = decoder_hidden_init.clone()
        decoder_input = decoder_input_init.clone()
        sampled_steps = []
        for _ in range(HORIZON):
            with torch.no_grad():
                mu, log_sigma, decoder_hidden = model.decode_step(decoder_input, decoder_hidden, encoder_memory)
                sigma = torch.exp(log_sigma).clamp(min=bundle.inference_config["min_predicted_vol"]) * temperature * regime_multiplier
                sample = mu + torch.randn_like(mu) * sigma
            sampled_steps.append(sample.squeeze(0).detach().cpu().numpy().astype(np.float32))
            decoder_input = sample
        sampled_returns = np.stack(sampled_steps).astype(np.float32)
        if bundle.retrieval_artifact is not None:
            sampled_returns = apply_retrieval_blend(
                model=model,
                x_scaled_single=model_input["scaled_input"],
                generated_returns=sampled_returns,
                retrieval_artifact=bundle.retrieval_artifact,
                k_retrieve=5,
                blend_weight=0.25,
            )
        candidate_paths.append(returns_to_prices_seq(model_input["anchor_prev_close"], sampled_returns))

    candidate_paths = np.stack(candidate_paths).astype(np.float32)
    return select_best_path_by_trend(
        model_input["historical_closes"],
        candidate_paths,
        trend_lookback_bars=int(bundle.inference_config["trend_lookback_bars"]),
        strong_trend_threshold=float(bundle.inference_config["strong_trend_threshold"]),
    )


def regime_scale_from_indicator(value: float) -> float:
    if value >= 1.0:
        return 0.3
    if value >= 0.5:
        return 0.7
    return 1.0


def regime_name_from_indicator(value: float) -> str:
    if value >= 1.0:
        return "CRISIS"
    if value >= 0.5:
        return "ELEVATED"
    return "NORMAL"


def stance_from_action(action: float) -> str:
    if action > ACTION_NEUTRAL_BAND:
        return "LONG"
    if action < -ACTION_NEUTRAL_BAND:
        return "SHORT"
    return "NEUTRAL"
'@

$cells += New-CodeCell @'
EXPERT_PREDICTIONS: Dict[str, Dict[str, Any]] = {}
expert_summary_rows: List[Dict[str, Any]] = []

for expert_name in EXPECTED_EXPERTS:
    bundle = load_expert_bundle(expert_name)
    feature_df = FEATURE_FRAMES[bundle.feature_mode]
    model_input = build_latest_input(feature_df, bundle)

    regime_name = "NORMAL"
    regime_indicator = 0.0
    regime_multiplier = 1.0
    if bundle.feature_mode == "regime":
        recent_history = feature_df.iloc[max(0, len(feature_df) - 390) :].copy()
        regime_name, regime_multiplier, regime_indicator = detect_regime_multiplier(recent_history)

    base_temperature = float(bundle.inference_config["sampling_temperature"])
    time_temperature = intraday_temperature(model_input["anchor_timestamp"])
    effective_temperature = base_temperature * (time_temperature / 1.5)

    predicted_path = generate_ensemble_with_trend_selection(
        bundle=bundle,
        model_input=model_input,
        temperature=effective_temperature,
        regime_multiplier=regime_multiplier,
    )

    EXPERT_PREDICTIONS[expert_name] = {
        "version": bundle.version,
        "weight": float(ensemble_weights[expert_name]),
        "lookback": bundle.lookback,
        "feature_mode": bundle.feature_mode,
        "architecture": bundle.architecture,
        "anchor_timestamp": model_input["anchor_timestamp"],
        "anchor_prev_close": model_input["anchor_prev_close"],
        "path": predicted_path,
        "regime_name": regime_name,
        "regime_indicator": regime_indicator,
        "effective_temperature": effective_temperature * regime_multiplier,
    }

    expert_summary_rows.append(
        {
            "expert": expert_name,
            "version": bundle.version,
            "weight": float(ensemble_weights[expert_name]),
            "lookback": bundle.lookback,
            "feature_mode": bundle.feature_mode,
            "architecture": bundle.architecture,
            "anchor_timestamp": model_input["anchor_timestamp"],
            "next_close": float(predicted_path[0, 3]),
            "horizon_close": float(predicted_path[-1, 3]),
            "next_close_return_pct": float((predicted_path[0, 3] / model_input["anchor_prev_close"] - 1.0) * 100.0),
            "horizon_close_return_pct": float((predicted_path[-1, 3] / model_input["anchor_prev_close"] - 1.0) * 100.0),
            "regime_name": regime_name,
            "temperature": float(effective_temperature * regime_multiplier),
        }
    )

    del bundle.model
    clear_torch_memory()

expert_summary_df = pd.DataFrame(expert_summary_rows).sort_values("weight", ascending=False).reset_index(drop=True)
display(expert_summary_df)
'@

$cells += New-CodeCell @'
aggregate_path_raw = np.zeros((HORIZON, 4), dtype=np.float32)
aggregate_regime_indicator = 0.0
for expert_name in EXPECTED_EXPERTS:
    weight = float(ensemble_weights[expert_name])
    aggregate_path_raw += weight * EXPERT_PREDICTIONS[expert_name]["path"]
    aggregate_regime_indicator += weight * EXPERT_PREDICTIONS[expert_name]["regime_indicator"]

aggregate_path = enforce_candle_validity(aggregate_path_raw)
aggregate_regime_name = regime_name_from_indicator(aggregate_regime_indicator)

close_path_df = pd.DataFrame({"step": np.arange(1, HORIZON + 1)})
for expert_name in EXPECTED_EXPERTS:
    close_path_df[expert_name] = EXPERT_PREDICTIONS[expert_name]["path"][:, 3]
close_path_df["aggregate_close"] = aggregate_path[:, 3]

aggregate_summary_df = pd.DataFrame(
    [
        {
            "anchor_timestamp": list(EXPERT_PREDICTIONS.values())[0]["anchor_timestamp"],
            "prev_close": list(EXPERT_PREDICTIONS.values())[0]["anchor_prev_close"],
            "aggregate_next_close": float(aggregate_path[0, 3]),
            "aggregate_horizon_close": float(aggregate_path[-1, 3]),
            "aggregate_next_return_pct": float((aggregate_path[0, 3] / list(EXPERT_PREDICTIONS.values())[0]["anchor_prev_close"] - 1.0) * 100.0),
            "aggregate_horizon_return_pct": float((aggregate_path[-1, 3] / list(EXPERT_PREDICTIONS.values())[0]["anchor_prev_close"] - 1.0) * 100.0),
            "aggregate_regime_name": aggregate_regime_name,
            "aggregate_regime_indicator": float(aggregate_regime_indicator),
        }
    ]
)

display(aggregate_summary_df)
display(close_path_df.head(10))
'@

$cells += New-CodeCell @'
policy_checkpoint = torch.load(policy_path, map_location="cpu")
env_config = load_json(env_config_path)
state_schema = load_json(state_schema_path)

technical_frame = FEATURE_FRAMES["technical"]
latest_market_row = technical_frame.iloc[-1]
market_state = latest_market_row[MARKET_STATE_COLUMNS].to_numpy(dtype=np.float32)
portfolio_state = np.array(
    [
        float(PORTFOLIO_STATE["cash_norm"]),
        float(PORTFOLIO_STATE["shares_norm"]),
        float(PORTFOLIO_STATE["portfolio_value_norm"]),
        float(PORTFOLIO_STATE["position_pct"]),
    ],
    dtype=np.float32,
)
regime_state = np.array([aggregate_regime_indicator], dtype=np.float32)
policy_state = np.concatenate([aggregate_path.reshape(-1).astype(np.float32), market_state, portfolio_state, regime_state]).astype(np.float32)

expected_state_dim = int(policy_checkpoint["state_dim"])
if policy_state.shape[0] != expected_state_dim:
    raise ValueError(f"Policy state dimension mismatch. Expected {expected_state_dim}, got {policy_state.shape[0]}")

policy = ActorCritic(expected_state_dim).to("cpu")
policy.load_state_dict(policy_checkpoint["state_dict"])
policy.eval()

with torch.no_grad():
    mean_action, action_std, value_estimate = policy(torch.from_numpy(policy_state).float().unsqueeze(0))

raw_action = float(mean_action.item())
policy_std = float(action_std.item())
confidence_score = float(1.0 / (1.0 + policy_std))
regime_scale = regime_scale_from_indicator(aggregate_regime_indicator)
adjusted_action = raw_action * regime_scale
policy_summary_df = pd.DataFrame(
    [
        {
            "raw_action": raw_action,
            "policy_std": policy_std,
            "confidence_score": confidence_score,
            "critic_value_estimate": float(value_estimate.item()),
            "aggregate_regime_name": aggregate_regime_name,
            "regime_scale": regime_scale,
            "recommended_position_pct": adjusted_action,
            "stance": stance_from_action(adjusted_action),
        }
    ]
)

display(policy_summary_df)
'@

$cells += New-CodeCell @'
def plot_aggregate_candles(path: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, candle in enumerate(path, start=1):
        open_price, high_price, low_price, close_price = candle
        color = "#1a7f37" if close_price >= open_price else "#b42318"
        ax.plot([idx, idx], [low_price, high_price], color=color, linewidth=1.0)
        body_low = min(open_price, close_price)
        body_high = max(open_price, close_price)
        body_height = max(body_high - body_low, 1e-6)
        rect = Rectangle((idx - 0.3, body_low), 0.6, body_height, facecolor=color, edgecolor=color, alpha=0.75)
        ax.add_patch(rect)
    ax.set_title(title)
    ax.set_xlabel("Forecast Step")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2)
    plt.show()


plt.figure(figsize=(12, 5))
for expert_name in EXPECTED_EXPERTS:
    plt.plot(close_path_df["step"], close_path_df[expert_name], linewidth=1.2, alpha=0.65, label=expert_name)
plt.plot(close_path_df["step"], close_path_df["aggregate_close"], linewidth=2.6, color="black", label="aggregate")
plt.title(f"{SYMBOL} Forecast Close Paths")
plt.xlabel("Forecast Step")
plt.ylabel("Close Price")
plt.grid(alpha=0.2)
plt.legend(ncol=3)
plt.show()

plot_aggregate_candles(aggregate_path, f"{SYMBOL} Aggregate OHLC Forecast")
'@

$cells += New-CodeCell @'
if SAVE_RUN_OUTPUTS:
    ensure_dir(FINAL_RUN_DIR)
    save_dataframe(artifact_summary_df, FINAL_RUN_DIR / "expert_artifact_summary.csv")
    save_dataframe(expert_summary_df, FINAL_RUN_DIR / "expert_prediction_summary.csv")
    save_dataframe(close_path_df, FINAL_RUN_DIR / "close_path_comparison.csv")
    save_dataframe(pd.DataFrame(aggregate_path, columns=["Open", "High", "Low", "Close"]), FINAL_RUN_DIR / "aggregate_forecast.csv")
    save_json(
        FINAL_RUN_DIR / "policy_summary.json",
        {
            "summary": policy_summary_df.iloc[0].to_dict(),
            "portfolio_state": PORTFOLIO_STATE,
            "run_label": RUN_LABEL,
            "symbol": SYMBOL,
        },
    )
    np.savez_compressed(
        FINAL_RUN_DIR / "forecast_bundle.npz",
        aggregate_path=aggregate_path.astype(np.float32),
        aggregate_path_raw=aggregate_path_raw.astype(np.float32),
        aggregate_regime_indicator=np.asarray([aggregate_regime_indicator], dtype=np.float32),
        close_path_steps=close_path_df["step"].to_numpy(dtype=np.int32),
        close_path_values=close_path_df.drop(columns=["step"]).to_numpy(dtype=np.float32),
    )
    save_json(
        FINAL_RUN_DIR / "run_metadata.json",
        {
            "created_at_utc": production_timestamp_utc(),
            "symbol": SYMBOL,
            "anchor_timestamp": str(list(EXPERT_PREDICTIONS.values())[0]["anchor_timestamp"]),
            "manifest_path": str(MANIFEST_PATH),
            "weights_path": str(weights_path),
            "policy_path": str(policy_path),
        },
    )
    print(f"Inference outputs written to: {FINAL_RUN_DIR}")
else:
    print("SAVE_RUN_OUTPUTS=False, so no run files were written.")
'@

$notebook = @{
    cells = $cells
    metadata = @{
        kernelspec = @{
            display_name = "Python 3"
            language = "python"
            name = "python3"
        }
        language_info = @{
            name = "python"
            version = "3.11"
        }
    }
    nbformat = 4
    nbformat_minor = 5
}

$outputPath = Join-Path $PWD "FINAL.ipynb"
[System.IO.File]::WriteAllText(
    $outputPath,
    ($notebook | ConvertTo-Json -Depth 100),
    (New-Object System.Text.UTF8Encoding($false))
)
Write-Output "Wrote $outputPath"
