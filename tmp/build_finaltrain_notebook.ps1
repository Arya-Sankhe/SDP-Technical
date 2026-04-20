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
# FinalTrain.ipynb

This notebook is the offline build step for the final stacked system. It trains the five forecast experts (`v8.5`, `v9.1`, `v9.2`, `v9.3`, `v9.5`), saves every reusable artifact they need for inference, computes ensemble weights on aligned validation/rolling metrics, and then trains the `v9.6` PPO decision layer on top of the aggregate forecast stream.

The output of this notebook is a frozen artifact tree under `output/final_artifacts/`. `FINAL.ipynb` should only load those artifacts and run inference.
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
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
from torch.distributions import Normal
from torch.utils.data import DataLoader, Dataset
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
SKIP_HEAVY_EXPERTS_ON_WINDOWS_8GB = os.name == "nt" and LOW_VRAM_GPU
LIGHTWEIGHT_V95_ON_LOW_VRAM = os.name == "nt" and LOW_VRAM_GPU
SKIPPED_EXPERT_NAMES = {"v9_3"} if SKIP_HEAVY_EXPERTS_ON_WINDOWS_8GB else set()

SYMBOL = "MSFT"
LOOKBACK_DAYS = 120
HORIZON = 50
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

ARTIFACT_ROOT = Path("output/final_artifacts")
SHARED_DIR = ARTIFACT_ROOT / "shared"
MODELS_DIR = ARTIFACT_ROOT / "models"
ENSEMBLE_DIR = ARTIFACT_ROOT / "ensemble"
RL_DIR = ARTIFACT_ROOT / "rl"

REFRESH_SHARED_SNAPSHOT = True
RESUME_COMPLETED_EXPERTS = True
SAVE_PARQUET_COMPRESSION = "snappy"
REQUEST_CHUNK_DAYS = 5
MAX_REQUESTS_PER_MINUTE = 120
MAX_RETRIES = 5
SESSION_TZ = "America/New_York"
SESSION_OPEN_SKIP_BARS = 6
VALIDATION_SLICE_FRACTION = 0.18
WALKFORWARD_SLICE_OVERLAP = 0.15
PRODUCTION_VAL_FRACTION = 0.10
MAX_VAL_WINDOWS_PER_SLICE = 192

TRAINING_DEFAULTS = {
    "hidden_size": 256,
    "num_layers": 2,
    "dropout": 0.20,
    "learning_rate": 5e-4,
    "weight_decay": 1e-5,
    "teacher_forcing_decay": 0.95,
    "gradient_clip_norm": 1.0,
    "final_max_epochs": 60,
    "final_patience": 12,
    "sweep_max_epochs": 15,
    "sweep_patience": 5,
    "scheduler_factor": 0.5,
    "scheduler_patience": 3,
    "scheduler_min_lr": 1e-6,
    "loss_weights": {
        "range": 0.30,
        "volatility": 0.50,
        "directional": 0.10,
    },
}

INFERENCE_DEFAULTS = {
    "sampling_temperature": 1.5,
    "trend_lookback_bars": 20,
    "strong_trend_threshold": 0.002,
    "min_predicted_vol": 0.0001,
}

RAG_CONFIG = {
    "embedding_dim": 64,
    "k_retrieve": 5,
    "blend_weight": 0.25,
    "max_patterns": 4000,
}

RL_CONFIG = {
    "ppo_lr": 3e-4,
    "ppo_gamma": 0.99,
    "ppo_gae_lambda": 0.95,
    "ppo_clip_epsilon": 0.2,
    "ppo_value_coef": 0.5,
    "ppo_entropy_coef": 0.01,
    "ppo_max_grad_norm": 0.5,
    "ppo_rollout_length": 512,
    "ppo_update_epochs": 4,
    "ppo_batch_size": 64,
    "rl_training_steps": 5000,
    "initial_balance": 100000.0,
    "transaction_cost": 0.001,
    "max_position": 1.0,
    "dsr_eta": 0.1,
    "run_training": True,
    "device": "cpu",
}


@dataclass
class ExpertSpec:
    name: str
    version: str
    lookback: int
    feature_mode: str
    architecture: str
    ensemble_size: int
    batch_size: int
    eval_batch_size: int
    low_vram: bool = False
    amp_enabled: bool = False
    use_regime: bool = False
    use_retrieval: bool = False
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    max_val_windows: int = MAX_VAL_WINDOWS_PER_SLICE
    retrieval_max_patterns: Optional[int] = None


FORECAST_SPECS = [
    ExpertSpec("v8_5", "v8.5", 64, "core", "gru", ensemble_size=20, batch_size=256, eval_batch_size=256),
    ExpertSpec("v9_1", "v9.1", 160, "technical", "gru", ensemble_size=20, batch_size=256, eval_batch_size=256),
    ExpertSpec("v9_2", "v9.2", 256, "regime", "gru", ensemble_size=20, batch_size=256, eval_batch_size=256, use_regime=True),
    ExpertSpec(
        "v9_3",
        "v9.3",
        96,
        "core",
        "hybrid_itransformer_gru",
        ensemble_size=8,
        batch_size=32,
        eval_batch_size=32,
        low_vram=True,
        amp_enabled=True,
        d_model=128,
        n_heads=8,
        n_layers=2,
    ),
    ExpertSpec(
        "v9_5",
        "v9.5",
        256,
        "core",
        "gru_rag",
        ensemble_size=8 if LIGHTWEIGHT_V95_ON_LOW_VRAM else 20,
        batch_size=128 if LIGHTWEIGHT_V95_ON_LOW_VRAM else 256,
        eval_batch_size=64 if LIGHTWEIGHT_V95_ON_LOW_VRAM else 256,
        low_vram=LIGHTWEIGHT_V95_ON_LOW_VRAM,
        use_retrieval=True,
        max_val_windows=96 if LIGHTWEIGHT_V95_ON_LOW_VRAM else MAX_VAL_WINDOWS_PER_SLICE,
        retrieval_max_patterns=1024 if LIGHTWEIGHT_V95_ON_LOW_VRAM else None,
    ),
]

ACTIVE_FORECAST_SPECS = [spec for spec in FORECAST_SPECS if spec.name not in SKIPPED_EXPERT_NAMES]

print({
    "symbol": SYMBOL,
    "device": str(DEVICE),
    "low_vram_gpu": LOW_VRAM_GPU,
    "skip_heavy_experts_on_windows_8gb": SKIP_HEAVY_EXPERTS_ON_WINDOWS_8GB,
    "lightweight_v95_on_low_vram": LIGHTWEIGHT_V95_ON_LOW_VRAM,
    "skipped_experts": sorted(SKIPPED_EXPERT_NAMES),
    "artifact_root": str(ARTIFACT_ROOT),
    "forecast_experts": [spec.version for spec in ACTIVE_FORECAST_SPECS],
    "decision_layer": "v9.6 PPO",
})
'@

$cells += New-CodeCell @'
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


for directory in [ARTIFACT_ROOT, SHARED_DIR, MODELS_DIR, ENSEMBLE_DIR, RL_DIR]:
    ensure_dir(directory)


def production_timestamp_utc() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def clear_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in state_dict.items()}


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


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_npz(path: Path, **arrays: Any) -> None:
    np.savez_compressed(path, **arrays)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False, compression=SAVE_PARQUET_COMPRESSION)


def feature_columns_for_spec(spec: ExpertSpec) -> List[str]:
    if spec.feature_mode == "core":
        return CORE_FEATURE_COLUMNS.copy()
    if spec.feature_mode == "technical":
        return CORE_FEATURE_COLUMNS + TECHNICAL_FEATURE_COLUMNS
    if spec.feature_mode == "regime":
        return CORE_FEATURE_COLUMNS + REGIME_FEATURE_COLUMNS
    raise ValueError(f"Unsupported feature mode: {spec.feature_mode}")


def model_dir_for_spec(spec: ExpertSpec) -> Path:
    return ensure_dir(MODELS_DIR / spec.name)


def expert_artifacts_complete(spec: ExpertSpec) -> bool:
    expert_dir = MODELS_DIR / spec.name
    required = [
        expert_dir / "model.pt",
        expert_dir / "scaler.npz",
        expert_dir / "feature_manifest.json",
        expert_dir / "train_config.json",
        expert_dir / "inference_config.json",
        expert_dir / "metrics.json",
        expert_dir / "history.csv",
        expert_dir / "rolling_predictions.npz",
        expert_dir / "rolling_summary.csv",
    ]
    if spec.use_retrieval:
        required.extend([expert_dir / "rag_database.npz", expert_dir / "rag_config.json"])
    return all(path.exists() for path in required)


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


def build_runtime_metadata() -> Dict[str, Any]:
    gpu_mem_gb = None
    if torch.cuda.is_available():
        gpu_mem_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
    return {
        "created_at_utc": production_timestamp_utc(),
        "symbol": SYMBOL,
        "device": str(DEVICE),
        "gpu_mem_gb": gpu_mem_gb,
        "lookback_days": LOOKBACK_DAYS,
        "horizon": HORIZON,
    }
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
        raise RuntimeError("Set ALPACA_API_KEY / ALPACA_API_SECRET before running the training notebook.")
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
            except Exception as exc:  # noqa: BLE001
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
            df[column] = np.nan if column != "volume" and column != "trade_count" else 0.0
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
        session["session_minutes"] = len(session)
        session["session_progress"] = session["bar_in_session"] / max(len(session) - 1, 1)
        frames.append(session)

    if not frames:
        raise RuntimeError("Sessionization removed every row.")

    output = pd.concat(frames, ignore_index=True)
    output["timestamp_ny"] = output["timestamp"].dt.tz_convert(SESSION_TZ)
    output["minute_of_day"] = output["timestamp_ny"].dt.hour * 60 + output["timestamp_ny"].dt.minute
    return output
'@

$cells += New-CodeCell @'
raw_snapshot_path = SHARED_DIR / "raw_snapshot.parquet"
session_snapshot_path = SHARED_DIR / "sessionized_snapshot.parquet"

if REFRESH_SHARED_SNAPSHOT or not raw_snapshot_path.exists() or not session_snapshot_path.exists():
    raw_df_utc, api_calls = fetch_bars_alpaca(SYMBOL, LOOKBACK_DAYS)
    session_df = sessionize_with_calendar(raw_df_utc, skip_open_bars=SESSION_OPEN_SKIP_BARS)
    save_dataframe(raw_df_utc, raw_snapshot_path)
    save_dataframe(session_df, session_snapshot_path)
    save_json(
        SHARED_DIR / "calendar_meta.json",
        {
            **build_runtime_metadata(),
            "api_calls": api_calls,
            "rows_raw": int(len(raw_df_utc)),
            "rows_sessionized": int(len(session_df)),
            "sessions_kept": int(session_df["session_date"].nunique()),
            "session_tz": SESSION_TZ,
            "session_open_skip_bars": SESSION_OPEN_SKIP_BARS,
        },
    )
else:
    raw_df_utc = pd.read_parquet(raw_snapshot_path)
    session_df = pd.read_parquet(session_snapshot_path)

display(session_df.head())
print(
    {
        "rows_raw": len(raw_df_utc),
        "rows_sessionized": len(session_df),
        "sessions": session_df["session_date"].nunique(),
        "first_timestamp": str(session_df["timestamp"].min()),
        "last_timestamp": str(session_df["timestamp"].max()),
    }
)
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
    def __init__(self, price_df: pd.DataFrame) -> None:
        self.df = price_df

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
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.sign(close.diff().fillna(0.0))
        return (direction * volume.fillna(0.0)).cumsum()


def calculate_technical_features(core_df: pd.DataFrame) -> pd.DataFrame:
    calc = TechnicalIndicatorCalculator(core_df)
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

    out = out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return out


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
        feature_df["turbulence_60"] = compute_turbulence_index(feature_df["returns"], lookback=60)
        feature_df["regime_indicator"] = build_dynamic_regime_indicator(feature_df["atr_14_pct"], feature_df["turbulence_60"])
    elif mode != "core":
        raise ValueError(f"Unsupported feature mode: {mode}")

    if "turbulence_60" not in feature_df.columns:
        feature_df["turbulence_60"] = compute_turbulence_index(feature_df["returns"], lookback=60)
    if "atr_14" not in feature_df.columns:
        feature_df["atr_14"] = technical_df["atr_14"]
    if "atr_14_pct" not in feature_df.columns:
        feature_df["atr_14_pct"] = technical_df["atr_14_pct"]
    if "regime_indicator" not in feature_df.columns:
        feature_df["regime_indicator"] = build_dynamic_regime_indicator(feature_df["atr_14_pct"], feature_df["turbulence_60"])

    feature_df["relative_volume"] = feature_df["volume"] / feature_df["volume"].rolling(20, min_periods=5).mean().replace(0.0, np.nan)
    feature_df["relative_volume"] = feature_df["relative_volume"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return feature_df


FEATURE_FRAMES = {
    "core": build_feature_frame_for_mode(session_df, "core"),
    "technical": build_feature_frame_for_mode(session_df, "technical"),
    "regime": build_feature_frame_for_mode(session_df, "regime"),
}

for mode, frame in FEATURE_FRAMES.items():
    print(mode, frame.shape, frame[feature_columns_for_spec(ACTIVE_FORECAST_SPECS[0])[:4]].head(2).to_dict("records"))
'@

$cells += New-CodeCell @'
def build_walkforward_slices(n_rows: int) -> List[Dict[str, int]]:
    first_end = int(round(n_rows * 0.85))
    second_start = max(0, int(round(n_rows * WALKFORWARD_SLICE_OVERLAP)))
    return [
        {"name": "slice_1", "start": 0, "end": first_end},
        {"name": "slice_2", "start": second_start, "end": n_rows},
    ]


def make_multistep_windows(
    feature_df: pd.DataFrame,
    feature_columns: Sequence[str],
    lookback: int,
    horizon: int,
    anchor_start: int,
    anchor_end: int,
) -> Dict[str, Any]:
    x_windows: List[np.ndarray] = []
    y_windows: List[np.ndarray] = []
    anchor_prev_closes: List[float] = []
    anchor_timestamps: List[pd.Timestamp] = []
    context_closes: List[np.ndarray] = []

    for anchor in range(max(anchor_start, lookback), anchor_end - horizon + 1):
        context_slice = feature_df.iloc[anchor - lookback : anchor]
        target_slice = feature_df.iloc[anchor : anchor + horizon]
        if len(context_slice) != lookback or len(target_slice) != horizon:
            continue
        if target_slice["row_imputed"].any() or target_slice["row_open_skip"].any():
            continue
        feature_block = context_slice.loc[:, feature_columns].to_numpy(dtype=np.float32)
        if not np.isfinite(feature_block).all():
            continue
        impute_frac = float(context_slice["row_imputed"].mean())
        impute_channel = np.full((lookback, 1), impute_frac, dtype=np.float32)
        x_windows.append(np.concatenate([feature_block, impute_channel], axis=1))
        y_windows.append(target_slice.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float32))
        anchor_prev_closes.append(float(target_slice["prev_close"].iloc[0]))
        anchor_timestamps.append(pd.Timestamp(target_slice["timestamp"].iloc[0]))
        context_closes.append(context_slice["close"].to_numpy(dtype=np.float32))

    if not x_windows:
        raise RuntimeError("No valid windows were generated for the requested slice.")

    return {
        "x": np.stack(x_windows).astype(np.float32),
        "y": np.stack(y_windows).astype(np.float32),
        "prev_close": np.asarray(anchor_prev_closes, dtype=np.float32),
        "anchor_timestamps": anchor_timestamps,
        "context_closes": np.stack(context_closes).astype(np.float32),
    }


def fit_input_scaler(train_x: np.ndarray) -> Dict[str, np.ndarray]:
    mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def apply_input_scaler(x: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
    return ((x - scaler["mean"]) / scaler["std"]).astype(np.float32)


class MultiStepDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def build_fold_tensors(feature_df: pd.DataFrame, spec: ExpertSpec, slice_cfg: Dict[str, int]) -> Dict[str, Any]:
    feature_columns = feature_columns_for_spec(spec)
    start = slice_cfg["start"]
    end = slice_cfg["end"]
    train_cut = int(round(start + (end - start) * (1.0 - VALIDATION_SLICE_FRACTION)))

    train_bundle = make_multistep_windows(feature_df, feature_columns, spec.lookback, HORIZON, start + spec.lookback, train_cut)
    val_bundle = make_multistep_windows(feature_df, feature_columns, spec.lookback, HORIZON, train_cut, end)

    scaler = fit_input_scaler(train_bundle["x"])
    train_bundle["x_scaled"] = apply_input_scaler(train_bundle["x"], scaler)
    val_bundle["x_scaled"] = apply_input_scaler(val_bundle["x"], scaler)

    return {
        "train": train_bundle,
        "val": val_bundle,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "slice": slice_cfg,
    }


def build_production_split(feature_df: pd.DataFrame, spec: ExpertSpec, cutoff_index: int) -> Dict[str, Any]:
    feature_columns = feature_columns_for_spec(spec)
    val_anchor_start = max(spec.lookback, int(round(cutoff_index * (1.0 - PRODUCTION_VAL_FRACTION))))
    train_bundle = make_multistep_windows(feature_df, feature_columns, spec.lookback, HORIZON, spec.lookback, val_anchor_start)
    val_bundle = make_multistep_windows(feature_df, feature_columns, spec.lookback, HORIZON, val_anchor_start, cutoff_index)
    scaler = fit_input_scaler(train_bundle["x"])
    train_bundle["x_scaled"] = apply_input_scaler(train_bundle["x"], scaler)
    val_bundle["x_scaled"] = apply_input_scaler(val_bundle["x"], scaler)
    return {
        "train": train_bundle,
        "val": val_bundle,
        "scaler": scaler,
        "feature_columns": feature_columns,
    }
'@

$cells += New-CodeCell @'
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
        self.hidden_size = hidden_size
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

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, teacher_forcing_ratio: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        encoder_memory, decoder_hidden = self.encode_sequence(x)
        decoder_input = x[:, -1, :4]
        mu_steps: List[torch.Tensor] = []
        log_sigma_steps: List[torch.Tensor] = []
        for step in range(self.horizon):
            mu, log_sigma, decoder_hidden = self.decode_step(decoder_input, decoder_hidden, encoder_memory)
            mu_steps.append(mu)
            log_sigma_steps.append(log_sigma)
            if self.training and y is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = y[:, step, :]
            else:
                decoder_input = mu
        return torch.stack(mu_steps, dim=1), torch.stack(log_sigma_steps, dim=1)


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
        tokens = self.norm(tokens)
        return tokens.flatten(start_dim=1)


class HybridSeq2SeqForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        lookback: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        horizon: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
    ) -> None:
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

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, teacher_forcing_ratio: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_memory, decoder_hidden = self.encode_sequence(x)
        decoder_input = x[:, -1, :4]
        mu_steps: List[torch.Tensor] = []
        log_sigma_steps: List[torch.Tensor] = []
        for step in range(self.horizon):
            mu, log_sigma, decoder_hidden = self.decode_step(decoder_input, decoder_hidden, encoder_memory)
            mu_steps.append(mu)
            log_sigma_steps.append(log_sigma)
            if self.training and y is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = y[:, step, :]
            else:
                decoder_input = mu
        return torch.stack(mu_steps, dim=1), torch.stack(log_sigma_steps, dim=1)


def build_model_for_spec(spec: ExpertSpec, input_dim: int) -> nn.Module:
    if spec.architecture == "gru" or spec.architecture == "gru_rag":
        return Seq2SeqAttnGRU(
            input_dim=input_dim,
            hidden_size=TRAINING_DEFAULTS["hidden_size"],
            num_layers=TRAINING_DEFAULTS["num_layers"],
            dropout=TRAINING_DEFAULTS["dropout"],
            horizon=HORIZON,
        )
    if spec.architecture == "hybrid_itransformer_gru":
        return HybridSeq2SeqForecaster(
            input_dim=input_dim,
            lookback=spec.lookback,
            hidden_size=TRAINING_DEFAULTS["hidden_size"],
            num_layers=TRAINING_DEFAULTS["num_layers"],
            dropout=TRAINING_DEFAULTS["dropout"],
            horizon=HORIZON,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            n_layers=spec.n_layers,
        )
    raise ValueError(f"Unsupported architecture: {spec.architecture}")


def rag_config_for_spec(spec: ExpertSpec) -> Dict[str, Any]:
    config = dict(RAG_CONFIG)
    if spec.retrieval_max_patterns is not None:
        config["max_patterns"] = int(spec.retrieval_max_patterns)
    return config


def build_retrieval_artifact(
    model: nn.Module,
    x_scaled: np.ndarray,
    y_returns: np.ndarray,
    max_patterns: int,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    model.eval()
    keep = min(len(x_scaled), max_patterns)
    indices = np.linspace(0, len(x_scaled) - 1, keep, dtype=int)
    x_subset = x_scaled[indices]
    y_subset = y_returns[indices]
    embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_subset), batch_size):
            batch = torch.from_numpy(x_subset[start : start + batch_size]).to(DEVICE)
            encoded = model.encode_context(batch).detach().cpu().numpy().astype(np.float32)
            embeddings.append(encoded)
    embedding_array = np.concatenate(embeddings, axis=0)
    embedding_array /= np.linalg.norm(embedding_array, axis=1, keepdims=True).clip(min=1e-8)
    return {
        "embeddings": embedding_array.astype(np.float32),
        "future_returns": y_subset.astype(np.float32),
        "indices": indices.astype(np.int32),
    }


def retrieve_future_returns(
    model: nn.Module,
    x_scaled_single: np.ndarray,
    retrieval_artifact: Optional[Dict[str, np.ndarray]],
    k_retrieve: int,
    ) -> Optional[np.ndarray]:
    if retrieval_artifact is None or len(retrieval_artifact["embeddings"]) == 0:
        return None

    with torch.no_grad():
        query = torch.from_numpy(x_scaled_single[None, ...]).to(DEVICE)
        query_embedding = model.encode_context(query).detach().cpu().numpy()[0].astype(np.float32)
    query_embedding /= np.linalg.norm(query_embedding).clip(min=1e-8)
    similarities = retrieval_artifact["embeddings"] @ query_embedding
    top_idx = np.argsort(similarities)[-k_retrieve:][::-1]
    top_scores = similarities[top_idx]
    top_weights = np.exp(top_scores - top_scores.max())
    top_weights /= top_weights.sum().clip(min=1e-8)
    return np.tensordot(top_weights, retrieval_artifact["future_returns"][top_idx], axes=(0, 0)).astype(np.float32)


def blend_retrieved_future(generated_returns: np.ndarray, retrieved_future: Optional[np.ndarray], blend_weight: float) -> np.ndarray:
    if retrieved_future is None:
        return generated_returns.astype(np.float32)
    return ((1.0 - blend_weight) * generated_returns + blend_weight * retrieved_future).astype(np.float32)
'@

$cells += New-CodeCell @'
def gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    sigma = torch.exp(log_sigma).clamp(min=INFERENCE_DEFAULTS["min_predicted_vol"])
    base = 0.5 * (((target - mu) / sigma) ** 2 + 2.0 * log_sigma)
    step_weights = torch.linspace(1.0, 2.0, target.size(1), device=target.device).view(1, -1, 1)
    return (base * step_weights).mean()


def candle_range_loss(mu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_range = (mu[:, :, 1] - mu[:, :, 2]).abs()
    true_range = (target[:, :, 1] - target[:, :, 2]).abs()
    return F.l1_loss(pred_range, true_range)


def volatility_match_loss(log_sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_sigma = torch.exp(log_sigma).mean(dim=-1)
    true_vol = target.std(dim=-1, unbiased=False)
    return F.l1_loss(pred_sigma, true_vol)


def directional_penalty(mu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_close = mu[:, :, 3]
    true_close = target[:, :, 3]
    return F.relu(-(pred_close * true_close)).mean()


def total_loss(mu: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss_weights = TRAINING_DEFAULTS["loss_weights"]
    return (
        gaussian_nll(mu, log_sigma, target)
        + loss_weights["range"] * candle_range_loss(mu, target)
        + loss_weights["volatility"] * volatility_match_loss(log_sigma, target)
        + loss_weights["directional"] * directional_penalty(mu, target)
    )


def tf_ratio_for_epoch(epoch: int) -> float:
    return float(TRAINING_DEFAULTS["teacher_forcing_decay"] ** epoch)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer] = None,
    teacher_forcing_ratio: float = 1.0,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    amp_enabled: bool = False,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses: List[float] = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        autocast_enabled = amp_enabled and DEVICE.type == "cuda"
        autocast_device = "cuda" if DEVICE.type == "cuda" else "cpu"

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=autocast_device, enabled=autocast_enabled):
            mu, log_sigma = model(x_batch, y=y_batch, teacher_forcing_ratio=teacher_forcing_ratio)
            loss = total_loss(mu, log_sigma, y_batch)

        if training:
            if scaler is not None and autocast_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_DEFAULTS["gradient_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_DEFAULTS["gradient_clip_norm"])
                optimizer.step()

        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def make_grad_scaler(enabled: bool):
    if DEVICE.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_model(
    spec: ExpertSpec,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
) -> Tuple[nn.Module, pd.DataFrame]:
    input_dim = train_x.shape[-1]
    model = build_model_for_spec(spec, input_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_DEFAULTS["learning_rate"],
        weight_decay=TRAINING_DEFAULTS["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=TRAINING_DEFAULTS["scheduler_factor"],
        patience=TRAINING_DEFAULTS["scheduler_patience"],
        min_lr=TRAINING_DEFAULTS["scheduler_min_lr"],
    )
    scaler = make_grad_scaler(enabled=spec.amp_enabled and DEVICE.type == "cuda")

    train_loader = DataLoader(MultiStepDataset(train_x, train_y), batch_size=spec.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(MultiStepDataset(val_x, val_y), batch_size=spec.eval_batch_size, shuffle=False, drop_last=False)

    best_state = None
    best_val = float("inf")
    patience_left = TRAINING_DEFAULTS["final_patience"]
    history_rows: List[Dict[str, float]] = []

    for epoch in range(TRAINING_DEFAULTS["final_max_epochs"]):
        teacher_forcing_ratio = tf_ratio_for_epoch(epoch)
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            teacher_forcing_ratio=teacher_forcing_ratio,
            scaler=scaler,
            amp_enabled=spec.amp_enabled,
        )
        with torch.no_grad():
            val_loss = run_epoch(
                model,
                val_loader,
                optimizer=None,
                teacher_forcing_ratio=0.0,
                scaler=None,
                amp_enabled=spec.amp_enabled,
            )
        scheduler.step(val_loss)
        history_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "teacher_forcing_ratio": teacher_forcing_ratio,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy(state_dict_to_cpu(model.state_dict()))
            patience_left = TRAINING_DEFAULTS["final_patience"]
        else:
            patience_left -= 1
        if patience_left <= 0:
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint.")

    model.load_state_dict(best_state)
    history_df = pd.DataFrame(history_rows)
    return model, history_df


def predict_mean_in_batches(model: nn.Module, x_scaled: np.ndarray, batch_size: int, amp_enabled: bool) -> np.ndarray:
    outputs: List[np.ndarray] = []
    model.eval()
    autocast_device = "cuda" if DEVICE.type == "cuda" else "cpu"
    with torch.no_grad():
        for start in range(0, len(x_scaled), batch_size):
            batch = torch.from_numpy(x_scaled[start : start + batch_size]).to(DEVICE)
            with torch.autocast(device_type=autocast_device, enabled=amp_enabled and DEVICE.type == "cuda"):
                mu, _ = model(batch, y=None, teacher_forcing_ratio=0.0)
            outputs.append(mu.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def calculate_trend_slope(close_seq: np.ndarray) -> float:
    if len(close_seq) < 2:
        return 0.0
    x_axis = np.arange(len(close_seq), dtype=np.float32)
    slope, _ = np.polyfit(x_axis, close_seq.astype(np.float32), 1)
    return float(slope / max(abs(close_seq[-1]), 1e-6))


def select_best_path_by_trend(historical_closes: np.ndarray, candidate_paths: np.ndarray) -> np.ndarray:
    history_slice = historical_closes[-INFERENCE_DEFAULTS["trend_lookback_bars"] :]
    historical_slope = calculate_trend_slope(history_slice)
    candidate_slopes = np.asarray([calculate_trend_slope(path[:, 3]) for path in candidate_paths], dtype=np.float32)
    if abs(historical_slope) >= INFERENCE_DEFAULTS["strong_trend_threshold"]:
        same_sign = np.sign(candidate_slopes) == np.sign(historical_slope)
        filtered_idx = np.where(same_sign)[0]
        if len(filtered_idx) > 0:
            candidate_paths = candidate_paths[filtered_idx]
            candidate_slopes = candidate_slopes[filtered_idx]
    best_idx = int(np.argmin(np.abs(candidate_slopes - historical_slope)))
    return candidate_paths[best_idx]


def generate_ensemble_with_trend_selection(
    model: nn.Module,
    spec: ExpertSpec,
    x_scaled_single: np.ndarray,
    anchor_prev_close: float,
    historical_closes: np.ndarray,
    temperature: float,
    retrieval_artifact: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    model.eval()
    x_tensor = torch.from_numpy(x_scaled_single[None, ...]).to(DEVICE)
    with torch.no_grad():
        encoder_memory, decoder_hidden_init = model.encode_sequence(x_tensor)
    decoder_input_init = x_tensor[:, -1, :4]
    candidate_paths: List[np.ndarray] = []
    retrieved_future = None

    if retrieval_artifact is not None:
        retrieved_future = retrieve_future_returns(
            model=model,
            x_scaled_single=x_scaled_single,
            retrieval_artifact=retrieval_artifact,
            k_retrieve=RAG_CONFIG["k_retrieve"],
        )

    for seed_offset in range(spec.ensemble_size):
        torch.manual_seed(SEED + seed_offset)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + seed_offset)
        decoder_hidden = decoder_hidden_init.clone()
        decoder_input = decoder_input_init.clone()
        sampled_steps: List[np.ndarray] = []
        with torch.no_grad():
            for _ in range(HORIZON):
                mu, log_sigma, decoder_hidden = model.decode_step(decoder_input, decoder_hidden, encoder_memory)
                sigma = torch.exp(log_sigma).clamp(min=INFERENCE_DEFAULTS["min_predicted_vol"]) * temperature
                sample = mu + torch.randn_like(mu) * sigma
                sampled_steps.append(sample.squeeze(0).cpu().numpy().astype(np.float32))
                decoder_input = sample
        sampled_returns = np.stack(sampled_steps).astype(np.float32)
        sampled_returns = blend_retrieved_future(
            generated_returns=sampled_returns,
            retrieved_future=retrieved_future,
            blend_weight=RAG_CONFIG["blend_weight"],
        )
        candidate_paths.append(returns_to_prices_seq(anchor_prev_close, sampled_returns))

    candidate_paths_array = np.stack(candidate_paths).astype(np.float32)
    return select_best_path_by_trend(historical_closes, candidate_paths_array)


def evaluate_metrics(pred_paths: np.ndarray, actual_paths: np.ndarray, anchor_prev_close: np.ndarray) -> Dict[str, float]:
    pred_close = pred_paths[:, :, 3]
    actual_close = actual_paths[:, :, 3]
    step_1_mae = float(np.mean(np.abs(pred_close[:, 0] - actual_close[:, 0])))
    horizon_mae = float(np.mean(np.abs(pred_close[:, -1] - actual_close[:, -1])))
    path_mae = float(np.mean(np.abs(pred_close - actual_close)))
    pred_direction = np.sign(pred_close[:, 0] - anchor_prev_close)
    actual_direction = np.sign(actual_close[:, 0] - anchor_prev_close)
    directional_accuracy = float(np.mean(pred_direction == actual_direction))
    return {
        "step_1_close_mae": step_1_mae,
        "horizon_close_mae": horizon_mae,
        "path_close_mae": path_mae,
        "directional_accuracy": directional_accuracy,
    }


def evaluate_persistence_baseline(actual_paths: np.ndarray, anchor_prev_close: np.ndarray) -> Dict[str, float]:
    baseline = np.repeat(anchor_prev_close[:, None, None], HORIZON, axis=1)
    baseline = np.repeat(baseline, 4, axis=2).astype(np.float32)
    return evaluate_metrics(baseline, actual_paths, anchor_prev_close)
'@

$cells += New-CodeCell @'
def summarize_slice_results(slice_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_keys = ["step_1_close_mae", "horizon_close_mae", "path_close_mae", "directional_accuracy"]
    summary = {}
    for key in metric_keys:
        summary[key] = float(np.mean([result["metrics"][key] for result in slice_results]))
        summary[f"baseline_{key}"] = float(np.mean([result["baseline_metrics"][key] for result in slice_results]))
    summary["slice_count"] = len(slice_results)
    summary["val_windows_used"] = int(np.sum([result["val_windows_used"] for result in slice_results]))
    return summary


def run_walkforward_for_spec(feature_df: pd.DataFrame, spec: ExpertSpec) -> Dict[str, Any]:
    slice_results: List[Dict[str, Any]] = []
    walkforward_slices = build_walkforward_slices(len(feature_df))
    for slice_cfg in walkforward_slices:
        fold = build_fold_tensors(feature_df, spec, slice_cfg)
        model, history_df = train_model(
            spec,
            fold["train"]["x_scaled"],
            fold["train"]["y"],
            fold["val"]["x_scaled"],
            fold["val"]["y"],
        )

        limit = min(spec.max_val_windows, len(fold["val"]["x_scaled"]))
        pred_paths: List[np.ndarray] = []
        actual_paths: List[np.ndarray] = []
        anchor_prev_close = fold["val"]["prev_close"][:limit]
        for index in range(limit):
            pred_paths.append(
                generate_ensemble_with_trend_selection(
                    model=model,
                    spec=spec,
                    x_scaled_single=fold["val"]["x_scaled"][index],
                    anchor_prev_close=float(fold["val"]["prev_close"][index]),
                    historical_closes=fold["val"]["context_closes"][index],
                    temperature=INFERENCE_DEFAULTS["sampling_temperature"],
                )
            )
            actual_paths.append(
                returns_to_prices_seq(
                    float(fold["val"]["prev_close"][index]),
                    fold["val"]["y"][index],
                )
            )

        pred_paths_arr = np.stack(pred_paths).astype(np.float32)
        actual_paths_arr = np.stack(actual_paths).astype(np.float32)
        metrics = evaluate_metrics(pred_paths_arr, actual_paths_arr, anchor_prev_close)
        baseline_metrics = evaluate_persistence_baseline(actual_paths_arr, anchor_prev_close)
        slice_results.append(
            {
                "slice_name": slice_cfg["name"],
                "metrics": metrics,
                "baseline_metrics": baseline_metrics,
                "val_windows_used": limit,
            }
        )
        model = model.to("cpu")
        del model, history_df, fold, pred_paths, actual_paths, pred_paths_arr, actual_paths_arr, metrics, baseline_metrics
        clear_torch_memory()
    return {
        "slice_results": slice_results,
        "summary": summarize_slice_results(slice_results),
        "walkforward_slices": walkforward_slices,
    }


def intraday_temperature(anchor_ts: pd.Timestamp) -> float:
    local_ts = pd.Timestamp(anchor_ts).tz_convert(SESSION_TZ)
    hhmm = local_ts.hour * 100 + local_ts.minute
    if hhmm < 1015:
        return 1.25
    if hhmm < 1400:
        return 1.45
    return 1.60


def select_backtest_date(session_df: pd.DataFrame, max_lookback: int, horizon: int) -> str:
    dates = session_df["session_date"].drop_duplicates().tolist()
    for session_date in reversed(dates):
        date_mask = session_df["session_date"] == session_date
        if date_mask.sum() < horizon + 30:
            continue
        first_index = int(np.flatnonzero(date_mask.to_numpy())[0])
        if first_index >= max_lookback:
            return session_date
    raise RuntimeError("No valid backtest date found.")


def train_production_model_for_spec(feature_df: pd.DataFrame, spec: ExpertSpec, cutoff_index: int) -> Dict[str, Any]:
    prod_split = build_production_split(feature_df.iloc[:cutoff_index].reset_index(drop=True), spec, cutoff_index=cutoff_index)
    model, history_df = train_model(
        spec,
        prod_split["train"]["x_scaled"],
        prod_split["train"]["y"],
        prod_split["val"]["x_scaled"],
        prod_split["val"]["y"],
    )
    retrieval_artifact = None
    if spec.use_retrieval:
        rag_config = rag_config_for_spec(spec)
        combined_x = np.concatenate([prod_split["train"]["x_scaled"], prod_split["val"]["x_scaled"]], axis=0)
        combined_y = np.concatenate([prod_split["train"]["y"], prod_split["val"]["y"]], axis=0)
        retrieval_artifact = build_retrieval_artifact(
            model=model,
            x_scaled=combined_x,
            y_returns=combined_y,
            max_patterns=rag_config["max_patterns"],
            batch_size=spec.eval_batch_size,
        )
    return {
        "model": model,
        "history": history_df,
        "scaler": prod_split["scaler"],
        "feature_columns": prod_split["feature_columns"],
        "retrieval_artifact": retrieval_artifact,
    }


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


def run_rolling_for_spec(
    feature_df: pd.DataFrame,
    spec: ExpertSpec,
    backtest_date: str,
    production_bundle: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    feature_columns = production_bundle["feature_columns"]
    scaler = production_bundle["scaler"]
    model = production_bundle["model"]
    retrieval_artifact = production_bundle["retrieval_artifact"]
    model.eval()

    session_indices = np.where(feature_df["session_date"].to_numpy() == backtest_date)[0]
    if len(session_indices) == 0:
        raise RuntimeError(f"No rows found for backtest date {backtest_date}")

    logs: List[Dict[str, Any]] = []
    first_anchor = session_indices[0] + SESSION_OPEN_SKIP_BARS
    last_anchor = session_indices[-1] - HORIZON

    for anchor in tqdm(range(max(first_anchor, spec.lookback), last_anchor + 1), desc=f"{spec.version} rolling"):
        context = feature_df.iloc[anchor - spec.lookback : anchor]
        feature_block = context.loc[:, feature_columns].to_numpy(dtype=np.float32)
        impute_frac = float(context["row_imputed"].mean())
        feature_block = np.concatenate([feature_block, np.full((spec.lookback, 1), impute_frac, dtype=np.float32)], axis=1)
        feature_block = apply_input_scaler(feature_block[None, ...], scaler)[0]

        anchor_prev_close = float(feature_df["prev_close"].iloc[anchor])
        actual_returns = feature_df.iloc[anchor : anchor + HORIZON].loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float32)
        actual_path = returns_to_prices_seq(anchor_prev_close, actual_returns)
        base_temp = intraday_temperature(pd.Timestamp(feature_df["timestamp"].iloc[anchor]))
        regime_name = "NORMAL"
        regime_indicator = 0.0
        regime_multiplier = 1.0
        if spec.use_regime:
            regime_name, regime_multiplier, regime_indicator = detect_regime_multiplier(feature_df.iloc[max(0, anchor - 390) : anchor])
        predicted_path = generate_ensemble_with_trend_selection(
            model=model,
            spec=spec,
            x_scaled_single=feature_block,
            anchor_prev_close=anchor_prev_close,
            historical_closes=context["close"].to_numpy(dtype=np.float32),
            temperature=base_temp * regime_multiplier,
            retrieval_artifact=retrieval_artifact,
        )
        logs.append(
            {
                "timestamp": pd.Timestamp(feature_df["timestamp"].iloc[anchor]),
                "prev_close": anchor_prev_close,
                "predicted_path": predicted_path,
                "actual_path": actual_path,
                "step_1_close_mae": float(abs(predicted_path[0, 3] - actual_path[0, 3])),
                "step_5_close_mae": float(abs(predicted_path[4, 3] - actual_path[4, 3])),
                "step_10_close_mae": float(abs(predicted_path[9, 3] - actual_path[9, 3])),
                "step_15_close_mae": float(abs(predicted_path[14, 3] - actual_path[14, 3])),
                "direction_hit": float(np.sign(predicted_path[0, 3] - anchor_prev_close) == np.sign(actual_path[0, 3] - anchor_prev_close)),
                "regime_name": regime_name,
                "regime_indicator": regime_indicator,
                "temperature": base_temp * regime_multiplier,
            }
        )

    rolling_df = pd.DataFrame(
        [
            {
                "timestamp": row["timestamp"],
                "prev_close": row["prev_close"],
                "step_1_close_mae": row["step_1_close_mae"],
                "step_5_close_mae": row["step_5_close_mae"],
                "step_10_close_mae": row["step_10_close_mae"],
                "step_15_close_mae": row["step_15_close_mae"],
                "direction_hit": row["direction_hit"],
                "regime_name": row["regime_name"],
                "regime_indicator": row["regime_indicator"],
                "temperature": row["temperature"],
            }
            for row in logs
        ]
    )
    summary = {
        "rolling_predictions": int(len(rolling_df)),
        "directional_hit_rate": float(rolling_df["direction_hit"].mean()),
        "step_1_close_mae": float(rolling_df["step_1_close_mae"].mean()),
        "step_5_close_mae": float(rolling_df["step_5_close_mae"].mean()),
        "step_10_close_mae": float(rolling_df["step_10_close_mae"].mean()),
        "step_15_close_mae": float(rolling_df["step_15_close_mae"].mean()),
    }
    arrays = {
        "timestamps": np.array([pd.Timestamp(row["timestamp"]).isoformat() for row in logs]),
        "predicted_paths": np.stack([row["predicted_path"] for row in logs]).astype(np.float32),
        "actual_paths": np.stack([row["actual_path"] for row in logs]).astype(np.float32),
        "prev_close": np.asarray([row["prev_close"] for row in logs], dtype=np.float32),
        "regime_indicator": np.asarray([row["regime_indicator"] for row in logs], dtype=np.float32),
    }
    return rolling_df, {"summary": summary, "arrays": arrays}


def compute_ensemble_weights(validation_summary_df: pd.DataFrame, rolling_summary_df: pd.DataFrame) -> pd.DataFrame:
    validation_renamed = validation_summary_df.rename(
        columns={column: f"{column}_val" for column in validation_summary_df.columns if column != "expert"}
    )
    rolling_renamed = rolling_summary_df.rename(
        columns={column: f"{column}_rolling" for column in rolling_summary_df.columns if column != "expert"}
    )
    merged = validation_renamed.merge(rolling_renamed, on="expert")
    merged["reliability_score"] = 1.0 / (
        1e-6
        + 0.45 * merged["step_1_close_mae_val"]
        + 0.25 * merged["path_close_mae_val"]
        + 0.10 * (1.0 - merged["directional_accuracy_val"])
        + 0.20 * merged["step_1_close_mae_rolling"]
    )
    merged["weight"] = merged["reliability_score"] / merged["reliability_score"].sum()
    return merged[["expert", "weight", "reliability_score"]].sort_values("weight", ascending=False).reset_index(drop=True)


def save_expert_artifacts(
    spec: ExpertSpec,
    walkforward_results: Dict[str, Any],
    production_bundle: Dict[str, Any],
    rolling_df: pd.DataFrame,
    rolling_payload: Dict[str, Any],
    backtest_date: str,
) -> Dict[str, Any]:
    expert_dir = model_dir_for_spec(spec)
    model = production_bundle["model"]
    scaler = production_bundle["scaler"]

    torch.save(
        {
            "spec": asdict(spec),
            "state_dict": state_dict_to_cpu(model.state_dict()),
            "input_dim": int(len(production_bundle["feature_columns"]) + 1),
            "training_defaults": TRAINING_DEFAULTS,
        },
        expert_dir / "model.pt",
    )
    save_npz(expert_dir / "scaler.npz", mean=scaler["mean"], std=scaler["std"])
    save_json(
        expert_dir / "feature_manifest.json",
        {
            "feature_mode": spec.feature_mode,
            "feature_columns": production_bundle["feature_columns"],
            "input_channels_with_impute": len(production_bundle["feature_columns"]) + 1,
            "target_columns": TARGET_COLUMNS,
        },
    )
    save_json(
        expert_dir / "train_config.json",
        {
            "spec": asdict(spec),
            "training_defaults": TRAINING_DEFAULTS,
            "runtime": build_runtime_metadata(),
        },
    )
    save_json(
        expert_dir / "inference_config.json",
        {
            "lookback": spec.lookback,
            "horizon": HORIZON,
            "ensemble_size": spec.ensemble_size,
            **INFERENCE_DEFAULTS,
        },
    )
    save_json(
        expert_dir / "metrics.json",
        {
            "walkforward_summary": walkforward_results["summary"],
            "rolling_summary": rolling_payload["summary"],
            "backtest_date": backtest_date,
        },
    )
    save_dataframe(production_bundle["history"], expert_dir / "history.csv")
    save_dataframe(rolling_df, expert_dir / "rolling_summary.csv")
    save_npz(expert_dir / "rolling_predictions.npz", **rolling_payload["arrays"])

    if spec.use_retrieval and production_bundle["retrieval_artifact"] is not None:
        save_npz(expert_dir / "rag_database.npz", **production_bundle["retrieval_artifact"])
        save_json(expert_dir / "rag_config.json", rag_config_for_spec(spec))

    return {
        "expert": spec.name,
        "dir": expert_dir,
        "walkforward_summary": walkforward_results["summary"],
        "rolling_summary": rolling_payload["summary"],
    }


def load_saved_rolling_arrays(rolling_predictions_path: Path) -> Dict[str, np.ndarray]:
    with np.load(rolling_predictions_path) as payload:
        return {key: payload[key] for key in payload.files}
'@

$cells += New-CodeCell @'
MAX_LOOKBACK = max(spec.lookback for spec in ACTIVE_FORECAST_SPECS)
BACKTEST_DATE = select_backtest_date(session_df, max_lookback=MAX_LOOKBACK, horizon=HORIZON)
BACKTEST_CUTOFF_INDEX = int(np.flatnonzero(session_df["session_date"].to_numpy() == BACKTEST_DATE)[0])

save_json(
    SHARED_DIR / "split_spec.json",
    {
        "backtest_date": BACKTEST_DATE,
        "backtest_cutoff_index": BACKTEST_CUTOFF_INDEX,
        "walkforward_slices": build_walkforward_slices(len(session_df)),
        "production_val_fraction": PRODUCTION_VAL_FRACTION,
        "validation_slice_fraction": VALIDATION_SLICE_FRACTION,
    },
)

EXPERT_ARTIFACTS: Dict[str, Dict[str, Any]] = {}
validation_rows: List[Dict[str, Any]] = []
rolling_rows: List[Dict[str, Any]] = []

for spec in ACTIVE_FORECAST_SPECS:
    if RESUME_COMPLETED_EXPERTS and expert_artifacts_complete(spec):
        expert_dir = model_dir_for_spec(spec)
        saved_metrics = load_json(expert_dir / "metrics.json")
        print(f"Skipping {spec.version}; found complete saved artifacts in {expert_dir}")
        EXPERT_ARTIFACTS[spec.name] = {
            "spec": asdict(spec),
            "artifact_dir": str(expert_dir),
            "rolling_predictions_path": str(expert_dir / "rolling_predictions.npz"),
            "walkforward_summary": saved_metrics["walkforward_summary"],
            "rolling_summary": saved_metrics["rolling_summary"],
            "artifact_info": {
                "expert": spec.name,
                "dir": expert_dir,
                "walkforward_summary": saved_metrics["walkforward_summary"],
                "rolling_summary": saved_metrics["rolling_summary"],
            },
        }
        validation_rows.append({"expert": spec.name, **saved_metrics["walkforward_summary"]})
        rolling_rows.append({"expert": spec.name, **saved_metrics["rolling_summary"]})
        continue

    print(f"Training {spec.version} with lookback={spec.lookback}, feature_mode={spec.feature_mode}, architecture={spec.architecture}")
    feature_df = FEATURE_FRAMES[spec.feature_mode]
    walkforward_results = run_walkforward_for_spec(feature_df.iloc[:BACKTEST_CUTOFF_INDEX].reset_index(drop=True), spec)
    production_bundle = train_production_model_for_spec(feature_df, spec, cutoff_index=BACKTEST_CUTOFF_INDEX)
    rolling_df, rolling_payload = run_rolling_for_spec(feature_df, spec, BACKTEST_DATE, production_bundle)
    artifact_info = save_expert_artifacts(spec, walkforward_results, production_bundle, rolling_df, rolling_payload, BACKTEST_DATE)
    EXPERT_ARTIFACTS[spec.name] = {
        "spec": asdict(spec),
        "artifact_dir": str(model_dir_for_spec(spec)),
        "rolling_predictions_path": str(model_dir_for_spec(spec) / "rolling_predictions.npz"),
        "walkforward_summary": walkforward_results["summary"],
        "rolling_summary": rolling_payload["summary"],
        "artifact_info": artifact_info,
    }
    validation_rows.append({"expert": spec.name, **walkforward_results["summary"]})
    rolling_rows.append({"expert": spec.name, **rolling_payload["summary"]})
    production_bundle["model"] = production_bundle["model"].to("cpu")
    del production_bundle, rolling_df, rolling_payload, walkforward_results
    clear_torch_memory()

validation_summary_df = pd.DataFrame(validation_rows).sort_values("expert").reset_index(drop=True)
rolling_summary_df = pd.DataFrame(rolling_rows).sort_values("expert").reset_index(drop=True)
ensemble_weights_df = compute_ensemble_weights(validation_summary_df, rolling_summary_df)

save_dataframe(validation_summary_df, ENSEMBLE_DIR / "validation_summary.csv")
save_dataframe(rolling_summary_df, ENSEMBLE_DIR / "rolling_summary.csv")
save_dataframe(ensemble_weights_df, ENSEMBLE_DIR / "weights.csv")
save_json(
    ENSEMBLE_DIR / "weights.json",
    {row["expert"]: float(row["weight"]) for _, row in ensemble_weights_df.iterrows()},
)
save_json(
    ENSEMBLE_DIR / "aggregate_config.json",
    {
        "experts": [spec.name for spec in ACTIVE_FORECAST_SPECS],
        "weights": {row["expert"]: float(row["weight"]) for _, row in ensemble_weights_df.iterrows()},
        "backtest_date": BACKTEST_DATE,
        "aggregation_rule": "reliability_weighted_average",
    },
)

display(validation_summary_df)
display(rolling_summary_df)
display(ensemble_weights_df)
'@

$cells += New-CodeCell @'
def build_rl_market_frame(session_df: pd.DataFrame) -> pd.DataFrame:
    technical_frame = build_feature_frame_for_mode(session_df, "technical")
    market_df = technical_frame.loc[:, ["timestamp", "close", "regime_indicator"] + MARKET_STATE_COLUMNS].copy()
    market_df = market_df.dropna().reset_index(drop=True)
    return market_df


def build_aggregate_rolling_bundle(expert_artifacts: Dict[str, Dict[str, Any]], weights_df: pd.DataFrame) -> Dict[str, Any]:
    weight_map = {row["expert"]: float(row["weight"]) for _, row in weights_df.iterrows()}
    first_artifact = next(iter(expert_artifacts.values()))
    reference = load_saved_rolling_arrays(Path(first_artifact["rolling_predictions_path"]))
    timestamps = reference["timestamps"]
    prev_close = reference["prev_close"]

    weighted_paths = []
    actual_paths = reference["actual_paths"]
    regime_indicators = []
    loaded_payloads = {
        expert_name: load_saved_rolling_arrays(Path(artifact["rolling_predictions_path"]))
        for expert_name, artifact in expert_artifacts.items()
    }
    for idx in range(len(timestamps)):
        combined_path = np.zeros((HORIZON, 4), dtype=np.float32)
        regime_vote = 0.0
        for expert_name, artifact in expert_artifacts.items():
            weight = weight_map[expert_name]
            expert_timestamps = loaded_payloads[expert_name]["timestamps"]
            if not np.array_equal(expert_timestamps, timestamps):
                raise RuntimeError("Rolling prediction timestamps are not aligned across experts.")
            combined_path += weight * loaded_payloads[expert_name]["predicted_paths"][idx]
            regime_vote += weight * loaded_payloads[expert_name]["regime_indicator"][idx]
        weighted_paths.append(enforce_candle_validity(combined_path))
        regime_indicators.append(regime_vote)
    return {
        "timestamps": timestamps,
        "prev_close": prev_close,
        "predicted_paths": np.stack(weighted_paths).astype(np.float32),
        "actual_paths": actual_paths.astype(np.float32),
        "regime_indicator": np.asarray(regime_indicators, dtype=np.float32),
    }


class StockTradingEnv:
    def __init__(self, aggregate_bundle: Dict[str, np.ndarray], market_df: pd.DataFrame, config: Dict[str, Any]) -> None:
        self.aggregate_bundle = aggregate_bundle
        self.market_df = market_df
        self.config = config
        self.forecasts = aggregate_bundle["predicted_paths"]
        self.actual_paths = aggregate_bundle["actual_paths"]
        self.regime_indicator = aggregate_bundle["regime_indicator"]
        self.timestamps = aggregate_bundle["timestamps"]
        self.initial_balance = float(config["initial_balance"])
        self.transaction_cost = float(config["transaction_cost"])
        self.max_position = float(config["max_position"])
        self.dsr_eta = float(config["dsr_eta"])
        self.reset()

    @property
    def state_dim(self) -> int:
        return int(self.forecasts.shape[1] * self.forecasts.shape[2] + len(MARKET_STATE_COLUMNS) + 4 + 1)

    def reset(self) -> np.ndarray:
        self.step_index = 0
        self.cash = self.initial_balance
        self.shares = 0.0
        self.portfolio_value = self.initial_balance
        self.position_pct = 0.0
        self.A = 0.0
        self.B = 0.0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        market_row = self.market_df.iloc[self.step_index]
        market_state = market_row[MARKET_STATE_COLUMNS].to_numpy(dtype=np.float32)
        portfolio_state = np.array(
            [
                self.cash / self.initial_balance,
                self.shares / max(self.initial_balance / max(market_row["close"], 1e-6), 1.0),
                self.portfolio_value / self.initial_balance,
                self.position_pct,
            ],
            dtype=np.float32,
        )
        forecast_state = self.forecasts[self.step_index].reshape(-1).astype(np.float32)
        regime_state = np.array([self.regime_indicator[self.step_index]], dtype=np.float32)
        return np.concatenate([forecast_state, market_state, portfolio_state, regime_state]).astype(np.float32)

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        price = float(self.actual_paths[self.step_index, 0, 3])
        target_pct = float(np.clip(action, -1.0, 1.0)) * self.max_position
        regime_scale = 1.0 if self.regime_indicator[self.step_index] < 0.5 else (0.7 if self.regime_indicator[self.step_index] < 1.0 else 0.3)
        target_pct *= regime_scale
        target_value = target_pct * self.portfolio_value
        current_value = self.shares * price
        trade_value = target_value - current_value

        if abs(trade_value) >= 10.0:
            fee = abs(trade_value) * self.transaction_cost
            self.cash -= trade_value + fee
            self.shares += trade_value / max(price, 1e-6)
        next_price = float(self.actual_paths[self.step_index, 1, 3]) if self.step_index < len(self.actual_paths) - 1 else price
        previous_value = self.portfolio_value
        self.portfolio_value = self.cash + self.shares * next_price
        self.position_pct = 0.0 if self.portfolio_value == 0 else float((self.shares * next_price) / self.portfolio_value)
        portfolio_return = (self.portfolio_value - previous_value) / max(previous_value, 1e-6)

        eta = self.dsr_eta
        delta_A = portfolio_return - self.A
        delta_B = portfolio_return**2 - self.B
        self.A += eta * delta_A
        self.B += eta * delta_B
        variance = max(self.B - self.A**2, 1e-8)
        reward = float(delta_A / math.sqrt(variance))

        self.step_index += 1
        done = self.step_index >= len(self.forecasts) - 1
        next_state = self._get_state() if not done else np.zeros(self.state_dim, dtype=np.float32)
        info = {
            "portfolio_value": self.portfolio_value,
            "position_pct": self.position_pct,
            "regime_indicator": float(self.regime_indicator[min(self.step_index, len(self.regime_indicator) - 1)]),
        }
        return next_state, reward, done, info


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


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for step in reversed(range(len(rewards))):
        next_value = 0.0 if step == len(rewards) - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_value * (1.0 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1.0 - dones[step]) * gae
        advantages[step] = gae
    returns = advantages + values
    return advantages, returns


def train_ppo(env: StockTradingEnv, config: Dict[str, Any]) -> Tuple[ActorCritic, Dict[str, Any]]:
    policy = ActorCritic(env.state_dim).to(config["device"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=config["ppo_lr"])
    training_stats: List[Dict[str, float]] = []

    state = env.reset()
    rollout_states: List[np.ndarray] = []
    rollout_actions: List[np.ndarray] = []
    rollout_log_probs: List[np.ndarray] = []
    rollout_rewards: List[float] = []
    rollout_values: List[float] = []
    rollout_dones: List[float] = []

    for step in tqdm(range(config["rl_training_steps"]), desc="PPO"):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(config["device"])
        with torch.no_grad():
            mean, std, value = policy(state_tensor)
            dist = Normal(mean, std)
            action = torch.clamp(dist.sample(), -1.0, 1.0)
            log_prob = dist.log_prob(action).sum(dim=-1)
        next_state, reward, done, info = env.step(float(action.item()))

        rollout_states.append(state.copy())
        rollout_actions.append(np.array([float(action.item())], dtype=np.float32))
        rollout_log_probs.append(np.array([float(log_prob.item())], dtype=np.float32))
        rollout_rewards.append(float(reward))
        rollout_values.append(float(value.item()))
        rollout_dones.append(float(done))

        state = env.reset() if done else next_state

        if len(rollout_states) >= config["ppo_rollout_length"] or step == config["rl_training_steps"] - 1:
            states = torch.from_numpy(np.stack(rollout_states)).float().to(config["device"])
            actions = torch.from_numpy(np.stack(rollout_actions)).float().to(config["device"])
            old_log_probs = torch.from_numpy(np.stack(rollout_log_probs)).float().to(config["device"]).squeeze(-1)
            rewards = np.asarray(rollout_rewards, dtype=np.float32)
            values = np.asarray(rollout_values, dtype=np.float32)
            dones = np.asarray(rollout_dones, dtype=np.float32)

            advantages, returns = compute_gae(rewards, values, dones, config["ppo_gamma"], config["ppo_gae_lambda"])
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages_t = torch.from_numpy(advantages).float().to(config["device"])
            returns_t = torch.from_numpy(returns).float().to(config["device"])

            for _ in range(config["ppo_update_epochs"]):
                indices = np.arange(len(states))
                np.random.shuffle(indices)
                for start in range(0, len(indices), config["ppo_batch_size"]):
                    batch_idx = indices[start : start + config["ppo_batch_size"]]
                    batch_states = states[batch_idx]
                    batch_actions = actions[batch_idx]
                    batch_old_log_probs = old_log_probs[batch_idx]
                    batch_advantages = advantages_t[batch_idx]
                    batch_returns = returns_t[batch_idx]

                    mean, std, values_pred = policy(batch_states)
                    dist = Normal(mean, std)
                    new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()

                    ratios = torch.exp(new_log_probs - batch_old_log_probs)
                    surrogate_1 = ratios * batch_advantages
                    surrogate_2 = torch.clamp(ratios, 1.0 - config["ppo_clip_epsilon"], 1.0 + config["ppo_clip_epsilon"]) * batch_advantages
                    actor_loss = -torch.min(surrogate_1, surrogate_2).mean()
                    critic_loss = F.mse_loss(values_pred.squeeze(-1), batch_returns)
                    loss = actor_loss + config["ppo_value_coef"] * critic_loss - config["ppo_entropy_coef"] * entropy

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), config["ppo_max_grad_norm"])
                    optimizer.step()

            training_stats.append(
                {
                    "step": step + 1,
                    "rollout_reward_mean": float(np.mean(rollout_rewards)),
                    "portfolio_value": float(info["portfolio_value"]),
                }
            )
            rollout_states.clear()
            rollout_actions.clear()
            rollout_log_probs.clear()
            rollout_rewards.clear()
            rollout_values.clear()
            rollout_dones.clear()

    return policy, {"stats": training_stats}
'@

$cells += New-CodeCell @'
aggregate_bundle = build_aggregate_rolling_bundle(EXPERT_ARTIFACTS, ensemble_weights_df)
save_npz(RL_DIR / "aggregate_predictions.npz", **aggregate_bundle)

market_df = build_rl_market_frame(session_df)
aggregate_timestamps = pd.to_datetime(aggregate_bundle["timestamps"])
market_df = market_df[market_df["timestamp"].isin(aggregate_timestamps)].reset_index(drop=True)

if len(market_df) != len(aggregate_bundle["predicted_paths"]):
    common_ts = pd.Index(aggregate_timestamps).intersection(pd.Index(market_df["timestamp"]))
    keep_mask = pd.Index(aggregate_timestamps).isin(common_ts)
    aggregate_bundle = {
        key: value[keep_mask] if isinstance(value, np.ndarray) and len(value) == len(aggregate_timestamps) else value
        for key, value in aggregate_bundle.items()
    }
    market_df = market_df[market_df["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)

env = StockTradingEnv(aggregate_bundle, market_df, RL_CONFIG)
state_dim = env.state_dim

if RL_CONFIG["run_training"]:
    policy, rl_training_payload = train_ppo(env, RL_CONFIG)
else:
    policy = ActorCritic(state_dim).to(RL_CONFIG["device"])
    rl_training_payload = {"stats": [], "note": "RL training skipped by config."}

torch.save({"state_dict": state_dict_to_cpu(policy.state_dict()), "state_dim": state_dim, "config": RL_CONFIG}, RL_DIR / "policy.pt")
save_json(RL_DIR / "env_config.json", {**RL_CONFIG, "state_dim": state_dim})
save_json(
    RL_DIR / "state_schema.json",
    {
        "forecast_path_features": int(HORIZON * 4),
        "market_features": MARKET_STATE_COLUMNS,
        "portfolio_features": ["cash_norm", "shares_norm", "portfolio_value_norm", "position_pct"],
        "regime_feature": ["regime_indicator"],
        "state_dim": state_dim,
    },
)
save_json(RL_DIR / "training_metrics.json", rl_training_payload)

pd.DataFrame(rl_training_payload["stats"]).tail()
'@

$cells += New-CodeCell @'
manifest = {
    "runtime": build_runtime_metadata(),
    "shared": {
        "raw_snapshot": str(raw_snapshot_path),
        "sessionized_snapshot": str(session_snapshot_path),
        "split_spec": str(SHARED_DIR / "split_spec.json"),
        "calendar_meta": str(SHARED_DIR / "calendar_meta.json"),
    },
    "forecast_models": {
        spec.name: {
            "version": spec.version,
            "dir": str(model_dir_for_spec(spec)),
            "lookback": spec.lookback,
            "feature_mode": spec.feature_mode,
            "architecture": spec.architecture,
            "model_path": str(model_dir_for_spec(spec) / "model.pt"),
            "scaler_path": str(model_dir_for_spec(spec) / "scaler.npz"),
            "feature_manifest_path": str(model_dir_for_spec(spec) / "feature_manifest.json"),
            "train_config_path": str(model_dir_for_spec(spec) / "train_config.json"),
            "inference_config_path": str(model_dir_for_spec(spec) / "inference_config.json"),
            "metrics_path": str(model_dir_for_spec(spec) / "metrics.json"),
            "history_path": str(model_dir_for_spec(spec) / "history.csv"),
            "rolling_predictions_path": str(model_dir_for_spec(spec) / "rolling_predictions.npz"),
        }
        for spec in ACTIVE_FORECAST_SPECS
    },
    "ensemble": {
        "weights_json": str(ENSEMBLE_DIR / "weights.json"),
        "weights_csv": str(ENSEMBLE_DIR / "weights.csv"),
        "validation_summary": str(ENSEMBLE_DIR / "validation_summary.csv"),
        "rolling_summary": str(ENSEMBLE_DIR / "rolling_summary.csv"),
    },
    "rl": {
        "policy_path": str(RL_DIR / "policy.pt"),
        "env_config_path": str(RL_DIR / "env_config.json"),
        "state_schema_path": str(RL_DIR / "state_schema.json"),
        "training_metrics_path": str(RL_DIR / "training_metrics.json"),
        "aggregate_predictions_path": str(RL_DIR / "aggregate_predictions.npz"),
    },
}

save_json(ARTIFACT_ROOT / "manifest.json", manifest)

for spec in ACTIVE_FORECAST_SPECS:
    expert_dir = model_dir_for_spec(spec)
    assert (expert_dir / "model.pt").exists(), f"Missing model for {spec.name}"
    assert (expert_dir / "scaler.npz").exists(), f"Missing scaler for {spec.name}"
    assert (expert_dir / "feature_manifest.json").exists(), f"Missing feature manifest for {spec.name}"

weights_payload = json.loads((ENSEMBLE_DIR / "weights.json").read_text(encoding="utf-8"))
assert abs(sum(weights_payload.values()) - 1.0) < 1e-6, "Ensemble weights must sum to 1."
print(f"Manifest written to: {ARTIFACT_ROOT / 'manifest.json'}")
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

$outputPath = Join-Path $PWD "FinalTrain.ipynb"
[System.IO.File]::WriteAllText(
    $outputPath,
    ($notebook | ConvertTo-Json -Depth 100),
    (New-Object System.Text.UTF8Encoding($false))
)
Write-Output "Wrote $outputPath"
