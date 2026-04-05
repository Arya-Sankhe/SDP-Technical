from __future__ import annotations

import json
import textwrap
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "output" / "jupyter-notebook"
BASE_PATH = OUTPUT_DIR / "v8.5rollingbacktest.ipynb"

EXISTING_NOTEBOOKS = {
    "v9.2": OUTPUT_DIR / "v9.2.ipynb",
    "v9.3": OUTPUT_DIR / "v9.3.ipynb",
    "v9.4": OUTPUT_DIR / "v9.4.ipynb",
    "v9.5": OUTPUT_DIR / "v9.5.ipynb",
    "v9.6": OUTPUT_DIR / "v9.6.ipynb",
}


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text())


BASE_NOTEBOOK = load_notebook(BASE_PATH)
SNIPPET_NOTEBOOKS = {
    name: load_notebook(path) for name, path in EXISTING_NOTEBOOKS.items() if path.exists()
}


def lines(source: str) -> list[str]:
    return source.splitlines(keepends=True)


def cell_text(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def set_cell_text(cell: dict, source: str) -> None:
    cell["source"] = lines(textwrap.dedent(source).lstrip("\n"))


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(textwrap.dedent(source).lstrip("\n")),
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(textwrap.dedent(source).lstrip("\n")),
    }


def clear_notebook_outputs(nb: dict) -> None:
    for cell in nb["cells"]:
        cell.setdefault("metadata", {})
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


def find_cell_index(nb: dict, needle: str, cell_type: str | None = None) -> int:
    for idx, cell in enumerate(nb["cells"]):
        if cell_type is not None and cell.get("cell_type") != cell_type:
            continue
        if needle in cell_text(cell):
            return idx
    raise ValueError(f"Could not find cell with needle: {needle!r}")


def replace_cell_source(nb: dict, needle: str, source: str, cell_type: str | None = None) -> None:
    idx = find_cell_index(nb, needle, cell_type=cell_type)
    set_cell_text(nb["cells"][idx], source)


def insert_after(nb: dict, needle: str, new_cells: list[dict], cell_type: str | None = None) -> None:
    idx = find_cell_index(nb, needle, cell_type=cell_type)
    nb["cells"][idx + 1 : idx + 1] = new_cells


def snippet(name: str, idx: int) -> str:
    return cell_text(SNIPPET_NOTEBOOKS[name]["cells"][idx])


def format_py_list(items: list[str], indent: int = 4) -> str:
    pad = " " * indent
    return "[\n" + "".join(f"{pad}{item!r},\n" for item in items) + "]"


BASE_CORE_FEATURES = [
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

TECHNICAL_FEATURES = [
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

REGIME_FEATURES = [
    "atr_14",
    "atr_14_pct",
    "returns",
    "turbulence_60",
    "regime_indicator",
]


COMMON_IMPORTS = """
from __future__ import annotations
import copy
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Patch, Rectangle
from torch.utils.data import DataLoader, Dataset

print("Imports complete")
"""


def data_config_source(extra_var_name: str | None = None, extra_features: list[str] | None = None) -> str:
    if extra_var_name and extra_features:
        extra_block = f"{extra_var_name} = {format_py_list(extra_features)}\nBASE_FEATURE_COLS = BASE_CORE_FEATURES + {extra_var_name}\n"
    else:
        extra_block = "BASE_FEATURE_COLS = BASE_CORE_FEATURES.copy()\n"
    return f"""
# Data Configuration
SYMBOL = 'MSFT'
LOOKBACK_DAYS = 120
OHLC_COLS = ['Open', 'High', 'Low', 'Close']
RAW_COLS = OHLC_COLS + ['Volume', 'TradeCount', 'VWAP']
BASE_CORE_FEATURES = {format_py_list(BASE_CORE_FEATURES)}
{extra_block}TARGET_COLS = ['rOpen', 'rHigh', 'rLow', 'rClose']
INPUT_EXTRA_COL = 'imputedFracWindow'

HORIZON = 50
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
LOOKBACK_CANDIDATES = [64, 96, 160, 256]
DEFAULT_LOOKBACK = 96
ENABLE_LOOKBACK_SWEEP = True
SKIP_OPEN_BARS_TARGET = 6
"""


COMMON_MODEL_CONFIG = """
# Model Configuration
HIDDEN_SIZE = 256  # Increased for better generation capacity
NUM_LAYERS = 2
DROPOUT = 0.20     # Slightly higher for stochasticity
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 256
"""


def summary_source(version: str, phase_label: str) -> str:
    return f"""
# Print Configuration Summary
summary = {{
    'version': {version!r},
    'phase': {phase_label!r},
    'symbol': SYMBOL,
    'lookback_days': LOOKBACK_DAYS,
    'horizon': HORIZON,
    'feature_dim': len(BASE_FEATURE_COLS),
    'ensemble_size': ENSEMBLE_SIZE,
    'sampling_temperature': SAMPLING_TEMPERATURE,
    'device': str(DEVICE),
}}

if 'TECHNICAL_FEATURE_COLS' in globals():
    summary['technical_feature_count'] = len(TECHNICAL_FEATURE_COLS)
if 'REGIME_FEATURE_COLS' in globals():
    summary['regime_feature_count'] = len(REGIME_FEATURE_COLS)
if 'D_MODEL' in globals():
    summary['d_model'] = D_MODEL
if 'N_HEADS' in globals():
    summary['n_heads'] = N_HEADS
if 'N_LAYERS' in globals():
    summary['n_layers'] = N_LAYERS
if 'USE_FREQUENCY' in globals():
    summary['use_frequency'] = USE_FREQUENCY
    summary['multiscale_n_ffts'] = MULTISCALE_N_FFTS
if 'RAG_EMBEDDING_DIM' in globals():
    summary['rag_embedding_dim'] = RAG_EMBEDDING_DIM
    summary['rag_k_retrieve'] = RAG_K_RETRIEVE
    summary['rag_blend_weight'] = RAG_BLEND_WEIGHT
if 'RL_CONFIG' in globals():
    summary['rl_training_steps'] = RL_CONFIG['rl_training_steps']
    summary['run_rl_training'] = RUN_RL_TRAINING

print(summary)
"""


def rolling_config_source(version: str) -> str:
    return f"""
# V8 rolling configuration (frame generator mode)
ROLLINGSTARTTIME = '09:30'
ROLLINGENDTIME = '16:00'
ROLLING_STEP = 1  # 1 = every minute

DEFAULT_ROLLING_TEMPERATURE = 1.5
BASE_ROLLING_TEMPERATURE = DEFAULT_ROLLING_TEMPERATURE
USE_TEMPERATURE_SCHEDULE = True
TEMPERATURESCHEDULE = [
    ('09:30', '10:15', 1.25),
    ('10:15', '14:00', 1.45),
    ('14:00', '16:00', 1.60),
]

ROLLING_BACKTEST_DATE = None  # e.g. '2025-02-13'

FRAME_OUTPUT_DIR = Path('output/jupyter-notebook/frames/{version}')
FRAME_FILENAME_PATTERN = 'frame_{{:04d}}.png'
FRAME_DPI = 180
FRAME_FIGSIZE = (18, 8)
FRAME_HISTORY_BARS = 220

print({{
    'ROLLINGSTARTTIME': ROLLINGSTARTTIME,
    'ROLLINGENDTIME': ROLLINGENDTIME,
    'ROLLING_STEP': ROLLING_STEP,
    'DEFAULT_ROLLING_TEMPERATURE': DEFAULT_ROLLING_TEMPERATURE,
    'USE_TEMPERATURE_SCHEDULE': USE_TEMPERATURE_SCHEDULE,
    'ROLLING_BACKTEST_DATE': ROLLING_BACKTEST_DATE,
    'FRAME_OUTPUT_DIR': str(FRAME_OUTPUT_DIR),
    'FRAME_DPI': FRAME_DPI,
}})
"""


COMMON_FEATURE_BUILD_VALIDATE = """
feat_df = build_feature_frame(price_df)
target_df = build_target_frame(feat_df)

missing_feature_cols = [col for col in BASE_FEATURE_COLS if col not in feat_df.columns]
if missing_feature_cols:
    raise RuntimeError(f'Missing engineered features: {missing_feature_cols}')

print('Feature rows:', len(feat_df))
print('Feature dimension:', len(BASE_FEATURE_COLS))
print('Target columns:', list(target_df.columns))
print('Feature preview:', BASE_FEATURE_COLS[:5], '...', BASE_FEATURE_COLS[-5:])
"""


TECHNICAL_INDICATOR_SRC = """
class TechnicalIndicatorCalculator:
    \"\"\"Compute technical indicators used in Phase 1.\"\"\"

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        required = {'Open', 'High', 'Low', 'Close', 'Volume'}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f'Missing columns for indicators: {sorted(missing)}')

    def add_sma(self, periods: List[int] = [5, 10, 20, 50]) -> 'TechnicalIndicatorCalculator':
        for period in periods:
            self.df[f'sma_{period}'] = self.df['Close'].rolling(period).mean()
        return self

    def add_ema(self, periods: List[int] = [12, 26]) -> 'TechnicalIndicatorCalculator':
        for period in periods:
            self.df[f'ema_{period}'] = self.df['Close'].ewm(span=period, adjust=False).mean()
        return self

    def add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> 'TechnicalIndicatorCalculator':
        ema_fast = self.df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['Close'].ewm(span=slow, adjust=False).mean()
        self.df['macd_line'] = ema_fast - ema_slow
        self.df['macd_signal'] = self.df['macd_line'].ewm(span=signal, adjust=False).mean()
        self.df['macd_histogram'] = self.df['macd_line'] - self.df['macd_signal']
        self.df['macd_momentum'] = self.df['macd_histogram'].diff()
        return self

    def add_rsi(self, period: int = 14) -> 'TechnicalIndicatorCalculator':
        delta = self.df['Close'].diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta.clip(upper=0.0))
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        self.df[f'rsi_{period}'] = 100.0 - (100.0 / (1.0 + rs))
        self.df[f'rsi_{period}_slope'] = self.df[f'rsi_{period}'].diff(5)
        return self

    def add_stochastic(self, k_period: int = 14, d_period: int = 3) -> 'TechnicalIndicatorCalculator':
        low_min = self.df['Low'].rolling(k_period).min()
        high_max = self.df['High'].rolling(k_period).max()
        denom = (high_max - low_min).replace(0.0, np.nan)
        self.df['stoch_k'] = 100.0 * (self.df['Close'] - low_min) / denom
        self.df['stoch_d'] = self.df['stoch_k'].rolling(d_period).mean()
        return self

    def add_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> 'TechnicalIndicatorCalculator':
        sma = self.df['Close'].rolling(period).mean()
        std = self.df['Close'].rolling(period).std()
        self.df['bb_upper'] = sma + std_dev * std
        self.df['bb_lower'] = sma - std_dev * std
        width = (self.df['bb_upper'] - self.df['bb_lower']).replace(0.0, np.nan)
        self.df['bb_width'] = width / sma.replace(0.0, np.nan)
        self.df['bb_position'] = (self.df['Close'] - self.df['bb_lower']) / width
        return self

    def add_atr(self, period: int = 14) -> 'TechnicalIndicatorCalculator':
        high_low = self.df['High'] - self.df['Low']
        high_close = (self.df['High'] - self.df['Close'].shift()).abs()
        low_close = (self.df['Low'] - self.df['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['atr_14'] = true_range.ewm(alpha=1 / period, adjust=False).mean()
        self.df['atr_14_pct'] = self.df['atr_14'] / self.df['Close'].replace(0.0, np.nan)
        return self

    def add_obv(self) -> 'TechnicalIndicatorCalculator':
        delta = np.sign(self.df['Close'].diff().fillna(0.0))
        self.df['obv'] = (delta * self.df['Volume']).cumsum()
        self.df['obv_slope'] = self.df['obv'].diff(5)
        return self

    def add_vwap(self, period: int = 20) -> 'TechnicalIndicatorCalculator':
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3.0
        tpv = typical_price * self.df['Volume']
        rolling_volume = self.df['Volume'].rolling(period).sum().replace(0.0, np.nan)
        self.df['vwap_20'] = tpv.rolling(period).sum() / rolling_volume
        self.df['vwap_20_dev'] = (self.df['Close'] - self.df['vwap_20']) / self.df['vwap_20'].replace(0.0, np.nan)
        return self

    def add_price_momentum(self, periods: List[int] = [5, 10, 20]) -> 'TechnicalIndicatorCalculator':
        for period in periods:
            self.df[f'price_momentum_{period}'] = self.df['Close'] / self.df['Close'].shift(period) - 1.0
        return self

    def add_candle_features(self) -> 'TechnicalIndicatorCalculator':
        spread = (self.df['High'] - self.df['Low']).replace(0.0, np.nan)
        self.df['body_size'] = (self.df['Close'] - self.df['Open']).abs()
        self.df['body_pct'] = self.df['body_size'] / spread
        self.df['upper_shadow'] = self.df['High'] - self.df[['Open', 'Close']].max(axis=1)
        self.df['lower_shadow'] = self.df[['Open', 'Close']].min(axis=1) - self.df['Low']
        self.df['direction'] = np.where(self.df['Close'] > self.df['Open'], 1.0, np.where(self.df['Close'] < self.df['Open'], -1.0, 0.0))
        return self

    def get_all_indicators(self) -> pd.DataFrame:
        return (
            self.add_sma()
            .add_ema()
            .add_macd()
            .add_rsi()
            .add_stochastic()
            .add_bollinger_bands()
            .add_atr()
            .add_obv()
            .add_vwap()
            .add_price_momentum()
            .add_candle_features()
            .df
        )


def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    return TechnicalIndicatorCalculator(df).get_all_indicators()
"""


PHASE1_FEATURE_ENGINEERING_SRC = """
def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9
    g = df.groupby('session_id', sort=False)
    prev_close = g['Close'].shift(1).fillna(df['Open'])
    prev_vol = g['Volume'].shift(1).fillna(df['Volume'])
    prev_tc = g['TradeCount'].shift(1).fillna(df['TradeCount'])
    prev_imp = g['is_imputed'].shift(1).fillna(0).astype(bool)

    row_imputed = (df['is_imputed'].astype(bool) | prev_imp)
    row_open_skip = df['bar_in_session'].astype(int) < SKIP_OPEN_BARS_TARGET

    out = pd.DataFrame(index=df.index, dtype=np.float32)
    out['rOpen'] = np.log(df['Open'] / (prev_close + eps))
    out['rHigh'] = np.log(df['High'] / (prev_close + eps))
    out['rLow'] = np.log(df['Low'] / (prev_close + eps))
    out['rClose'] = np.log(df['Close'] / (prev_close + eps))
    out['logVolChange'] = np.log((df['Volume'] + 1.0) / (prev_vol + 1.0))
    out['logTradeCountChange'] = np.log((df['TradeCount'] + 1.0) / (prev_tc + 1.0))
    out['vwapDelta'] = np.log((df['VWAP'] + eps) / (df['Close'] + eps))
    out['rangeFrac'] = np.maximum(out['rHigh'] - out['rLow'], 0.0) / (np.abs(out['rClose']) + eps)

    signed_body = (df['Close'] - df['Open']) / ((df['High'] - df['Low']) + eps)
    out['orderFlowProxy'] = signed_body * np.log1p(df['Volume'])
    out['tickPressure'] = np.sign(df['Close'] - df['Open']) * np.log1p(df['TradeCount'])

    technical = calculate_technical_features(df[OHLC_COLS + ['Volume']].copy())
    for col in TECHNICAL_FEATURE_COLS:
        out[col] = technical[col].astype(np.float32)

    out['row_imputed'] = row_imputed.astype(np.int8).to_numpy()
    out['row_open_skip'] = row_open_skip.astype(np.int8).to_numpy()
    out['prev_close'] = prev_close.astype(np.float32).to_numpy()

    out = out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return out.astype(np.float32)


def build_target_frame(feat_df: pd.DataFrame) -> pd.DataFrame:
    return feat_df[TARGET_COLS].copy().astype(np.float32)
"""


PHASE2_REGIME_CONFIG_SRC = """
# Phase 2: Regime Detection Configuration
REGIME_CONFIG = RegimeConfig(
    normal_threshold=0.75,
    elevated_threshold=0.90,
    normal_temp_mult=1.0,
    elevated_temp_mult=1.3,
    crisis_temp_mult=1.8,
    normal_position_mult=1.0,
    elevated_position_mult=0.7,
    crisis_position_mult=0.3,
    lookback=60,
)
"""


PHASE2_FEATURE_ENGINEERING_SRC = """
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9
    g = df.groupby('session_id', sort=False)
    prev_close = g['Close'].shift(1).fillna(df['Open'])
    prev_vol = g['Volume'].shift(1).fillna(df['Volume'])
    prev_tc = g['TradeCount'].shift(1).fillna(df['TradeCount'])
    prev_imp = g['is_imputed'].shift(1).fillna(0).astype(bool)

    row_imputed = (df['is_imputed'].astype(bool) | prev_imp)
    row_open_skip = df['bar_in_session'].astype(int) < SKIP_OPEN_BARS_TARGET

    out = pd.DataFrame(index=df.index, dtype=np.float32)
    out['rOpen'] = np.log(df['Open'] / (prev_close + eps))
    out['rHigh'] = np.log(df['High'] / (prev_close + eps))
    out['rLow'] = np.log(df['Low'] / (prev_close + eps))
    out['rClose'] = np.log(df['Close'] / (prev_close + eps))
    out['logVolChange'] = np.log((df['Volume'] + 1.0) / (prev_vol + 1.0))
    out['logTradeCountChange'] = np.log((df['TradeCount'] + 1.0) / (prev_tc + 1.0))
    out['vwapDelta'] = np.log((df['VWAP'] + eps) / (df['Close'] + eps))
    out['rangeFrac'] = np.maximum(out['rHigh'] - out['rLow'], 0.0) / (np.abs(out['rClose']) + eps)

    signed_body = (df['Close'] - df['Open']) / ((df['High'] - df['Low']) + eps)
    out['orderFlowProxy'] = signed_body * np.log1p(df['Volume'])
    out['tickPressure'] = np.sign(df['Close'] - df['Open']) * np.log1p(df['TradeCount'])

    out['atr_14'] = calculate_atr(df, period=14)
    out['atr_14_pct'] = out['atr_14'] / df['Close'].replace(0.0, np.nan)
    out['returns'] = out['rClose']
    out['turbulence_60'] = calculate_historical_turbulence(pd.DataFrame({'returns': out['returns']}, index=df.index), lookback=REGIME_CONFIG.lookback)
    out['regime_indicator'] = 1.0

    out['row_imputed'] = row_imputed.astype(np.int8).to_numpy()
    out['row_open_skip'] = row_open_skip.astype(np.int8).to_numpy()
    out['prev_close'] = prev_close.astype(np.float32).to_numpy()

    out = out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return out.astype(np.float32)


def build_target_frame(feat_df: pd.DataFrame) -> pd.DataFrame:
    return feat_df[TARGET_COLS].copy().astype(np.float32)
"""


PHASE5_RAG_CONFIG_SRC = """
# Phase 5: Retrieval-Augmented Pattern Memory
RAG_EMBEDDING_DIM = 64
RAG_K_RETRIEVE = 5
RAG_BLEND_WEIGHT = 0.25
RAG_MAX_PATTERNS = 4000
RAG_ENCODER_HIDDEN = 128
RAG_ENCODER_LAYERS = 2
"""


PHASE5_RAG_SRC = """
try:
    import faiss  # type: ignore
except Exception:
    faiss = None


@dataclass
class RetrievedPattern:
    sequence: torch.Tensor
    future_path: torch.Tensor
    similarity: float
    index: int


@dataclass
class PatternMatch:
    patterns: List[RetrievedPattern]
    avg_similarity: float = 0.0
    max_similarity: float = 0.0
    min_similarity: float = 0.0


class PatternEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, embedding_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.projection(hidden[-1])


class PatternDatabase:
    def __init__(self, embedding_dim: int = 64, use_faiss: bool = True):
        self.embedding_dim = embedding_dim
        self.use_faiss = bool(use_faiss and faiss is not None)
        self.index = faiss.IndexFlatIP(embedding_dim) if self.use_faiss else None
        self.embeddings: Optional[torch.Tensor] = None
        self.sequences: List[torch.Tensor] = []
        self.future_paths: List[torch.Tensor] = []

    def add_patterns(self, embeddings: torch.Tensor, sequences: torch.Tensor, future_paths: torch.Tensor) -> None:
        embeddings = F.normalize(embeddings.detach().cpu().float(), p=2, dim=1)
        sequences = sequences.detach().cpu().float()
        future_paths = future_paths.detach().cpu().float()
        if self.use_faiss and self.index is not None:
            self.index.add(embeddings.numpy().astype('float32'))
        else:
            self.embeddings = embeddings if self.embeddings is None else torch.cat([self.embeddings, embeddings], dim=0)
        self.sequences.extend(list(sequences))
        self.future_paths.extend(list(future_paths))

    def size(self) -> int:
        if self.use_faiss and self.index is not None:
            return int(self.index.ntotal)
        return 0 if self.embeddings is None else int(self.embeddings.shape[0])

    def search(self, query_embedding: torch.Tensor, k: int = 5) -> List[RetrievedPattern]:
        if self.size() == 0:
            return []
        query = F.normalize(query_embedding.detach().cpu().view(1, -1).float(), p=2, dim=1)
        if self.use_faiss and self.index is not None:
            sims, idxs = self.index.search(query.numpy().astype('float32'), min(k, self.size()))
            scored = zip(sims[0].tolist(), idxs[0].tolist())
        else:
            assert self.embeddings is not None
            sims = torch.matmul(self.embeddings, query.squeeze(0))
            topk = torch.topk(sims, k=min(k, sims.numel()))
            scored = zip(topk.values.tolist(), topk.indices.tolist())
        results = []
        for sim, idx in scored:
            if idx < 0:
                continue
            results.append(
                RetrievedPattern(
                    sequence=self.sequences[idx],
                    future_path=self.future_paths[idx],
                    similarity=float(sim),
                    index=int(idx),
                )
            )
        return results


class RAGPatternRetriever(nn.Module):
    def __init__(self, input_size: int, embedding_dim: int = 64, k_retrieve: int = 5, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.k_retrieve = k_retrieve
        self.encoder = PatternEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
        )
        self.database: Optional[PatternDatabase] = None

    def ready(self) -> bool:
        return self.database is not None and self.database.size() > 0

    def build_database(self, sequences: torch.Tensor, future_paths: torch.Tensor, batch_size: int = 256) -> None:
        if len(sequences) == 0:
            self.database = PatternDatabase(self.embedding_dim, use_faiss=False)
            return
        self.database = PatternDatabase(self.embedding_dim, use_faiss=False)
        device = next(self.encoder.parameters()).device
        embeddings: List[torch.Tensor] = []
        self.encoder.eval()
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                batch = sequences[start : start + batch_size].to(device)
                embeddings.append(self.encoder(batch).cpu())
        self.database.add_patterns(
            embeddings=torch.cat(embeddings, dim=0),
            sequences=sequences.cpu(),
            future_paths=future_paths.cpu(),
        )

    def adjust_path(self, query_sequence: torch.Tensor, base_path: torch.Tensor) -> Tuple[torch.Tensor, PatternMatch]:
        if not self.ready():
            return base_path, PatternMatch(patterns=[])
        device = next(self.encoder.parameters()).device
        self.encoder.eval()
        with torch.no_grad():
            query = query_sequence.unsqueeze(0).to(device).float()
            query_embedding = self.encoder(query).cpu().squeeze(0)
        patterns = self.database.search(query_embedding, k=self.k_retrieve) if self.database is not None else []
        if not patterns:
            return base_path, PatternMatch(patterns=[])
        weights = torch.tensor([p.similarity for p in patterns], dtype=torch.float32, device=base_path.device)
        weights = F.softmax(weights, dim=0)
        retrieved_paths = torch.stack([p.future_path[: base_path.shape[0]].to(base_path.device) for p in patterns], dim=0)
        blended_future = torch.sum(retrieved_paths * weights.view(-1, 1, 1), dim=0)
        adjusted = (1.0 - RAG_BLEND_WEIGHT) * base_path + RAG_BLEND_WEIGHT * blended_future
        match = PatternMatch(
            patterns=patterns,
            avg_similarity=float(weights.mean().item()),
            max_similarity=float(weights.max().item()),
            min_similarity=float(weights.min().item()),
        )
        return adjusted, match
"""


PHASE6_RL_CONFIG_SRC = """
# Phase 6: RL Decision Layer Configuration
RUN_RL_TRAINING = False
RL_CONFIG = {
    'initial_balance': 100000.0,
    'transaction_cost': 0.001,
    'max_position': 1.0,
    'dsr_eta': 0.1,
    'ppo_lr': 3e-4,
    'ppo_gamma': 0.99,
    'ppo_gae_lambda': 0.95,
    'ppo_clip_epsilon': 0.2,
    'ppo_value_coef': 0.5,
    'ppo_entropy_coef': 0.01,
    'ppo_max_grad_norm': 0.5,
    'ppo_rollout_length': 512,
    'ppo_update_epochs': 4,
    'ppo_batch_size': 64,
    'rl_training_steps': 5000,
}
"""


PHASE6_RL_SECTION_MD = """
## Phase 6: RL Decision Layer
This section uses the rolling forecast outputs as the state input to a PPO trading policy.
The price model and strictly causal rolling backtest are preserved from the V8.5 base notebook.
"""


PHASE6_RL_CODE = """
@dataclass
class Trade:
    timestamp: int
    action: str
    shares: float
    price: float
    cost: float


class MarketRegime(Enum):
    NORMAL = 'normal'
    ELEVATED = 'elevated'
    CRISIS = 'crisis'


@dataclass
class RegimeConfig:
    normal_threshold: float = 0.75
    elevated_threshold: float = 0.90
    normal_position_mult: float = 1.0
    elevated_position_mult: float = 0.7
    crisis_position_mult: float = 0.3
    lookback: int = 60


class TurbulenceIndexCalculator:
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.history: List[np.ndarray] = []

    def update(self, returns: np.ndarray) -> float:
        self.history.append(np.asarray(returns, dtype=np.float32))
        if len(self.history) < self.lookback:
            return 0.0
        history = np.asarray(self.history[-self.lookback :], dtype=np.float32)
        mean = history.mean(axis=0)
        cov = np.cov(history.T)
        if np.ndim(cov) < 2:
            cov = np.array([[cov]], dtype=np.float32)
        cov = cov + np.eye(cov.shape[0], dtype=np.float32) * 1e-6
        diff = np.asarray(returns, dtype=np.float32) - mean
        try:
            inv_cov = np.linalg.inv(cov)
            return float(np.sqrt(diff @ inv_cov @ diff))
        except np.linalg.LinAlgError:
            return float(np.linalg.norm(diff))


class MarketRegimeDetector:
    def __init__(self, config: RegimeConfig | None = None):
        self.config = config or RegimeConfig()
        self.turbulence_calc = TurbulenceIndexCalculator(self.config.lookback)
        self.atr_history: List[float] = []
        self.turbulence_history: List[float] = []
        self.current_regime = MarketRegime.NORMAL

    def detect_regime(self, returns: np.ndarray, atr: float) -> MarketRegime:
        turbulence = self.turbulence_calc.update(returns)
        self.atr_history.append(float(atr))
        self.atr_history = self.atr_history[-self.config.lookback :]
        self.turbulence_history.append(float(turbulence))
        self.turbulence_history = self.turbulence_history[-self.config.lookback :]
        if len(self.atr_history) < self.config.lookback:
            self.current_regime = MarketRegime.NORMAL
            return self.current_regime
        atr_percentile = np.mean(np.asarray(self.atr_history) <= atr)
        turb_values = np.asarray([t for t in self.turbulence_history if t > 0.0], dtype=np.float32)
        turb_percentile = np.mean(turb_values <= turbulence) if len(turb_values) else 0.5
        if atr_percentile > self.config.elevated_threshold and turb_percentile > self.config.elevated_threshold:
            self.current_regime = MarketRegime.CRISIS
        elif atr_percentile > self.config.normal_threshold or turb_percentile > self.config.normal_threshold:
            self.current_regime = MarketRegime.ELEVATED
        else:
            self.current_regime = MarketRegime.NORMAL
        return self.current_regime

    def get_position_multiplier(self) -> float:
        return {
            MarketRegime.NORMAL: self.config.normal_position_mult,
            MarketRegime.ELEVATED: self.config.elevated_position_mult,
            MarketRegime.CRISIS: self.config.crisis_position_mult,
        }[self.current_regime]

    def get_regime_indicator(self) -> float:
        return {
            MarketRegime.NORMAL: 0.0,
            MarketRegime.ELEVATED: 0.5,
            MarketRegime.CRISIS: 1.0,
        }[self.current_regime]


def build_rl_market_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out['Close']
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0))
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    out['macd_histogram'] = macd_line - macd_signal

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    bb_width = (bb_upper - bb_lower).replace(0.0, np.nan)
    out['bb_position'] = (close - bb_lower) / bb_width

    high_low = out['High'] - out['Low']
    high_close = (out['High'] - out['Close'].shift()).abs()
    low_close = (out['Low'] - out['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out['atr_14'] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    out['atr_14_pct'] = out['atr_14'] / close.replace(0.0, np.nan)

    out['price_momentum_5'] = close / close.shift(5) - 1.0
    out['price_momentum_20'] = close / close.shift(20) - 1.0
    typical_price = (out['High'] + out['Low'] + out['Close']) / 3.0
    rolling_volume = out['Volume'].rolling(20).sum().replace(0.0, np.nan)
    out['vwap_20'] = (typical_price * out['Volume']).rolling(20).sum() / rolling_volume
    out['vwap_20_dev'] = (out['Close'] - out['vwap_20']) / out['vwap_20'].replace(0.0, np.nan)

    direction = np.sign(out['Close'].diff().fillna(0.0))
    out['direction'] = direction
    out['returns'] = close.pct_change().fillna(0.0)
    out['obv'] = (direction * out['Volume']).cumsum()
    out['obv_slope'] = out['obv'].diff(5)
    return out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)


def pad_prediction_paths(logs: List[RollingPredictionLog], horizon: int) -> np.ndarray:
    padded = []
    for log in logs:
        path = log.predictedpath[OHLC_COLS].to_numpy(np.float32)
        if len(path) < horizon:
            pad_rows = np.repeat(path[-1:], horizon - len(path), axis=0)
            path = np.concatenate([path, pad_rows], axis=0)
        padded.append(path[:horizon])
    return np.stack(padded).astype(np.float32)


class StockTradingEnv:
    def __init__(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        max_position: float = 1.0,
        lookback: int = 60,
        use_turbulence: bool = True,
        dsr_eta: float = 0.1,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.predictions = predictions.astype(np.float32)
        self.initial_balance = float(initial_balance)
        self.transaction_cost = float(transaction_cost)
        self.max_position = float(max_position)
        self.lookback = int(min(lookback, max(1, len(df) - 2)))
        self.use_turbulence = bool(use_turbulence)
        self.dsr_eta = float(dsr_eta)
        self.regime_detector = MarketRegimeDetector() if use_turbulence else None
        self.state_dim = self.predictions.shape[1] * 4 + 10 + 4 + (1 if use_turbulence else 0)
        self.action_dim = 1
        self.reset()

    def reset(self) -> np.ndarray:
        self.balance = self.initial_balance
        self.shares = 0.0
        self.portfolio_value = self.initial_balance
        self.current_step = self.lookback
        self.trades: List[Trade] = []
        self.portfolio_history = [self.initial_balance]
        self.returns_history: List[float] = []
        self.A = 0.0
        self.B = 0.0
        if self.regime_detector is not None:
            self.regime_detector = MarketRegimeDetector(self.regime_detector.config)
        return self._get_observation()

    def _get_market_state(self) -> np.ndarray:
        row = self.df.iloc[self.current_step]
        vol_ma = self.df['Volume'].iloc[max(0, self.current_step - 19) : self.current_step + 1].mean()
        return np.asarray(
            [
                row['rsi_14'] / 100.0,
                row['macd_histogram'],
                row['bb_position'],
                row['atr_14_pct'],
                row['price_momentum_5'],
                row['price_momentum_20'],
                row['vwap_20_dev'],
                row['obv_slope'] / 1e6,
                row['direction'],
                row['Volume'] / vol_ma if vol_ma > 0 else 1.0,
            ],
            dtype=np.float32,
        )

    def _get_observation(self) -> np.ndarray:
        pred = self.predictions[self.current_step].reshape(-1).astype(np.float32)
        price = float(self.df['Close'].iloc[self.current_step])
        portfolio_value = self.balance + self.shares * price
        position_pct = (self.shares * price) / portfolio_value if portfolio_value else 0.0
        portfolio = np.asarray(
            [
                self.balance / self.initial_balance,
                self.shares / 1000.0,
                portfolio_value / self.initial_balance,
                position_pct,
            ],
            dtype=np.float32,
        )
        market = self._get_market_state()
        if self.regime_detector is not None:
            regime = np.asarray([self.regime_detector.get_regime_indicator()], dtype=np.float32)
            return np.concatenate([pred, market, portfolio, regime]).astype(np.float32)
        return np.concatenate([pred, market, portfolio]).astype(np.float32)

    def _calculate_dsr_reward(self, ret: float) -> float:
        if len(self.returns_history) < 2:
            return 0.0
        A_prev = self.A
        B_prev = self.B
        self.A = A_prev + self.dsr_eta * (ret - A_prev)
        self.B = B_prev + self.dsr_eta * (ret * ret - B_prev)
        denominator = max((B_prev - A_prev * A_prev) ** 1.5, 1e-8)
        return float((B_prev * ret - 0.5 * A_prev * ret * ret) / denominator)

    def step(self, action: Union[np.ndarray, float]) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        action_value = float(np.clip(action_value, -1.0, 1.0))
        price = float(self.df['Close'].iloc[self.current_step])
        if self.regime_detector is not None:
            action_value *= self.regime_detector.get_position_multiplier()
        target_position_value = action_value * self.max_position * self.portfolio_value
        current_position_value = self.shares * price
        trade_value = target_position_value - current_position_value
        if abs(trade_value) > 10.0:
            shares_delta = trade_value / price
            trade_cost = abs(trade_value) * self.transaction_cost
            self.balance -= shares_delta * price + trade_cost
            self.shares += shares_delta
            self.trades.append(
                Trade(
                    timestamp=self.current_step,
                    action='buy' if shares_delta > 0 else 'sell',
                    shares=shares_delta,
                    price=price,
                    cost=trade_cost,
                )
            )
        prev_value = self.portfolio_value
        self.portfolio_value = self.balance + self.shares * price
        ret = (self.portfolio_value / prev_value - 1.0) if prev_value else 0.0
        self.returns_history.append(ret)
        reward = self._calculate_dsr_reward(ret)
        self.portfolio_history.append(self.portfolio_value)
        if self.regime_detector is not None:
            self.regime_detector.detect_regime(np.asarray([self.df['returns'].iloc[self.current_step]], dtype=np.float32), float(self.df['atr_14'].iloc[self.current_step]))
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        info = {
            'portfolio_value': float(self.portfolio_value),
            'return': float(ret),
            'shares': float(self.shares),
            'balance': float(self.balance),
        }
        return self._get_observation(), float(reward), done, info

    def get_portfolio_metrics(self) -> Dict[str, float]:
        if len(self.portfolio_history) < 2:
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'num_trades': 0}
        returns = np.asarray(self.returns_history, dtype=np.float32)
        total_return = float(self.portfolio_history[-1] / self.initial_balance - 1.0)
        sharpe = float((returns.mean() / returns.std()) * np.sqrt(252 * 390)) if returns.std() > 0 else 0.0
        portfolio = np.asarray(self.portfolio_history, dtype=np.float32)
        peak = np.maximum.accumulate(portfolio)
        drawdown = (portfolio - peak) / peak
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': float(drawdown.min()),
            'num_trades': float(len(self.trades)),
            'final_value': float(self.portfolio_history[-1]),
        }


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 1, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.shared(state)
        mean = torch.tanh(self.actor_mean(features))
        std = torch.exp(self.actor_log_std).expand_as(mean)
        value = self.critic(features)
        return mean, std, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, value = self.forward(state)
        dist = Normal(mean, std)
        action = mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value


class PPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool], next_value: float) -> Tuple[List[float], List[float]]:
        advantages: List[float] = []
        gae = 0.0
        values = values + [next_value]
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        epochs: int = 4,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        metrics = {'policy_loss': [], 'value_loss': [], 'entropy': [], 'approx_kl': []}
        n = len(states)
        for _ in range(epochs):
            indices = torch.randperm(n, device=states.device)
            for start in range(0, n, batch_size):
                batch_idx = indices[start : start + batch_size]
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                mean, std, values = self.policy(batch_states)
                dist = Normal(mean, std)
                log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (values.squeeze(-1) - batch_returns).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - ratio.log()).mean()
                metrics['policy_loss'].append(float(policy_loss.item()))
                metrics['value_loss'].append(float(value_loss.item()))
                metrics['entropy'].append(float(entropy.item()))
                metrics['approx_kl'].append(float(approx_kl.item()))
        return {key: float(np.mean(values)) for key, values in metrics.items()}


def train_ppo(env: StockTradingEnv, ppo: PPO, total_timesteps: int = 5000, rollout_length: int = 512, update_epochs: int = 4) -> List[float]:
    state = env.reset()
    states_buf: List[np.ndarray] = []
    actions_buf: List[np.ndarray] = []
    rewards_buf: List[float] = []
    values_buf: List[float] = []
    log_probs_buf: List[float] = []
    dones_buf: List[bool] = []
    episode_values: List[float] = []

    for timestep in range(total_timesteps):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=ppo.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value = ppo.policy.get_action(state_tensor)
        next_state, reward, done, info = env.step(action.cpu().numpy()[0])
        states_buf.append(state)
        actions_buf.append(action.cpu().numpy()[0])
        rewards_buf.append(float(reward))
        values_buf.append(float(value.item()))
        log_probs_buf.append(float(log_prob.item()))
        dones_buf.append(bool(done))
        state = next_state
        if done:
            episode_values.append(float(info['portfolio_value']))
            state = env.reset()

        if len(states_buf) >= rollout_length:
            with torch.no_grad():
                next_state_tensor = torch.as_tensor(state, dtype=torch.float32, device=ppo.device).unsqueeze(0)
                _, _, next_value = ppo.policy(next_state_tensor)
                next_value_scalar = float(next_value.item())
            advantages, returns = ppo.compute_gae(rewards_buf, values_buf, dones_buf, next_value_scalar)
            states_tensor = torch.as_tensor(np.asarray(states_buf), dtype=torch.float32, device=ppo.device)
            actions_tensor = torch.as_tensor(np.asarray(actions_buf), dtype=torch.float32, device=ppo.device)
            old_log_probs_tensor = torch.as_tensor(np.asarray(log_probs_buf), dtype=torch.float32, device=ppo.device)
            advantages_tensor = torch.as_tensor(np.asarray(advantages), dtype=torch.float32, device=ppo.device)
            returns_tensor = torch.as_tensor(np.asarray(returns), dtype=torch.float32, device=ppo.device)
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
            metrics = ppo.update(
                states_tensor,
                actions_tensor,
                old_log_probs_tensor,
                advantages_tensor,
                returns_tensor,
                epochs=update_epochs,
                batch_size=RL_CONFIG['ppo_batch_size'],
            )
            print(f'PPO update @ step {timestep}: {metrics}')
            states_buf.clear()
            actions_buf.clear()
            rewards_buf.clear()
            values_buf.clear()
            log_probs_buf.clear()
            dones_buf.clear()
    return episode_values


selected_positions = rolling_backtester.selected_anchor_positions
if selected_positions is None or len(selected_positions) != len(rolling_logs):
    raise RuntimeError('Rolling logs are required before the RL section can run.')

rl_price_df = price_df.iloc[selected_positions][RAW_COLS].copy()
rl_market_df = build_rl_market_frame(rl_price_df)
rl_predictions = pad_prediction_paths(rolling_logs, horizon=HORIZON)

print({
    'rl_rows': len(rl_market_df),
    'prediction_tensor_shape': tuple(rl_predictions.shape),
    'run_rl_training': RUN_RL_TRAINING,
})

rl_env = StockTradingEnv(
    df=rl_market_df,
    predictions=rl_predictions,
    initial_balance=RL_CONFIG['initial_balance'],
    transaction_cost=RL_CONFIG['transaction_cost'],
    max_position=RL_CONFIG['max_position'],
    lookback=min(DEFAULT_LOOKBACK, len(rl_market_df) - 2),
    use_turbulence=True,
    dsr_eta=RL_CONFIG['dsr_eta'],
)

initial_state = rl_env.reset()
print('RL environment state dimension:', initial_state.shape[0])

probe_state, probe_reward, probe_done, probe_info = rl_env.step(np.asarray([0.1], dtype=np.float32))
print({
    'probe_reward': probe_reward,
    'probe_done': probe_done,
    'probe_portfolio_value': probe_info['portfolio_value'],
})

if RUN_RL_TRAINING:
    ppo = PPO(
        state_dim=rl_env.state_dim,
        action_dim=1,
        lr=RL_CONFIG['ppo_lr'],
        gamma=RL_CONFIG['ppo_gamma'],
        gae_lambda=RL_CONFIG['ppo_gae_lambda'],
        clip_epsilon=RL_CONFIG['ppo_clip_epsilon'],
        value_coef=RL_CONFIG['ppo_value_coef'],
        entropy_coef=RL_CONFIG['ppo_entropy_coef'],
        max_grad_norm=RL_CONFIG['ppo_max_grad_norm'],
        device=str(DEVICE),
    )
    episode_values = train_ppo(
        env=rl_env,
        ppo=ppo,
        total_timesteps=RL_CONFIG['rl_training_steps'],
        rollout_length=RL_CONFIG['ppo_rollout_length'],
        update_epochs=RL_CONFIG['ppo_update_epochs'],
    )
    print('RL training finished.')
    print('Episode portfolio values:', episode_values[-5:])
    print('Final portfolio metrics:', rl_env.get_portfolio_metrics())
else:
    print('Set RUN_RL_TRAINING = True to launch PPO training once data access and runtime are available.')
"""


def phase_intro(version: str, title: str, bullets: list[str], footer: str) -> str:
    bullet_lines = "\n".join(f"{i + 1}. {bullet}" for i, bullet in enumerate(bullets))
    return f"# {title}\n\n{bullet_lines}\n\n{footer}\n"


def apply_common_patches(nb: dict, version: str, title_md: str, data_config: str, phase_label: str) -> None:
    clear_notebook_outputs(nb)
    set_cell_text(nb["cells"][0], title_md)
    replace_cell_source(nb, "from __future__ import annotations", COMMON_IMPORTS, cell_type="code")
    replace_cell_source(nb, "# Data Configuration", data_config, cell_type="code")
    replace_cell_source(nb, "# Model Configuration", COMMON_MODEL_CONFIG, cell_type="code")
    replace_cell_source(nb, "# Print Configuration Summary", summary_source(version, phase_label), cell_type="code")
    replace_cell_source(nb, "feat_df = build_feature_frame(price_df)", COMMON_FEATURE_BUILD_VALIDATE, cell_type="code")
    replace_cell_source(nb, "# V8 rolling configuration (frame generator mode)", rolling_config_source(version), cell_type="code")

    rolling_run_src = cell_text(nb["cells"][find_cell_index(nb, "# Run rolling backtest + required validation asserts", cell_type="code")])
    rolling_run_src = rolling_run_src.replace("import sys\n!{sys.executable} -m pip install tqdm\nfrom tqdm import tqdm\n", "")
    if "get_regime_summary" not in rolling_run_src:
        rolling_run_src += "\nif hasattr(rolling_backtester, 'get_regime_summary'):\n    print({'regime_summary': rolling_backtester.get_regime_summary()})\n"
    replace_cell_source(nb, "# Run rolling backtest + required validation asserts", rolling_run_src, cell_type="code")

    validation_src = cell_text(nb["cells"][find_cell_index(nb, "VALIDATION TEST CELL", cell_type="code")])
    validation_src = validation_src.replace(
        'print("VALIDATION TEST CELL - v7.5 Ensemble Trend Injection")',
        'print("VALIDATION TEST CELL - Forecast and Rolling Backtest Metrics")',
    )
    replace_cell_source(nb, "VALIDATION TEST CELL", validation_src, cell_type="code")


def patch_min_predicted_vol(nb: dict) -> None:
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = cell_text(cell)
        src = src.replace(
            "torch.maximum(sigma, torch.tensor(MIN_PREDICTED_VOL))",
            "torch.maximum(sigma, torch.full_like(sigma, MIN_PREDICTED_VOL))",
        )
        set_cell_text(cell, src)


def apply_v94_stability_patches(nb: dict) -> None:
    replace_cell_source(
        nb,
        "# Training Configuration",
        """
# Training Configuration
SWEEP_MAX_EPOCHS = 15
SWEEP_PATIENCE = 5
FINAL_MAX_EPOCHS = 60  # More epochs for convergence
FINAL_PATIENCE = 12
CHECKPOINT_ROOT = Path('output/jupyter-notebook/checkpoints/v9.4_stable')
RESUME_FROM_CHECKPOINT = False
TF_START = 1.0
TF_END = 0.0
TF_DECAY_RATE = 0.95
""",
        cell_type="code",
    )
    replace_cell_source(
        nb,
        "# Inference Configuration - v7.5 Ensemble Settings",
        """
# Inference Configuration - tuned for realistic 1-minute bars
SAMPLING_TEMPERATURE = 0.50
BASE_ENSEMBLE_SIZE = 20
ENSEMBLE_SIZE = 8 if LOW_VRAM_MODE else BASE_ENSEMBLE_SIZE
TREND_LOOKBACK_BARS = 20
STRONG_TREND_THRESHOLD = 0.002
VOLATILITY_SCALING = True
MIN_PREDICTED_VOL = 0.0001
LOG_SIGMA_MIN = math.log(MIN_PREDICTED_VOL)
LOG_SIGMA_MAX = math.log(0.01)
""",
        cell_type="code",
    )
    replace_cell_source(
        nb,
        "# V8 rolling configuration (frame generator mode)",
        """
# V8 rolling configuration (frame generator mode)
ROLLINGSTARTTIME = '09:30'
ROLLINGENDTIME = '16:00'
ROLLING_STEP = 1  # 1 = every minute

DEFAULT_ROLLING_TEMPERATURE = 0.45
BASE_ROLLING_TEMPERATURE = DEFAULT_ROLLING_TEMPERATURE
USE_TEMPERATURE_SCHEDULE = True
TEMPERATURESCHEDULE = [
    ('09:30', '10:15', 0.35),
    ('10:15', '14:00', 0.45),
    ('14:00', '16:00', 0.55),
]

ROLLING_BACKTEST_DATE = None  # e.g. '2025-02-13'

FRAME_OUTPUT_DIR = Path('output/jupyter-notebook/frames/v9.4')
FRAME_FILENAME_PATTERN = 'frame_{:04d}.png'
FRAME_DPI = 180
FRAME_FIGSIZE = (18, 8)
FRAME_HISTORY_BARS = 220

print({
    'ROLLINGSTARTTIME': ROLLINGSTARTTIME,
    'ROLLINGENDTIME': ROLLINGENDTIME,
    'ROLLING_STEP': ROLLING_STEP,
    'DEFAULT_ROLLING_TEMPERATURE': DEFAULT_ROLLING_TEMPERATURE,
    'USE_TEMPERATURE_SCHEDULE': USE_TEMPERATURE_SCHEDULE,
    'ROLLING_BACKTEST_DATE': ROLLING_BACKTEST_DATE,
    'FRAME_OUTPUT_DIR': str(FRAME_OUTPUT_DIR),
    'FRAME_DPI': FRAME_DPI,
})
""",
        cell_type="code",
    )

    model_src = cell_text(nb["cells"][find_cell_index(nb, "class Seq2SeqAttnGRU", cell_type="code")])
    model_src = model_src.replace(
        "        self.hidden_size = hidden_size\n",
        "        self.hidden_size = hidden_size\n        self.return_clip_low: Optional[torch.Tensor] = None\n        self.return_clip_high: Optional[torch.Tensor] = None\n        self.sigma_cap: Optional[torch.Tensor] = None\n",
    )
    model_src = model_src.replace(
        """    def _bound_log_sigma(self, log_sigma):
        return torch.clamp(log_sigma, min=LOG_SIGMA_MIN, max=LOG_SIGMA_MAX)
""",
        """    def _bound_log_sigma(self, log_sigma):
        return torch.clamp(log_sigma, min=LOG_SIGMA_MIN, max=LOG_SIGMA_MAX)

    def set_prediction_constraints(self, clip_low: np.ndarray, clip_high: np.ndarray, sigma_cap: np.ndarray) -> None:
        self.return_clip_low = torch.as_tensor(clip_low, dtype=torch.float32)
        self.return_clip_high = torch.as_tensor(clip_high, dtype=torch.float32)
        self.sigma_cap = torch.as_tensor(sigma_cap, dtype=torch.float32)

    def _clip_returns(self, values: torch.Tensor, widen: float = 1.0) -> torch.Tensor:
        if self.return_clip_low is None or self.return_clip_high is None:
            return values
        low = self.return_clip_low.to(values.device).view(1, -1)
        high = self.return_clip_high.to(values.device).view(1, -1)
        if widen != 1.0:
            center = 0.5 * (low + high)
            half = 0.5 * (high - low) * widen
            low = center - half
            high = center + half
        return torch.maximum(torch.minimum(values, high), low)

    def _cap_sigma(self, sigma: torch.Tensor, historical_vol: Optional[float]) -> torch.Tensor:
        if self.sigma_cap is not None:
            sigma = torch.minimum(sigma, self.sigma_cap.to(sigma.device).view(1, -1))
        if historical_vol is not None:
            hist_cap = max(float(historical_vol) * 3.0, MIN_PREDICTED_VOL)
            sigma = torch.minimum(sigma, torch.full_like(sigma, hist_cap))
        return torch.maximum(sigma, torch.full_like(sigma, MIN_PREDICTED_VOL))
""",
    )
    model_src = model_src.replace(
        """            mu = self.mu_head(out_features)
            log_sigma = self._bound_log_sigma(self.log_sigma_head(out_features))
""",
        """            mu = self._clip_returns(self.mu_head(out_features), widen=1.0)
            log_sigma = self._bound_log_sigma(self.log_sigma_head(out_features))
""",
    )
    model_src = model_src.replace(
        """                else:
                    noise = torch.randn_like(mu) * torch.exp(log_sigma).detach()
                    dec_input = mu + noise
            else:
                dec_input = mu
""",
        """                else:
                    dec_input = mu.detach()
            else:
                dec_input = mu
""",
    )
    model_src = model_src.replace(
        """                mu = self.mu_head(out_features)
                log_sigma = self._bound_log_sigma(self.log_sigma_head(out_features))

                sigma = torch.exp(log_sigma) * temperature

                if historical_vol is not None and t < 5:
                    sigma = torch.ones_like(sigma) * historical_vol

                sigma = torch.maximum(sigma, torch.full_like(sigma, MIN_PREDICTED_VOL))

                noise = torch.randn_like(mu) * sigma
                sample = mu + noise
""",
        """                mu = self._clip_returns(self.mu_head(out_features), widen=1.10)
                log_sigma = self._bound_log_sigma(self.log_sigma_head(out_features))

                sigma = self._cap_sigma(torch.exp(log_sigma) * max(float(temperature), 0.0), historical_vol)
                if float(temperature) <= 0.0:
                    noise = torch.zeros_like(mu)
                else:
                    noise = torch.randn_like(mu) * sigma
                sample = self._clip_returns(mu + noise, widen=1.15)
""",
    )
    replace_cell_source(nb, "class Seq2SeqAttnGRU", model_src, cell_type="code")

    loss_src = cell_text(nb["cells"][find_cell_index(nb, "def clamp_log_sigma(", cell_type="code")])
    loss_src = loss_src.replace(
        """def nll_loss(mu, log_sigma, target):
    \"\"\"Negative log-likelihood for Gaussian\"\"\"
    bounded_log_sigma = clamp_log_sigma(log_sigma)
    sigma = torch.exp(bounded_log_sigma)
    nll = 0.5 * ((target - mu) / sigma) ** 2 + bounded_log_sigma + 0.5 * np.log(2 * np.pi)
    return nll.mean()
""",
        """def nll_loss(mu, log_sigma, target):
    \"\"\"Per-step Gaussian negative log-likelihood.\"\"\"
    bounded_log_sigma = clamp_log_sigma(log_sigma)
    sigma = torch.exp(bounded_log_sigma)
    return 0.5 * ((target - mu) / sigma) ** 2 + bounded_log_sigma + 0.5 * np.log(2 * np.pi)
""",
    )
    loss_src = loss_src.replace(
        """def volatility_match_loss(log_sigma, target):
    \"\"\"Encourage predicted uncertainty to match actual error magnitude\"\"\"
    pred_vol = torch.exp(clamp_log_sigma(log_sigma)).mean()
    actual_vol = target.std()
    return (pred_vol - actual_vol) ** 2
""",
        """def volatility_match_loss(mu, log_sigma, target):
    \"\"\"Calibrate sigma to realized autoregressive forecast error.\"\"\"
    pred_vol = torch.exp(clamp_log_sigma(log_sigma))
    realized_abs_error = (target - mu).detach().abs()
    return ((pred_vol - realized_abs_error) ** 2).mean()
""",
    )
    loss_src = loss_src.replace(
        """def directional_penalty(mu, target):
    pred_close = mu[:, :, 3]
    actual_close = target[:, :, 3]
    sign_match = torch.sign(pred_close) * torch.sign(actual_close)
    penalty = torch.clamp(-sign_match, min=0.0)
    return penalty.mean()
""",
        """def directional_penalty(mu, target):
    pred_close = mu[:, :, 3]
    actual_close = target[:, :, 3]
    mask = actual_close.abs() >= DIRECTION_EPS
    if not mask.any():
        return pred_close.new_tensor(0.0)
    sign_match = torch.sign(pred_close[mask]) * torch.sign(actual_close[mask])
    penalty = torch.clamp(-sign_match, min=0.0)
    return penalty.mean()

def compute_target_constraints(target_windows: np.ndarray) -> dict[str, np.ndarray]:
    flat = target_windows.reshape(-1, target_windows.shape[-1]).astype(np.float32)
    target_std = flat.std(axis=0).astype(np.float32)
    if APPLY_CLIPPING:
        clip_low = np.quantile(flat, CLIP_QUANTILES[0], axis=0).astype(np.float32)
        clip_high = np.quantile(flat, CLIP_QUANTILES[1], axis=0).astype(np.float32)
    else:
        clip_low = np.min(flat, axis=0).astype(np.float32)
        clip_high = np.max(flat, axis=0).astype(np.float32)
    sigma_cap = np.clip(target_std * 3.0, MIN_PREDICTED_VOL, 0.01).astype(np.float32)
    return {
        'clip_low': clip_low,
        'clip_high': clip_high,
        'target_std': target_std,
        'sigma_cap': sigma_cap,
    }

def apply_target_clipping(target_windows: np.ndarray, constraints: dict[str, np.ndarray]) -> np.ndarray:
    low = constraints['clip_low'].reshape(1, 1, -1)
    high = constraints['clip_high'].reshape(1, 1, -1)
    return np.clip(target_windows, low, high).astype(np.float32)
""",
    )
    loss_src = loss_src.replace(
        "            vol = volatility_match_loss(log_sigma, yb_s)\n",
        "            vol = volatility_match_loss(mu, log_sigma, yb_s)\n",
    )
    replace_cell_source(nb, "def clamp_log_sigma(", loss_src, cell_type="code")

    train_src = cell_text(nb["cells"][find_cell_index(nb, "def tf_ratio_for_epoch(", cell_type="code")])
    train_src = train_src.replace(
        "            vol = volatility_match_loss(log_sigma, yb_s)\n",
        "            vol = volatility_match_loss(mu, log_sigma, yb_s)\n",
    )
    replace_cell_source(nb, "def tf_ratio_for_epoch(", train_src, cell_type="code")

    run_fold_src = cell_text(nb["cells"][find_cell_index(nb, "def run_fold(", cell_type="code")])
    run_fold_src = run_fold_src.replace(
        """    X_train, y_train_s, y_train_r = X_all[tr_m], y_all_s[tr_m], y_all_r[tr_m]
    X_val, y_val_s, y_val_r = X_all[va_m], y_all_s[va_m], y_all_r[va_m]
    X_test, y_test_s, y_test_r = X_all[te_m], y_all_s[te_m], y_all_r[te_m]
    test_starts = starts[te_m]
    test_prev_close = prev_close_starts[te_m]
""",
        """    X_train, y_train_s, y_train_r = X_all[tr_m], y_all_s[tr_m], y_all_r[tr_m]
    X_val, y_val_s, y_val_r = X_all[va_m], y_all_s[va_m], y_all_r[va_m]
    X_test, y_test_s, y_test_r = X_all[te_m], y_all_s[te_m], y_all_r[te_m]
    test_starts = starts[te_m]
    test_prev_close = prev_close_starts[te_m]

    target_constraints = compute_target_constraints(y_train_r)
    if APPLY_CLIPPING:
        y_train_s = apply_target_clipping(y_train_s, target_constraints)
        y_val_s = apply_target_clipping(y_val_s, target_constraints)
""",
    )
    run_fold_src = run_fold_src.replace(
        """    model = Seq2SeqAttnGRU(
        input_size=X_train.shape[-1],
        output_size=len(TARGET_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        horizon=HORIZON,
        use_frequency=USE_FREQUENCY,
        freq_out_channels=FREQ_OUT_CHANNELS,
    ).to(DEVICE)
""",
        """    model = Seq2SeqAttnGRU(
        input_size=X_train.shape[-1],
        output_size=len(TARGET_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        horizon=HORIZON,
        use_frequency=USE_FREQUENCY,
        freq_out_channels=FREQ_OUT_CHANNELS,
    ).to(DEVICE)
    model.set_prediction_constraints(
        target_constraints['clip_low'],
        target_constraints['clip_high'],
        target_constraints['sigma_cap'],
    )
""",
    )
    run_fold_src = run_fold_src.replace(
        "        actual_ohlc_1 = price_vals[test_starts + 1]\n        prev_ohlc = price_vals[test_starts]\n",
        "        actual_ohlc_1 = price_vals[test_starts]\n        prev_ohlc = price_vals[test_starts - 1]\n",
    )
    run_fold_src = run_fold_src.replace(
        "        context_df = price_fold.iloc[test_starts[last_idx]-window:test_starts[last_idx]+1][OHLC_COLS]\n",
        "        context_df = price_fold.iloc[test_starts[last_idx]-window:test_starts[last_idx]][OHLC_COLS]\n",
    )
    replace_cell_source(nb, "def run_fold(", run_fold_src, cell_type="code")

    rolling_train_src = cell_text(nb["cells"][find_cell_index(nb, "def train_v7_model_for_rolling(", cell_type="code")])
    rolling_train_src = rolling_train_src.replace(
        """    X_train, y_train_s, y_train_r = X_all[:split], y_all_s[:split], y_all_r[:split]
    X_val, y_val_s, y_val_r = X_all[split:], y_all_s[split:], y_all_r[split:]
""",
        """    X_train, y_train_s, y_train_r = X_all[:split], y_all_s[:split], y_all_r[:split]
    X_val, y_val_s, y_val_r = X_all[split:], y_all_s[split:], y_all_r[split:]

    target_constraints = compute_target_constraints(y_train_r)
    if APPLY_CLIPPING:
        y_train_s = apply_target_clipping(y_train_s, target_constraints)
        y_val_s = apply_target_clipping(y_val_s, target_constraints)
""",
    )
    rolling_train_src = rolling_train_src.replace(
        """    model = Seq2SeqAttnGRU(
        input_size=X_train.shape[-1],
        output_size=len(TARGET_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        horizon=horizon,
        use_frequency=USE_FREQUENCY,
        freq_out_channels=FREQ_OUT_CHANNELS,
    ).to(DEVICE)
""",
        """    model = Seq2SeqAttnGRU(
        input_size=X_train.shape[-1],
        output_size=len(TARGET_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        horizon=horizon,
        use_frequency=USE_FREQUENCY,
        freq_out_channels=FREQ_OUT_CHANNELS,
    ).to(DEVICE)
    model.set_prediction_constraints(
        target_constraints['clip_low'],
        target_constraints['clip_high'],
        target_constraints['sigma_cap'],
    )
""",
    )
    replace_cell_source(nb, "def train_v7_model_for_rolling(", rolling_train_src, cell_type="code")


def build_v91() -> dict:
    nb = deepcopy(BASE_NOTEBOOK)
    title_md = phase_intro(
        "v9.1",
        "Experiment: MSFT 1-Minute GRU Forecast (Phase 1: Technical Indicators v9.1)",
        [
            "Adds a dedicated technical-indicator engine on top of the V8.5 rolling-backtest pipeline.",
            "Preserves the probabilistic GRU forecaster, walk-forward evaluation, and strictly causal rolling backtest.",
            "Extends the feature space with momentum, volatility, volume, and candlestick indicators from `UPDATE.md`.",
        ],
        "Base architecture preserved:\n- Probabilistic outputs (mu + log_sigma) with sampling\n- Autoregressive recursive generation\n- Strict candle-validity enforcement\n- Strictly causal rolling walk-forward backtest",
    )
    apply_common_patches(nb, "v9.1", title_md, data_config_source("TECHNICAL_FEATURE_COLS", TECHNICAL_FEATURES), "Phase 1: Technical Indicators")
    insert_after(
        nb,
        "# Inference Configuration",
        [
            code_cell(
                f"""
# Phase 1: Technical Indicator Configuration
TECHNICAL_FEATURE_COLS = {format_py_list(TECHNICAL_FEATURES)}
"""
            )
        ],
        cell_type="code",
    )
    replace_cell_source(nb, "## Feature Engineering Functions", "## Feature Engineering Functions (Phase 1: Technical Indicators)", cell_type="markdown")
    insert_after(
        nb,
        "## Feature Engineering Functions",
        [
            markdown_cell("### Technical Indicator Engine"),
            code_cell(TECHNICAL_INDICATOR_SRC),
        ],
        cell_type="markdown",
    )
    replace_cell_source(nb, "def build_feature_frame(df: pd.DataFrame)", PHASE1_FEATURE_ENGINEERING_SRC, cell_type="code")
    insert_after(
        nb,
        "Feature preview:",
        [
            markdown_cell("### Phase 1 Validation"),
            code_cell(
                """
phase1_snapshot = feat_df[TECHNICAL_FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
print({
    'indicator_columns': len(TECHNICAL_FEATURE_COLS),
    'remaining_nan_ratio': float(phase1_snapshot.isna().mean().mean()),
    'rsi_range': (float(phase1_snapshot['rsi_14'].min()), float(phase1_snapshot['rsi_14'].max())),
})
"""
            ),
        ],
        cell_type="code",
    )
    patch_min_predicted_vol(nb)
    return nb


def build_v92() -> dict:
    nb = deepcopy(BASE_NOTEBOOK)
    title_md = phase_intro(
        "v9.2",
        "Experiment: MSFT 1-Minute GRU Forecast (Phase 2: Market Regime Detection v9.2)",
        [
            "Adds turbulence-based market-regime detection on top of the V8.5 rolling-backtest pipeline.",
            "Uses regime-aware temperature adjustment during strictly causal rolling inference.",
            "Extends the feature frame with volatility, returns, turbulence, and regime context.",
        ],
        "Base architecture preserved:\n- Probabilistic GRU forecaster\n- Walk-forward training and lookback sweep\n- Strictly causal rolling walk-forward backtest",
    )
    apply_common_patches(nb, "v9.2", title_md, data_config_source("REGIME_FEATURE_COLS", REGIME_FEATURES), "Phase 2: Market Regime Detection")
    insert_after(
        nb,
        "print('GPU:'",
        [
            markdown_cell("## Phase 2: Market Regime Detection System"),
            code_cell(snippet("v9.2", 5)),
        ],
        cell_type="code",
    )
    insert_after(
        nb,
        "# Inference Configuration",
        [code_cell(PHASE2_REGIME_CONFIG_SRC)],
        cell_type="code",
    )
    replace_cell_source(nb, "## Feature Engineering Functions", "## Feature Engineering Functions (Phase 2 Enhanced)", cell_type="markdown")
    replace_cell_source(nb, "def build_feature_frame(df: pd.DataFrame)", PHASE2_FEATURE_ENGINEERING_SRC, cell_type="code")

    rolling_engine = snippet("v9.2", 44)
    rolling_engine += """

def runrollingbacktest(model, pricedf, windowsize, starttime, endtime):
    rb = RollingBacktester(
        model=model,
        pricedf=pricedf,
        featuredf=rolling_feat_df,
        input_mean=rolling_in_mean,
        input_std=rolling_in_std,
        windowsize=windowsize,
        horizon=HORIZON,
        regime_config=REGIME_CONFIG,
    )
    return rb.runrollingbacktest(starttime=starttime, endtime=endtime, date=ROLLING_BACKTEST_DATE, step=ROLLING_STEP), rb
"""
    replace_cell_source(nb, "class RollingPredictionLog", rolling_engine, cell_type="code")

    frame_src = snippet("v9.2", 46)
    frame_src = frame_src.replace("render_single_frame_with_regime", "render_single_frame")
    frame_src = frame_src.replace("generate_rolling_frames_with_regime", "generate_rolling_frames")
    replace_cell_source(nb, "def _draw_candles(", frame_src, cell_type="code")

    insert_after(
        nb,
        "def generate_rolling_frames(",
        [
            markdown_cell("## Phase 2 Validation"),
            code_cell(snippet("v9.2", 48)),
        ],
        cell_type="code",
    )
    patch_min_predicted_vol(nb)
    return nb


def build_v93() -> dict:
    nb = deepcopy(BASE_NOTEBOOK)
    title_md = phase_intro(
        "v9.3",
        "Experiment: MSFT 1-Minute Hybrid iTransformer Forecast (Phase 3 v9.3)",
        [
            "Replaces the pure GRU encoder with a hybrid iTransformer + GRU encoder.",
            "Preserves the V8.5 walk-forward training and strictly causal rolling backtest pipeline.",
            "Adds architecture smoke tests so the hybrid encoder can be validated in-notebook.",
        ],
        "Base architecture preserved:\n- Probabilistic decoder heads\n- Autoregressive recursive generation\n- Strict candle validity enforcement\n- Strictly causal rolling walk-forward backtest",
    )
    apply_common_patches(nb, "v9.3", title_md, data_config_source(), "Phase 3: iTransformer Architecture")
    insert_after(
        nb,
        "# Model Configuration",
        [
            code_cell(
                """
# Phase 3: iTransformer Configuration
D_MODEL = 128
N_HEADS = 8
N_LAYERS = 2
"""
            )
        ],
        cell_type="code",
    )

    model_src = snippet("v9.3", 22) + "\n\n" + snippet("v9.3", 24)
    model_src = model_src.replace("class HybridSeq2Seq", "class Seq2SeqAttnGRU")
    model_src = model_src.replace("HybridSeq2Seq model defined", "Seq2SeqAttnGRU model defined")
    replace_cell_source(nb, "## Model Definition", "## Phase 3: Hybrid iTransformer Model", cell_type="markdown")
    replace_cell_source(nb, "class Seq2SeqAttnGRU", model_src, cell_type="code")

    test_src = snippet("v9.3", 26).replace("HybridSeq2Seq", "Seq2SeqAttnGRU")
    insert_after(
        nb,
        "class Seq2SeqAttnGRU",
        [
            markdown_cell("### Phase 3 Validation"),
            code_cell(test_src),
        ],
        cell_type="code",
    )
    patch_min_predicted_vol(nb)
    return nb


def build_v94() -> dict:
    nb = deepcopy(BASE_NOTEBOOK)
    title_md = phase_intro(
        "v9.4",
        "Experiment: MSFT 1-Minute FTiTransformer Forecast (Phase 4 v9.4)",
        [
            "Adds multi-scale time-frequency feature extraction on top of the V8.5 rolling-backtest pipeline.",
            "Combines the original sequence model with STFT-derived frequency features.",
            "Preserves walk-forward training, rolling evaluation, and frame generation from the base notebook.",
        ],
        "Base architecture preserved:\n- Probabilistic decoder heads\n- Autoregressive recursive generation\n- Strict candle validity enforcement\n- Strictly causal rolling walk-forward backtest",
    )
    apply_common_patches(nb, "v9.4", title_md, data_config_source(), "Phase 4: Time-Frequency Features")
    insert_after(
        nb,
        "# Model Configuration",
        [
            code_cell(
                """
# Phase 4: Frequency Configuration
USE_FREQUENCY = True
FREQ_N_FFT = 16
FREQ_HOP_LENGTH = 4
FREQ_OUT_CHANNELS = 64
MULTISCALE_N_FFTS = [8, 16, 32]
"""
            )
        ],
        cell_type="code",
    )

    model_src = "\n\n".join(
        [
            snippet("v9.4", 27),
            snippet("v9.4", 28),
            snippet("v9.4", 29),
            snippet("v9.4", 36),
        ]
    )
    model_src = model_src.replace("class Seq2SeqAttnFTiTransformer", "class Seq2SeqAttnGRU")
    replace_cell_source(nb, "## Model Definition", "## Phase 4: Time-Frequency Encoder", cell_type="markdown")
    replace_cell_source(nb, "class Seq2SeqAttnGRU", model_src, cell_type="code")

    insert_after(
        nb,
        "class Seq2SeqAttnGRU",
        [
            markdown_cell("### Phase 4 Validation"),
            code_cell(snippet("v9.4", 49)),
        ],
        cell_type="code",
    )
    apply_v94_stability_patches(nb)
    patch_min_predicted_vol(nb)
    return nb


def build_v95() -> dict:
    nb = deepcopy(BASE_NOTEBOOK)
    title_md = phase_intro(
        "v9.5",
        "Experiment: MSFT 1-Minute GRU Forecast (Phase 5: RAG Retrieval Augmentation v9.5)",
        [
            "Adds a retrieval-augmented pattern memory on top of the V8.5 rolling-backtest pipeline.",
            "Builds a historical pattern database from training windows and blends retrieved future paths into inference.",
            "Preserves walk-forward training, rolling evaluation, and frame generation from the base notebook.",
        ],
        "Base architecture preserved:\n- Probabilistic GRU forecaster\n- Autoregressive recursive generation\n- Strict candle validity enforcement\n- Strictly causal rolling walk-forward backtest",
    )
    apply_common_patches(nb, "v9.5", title_md, data_config_source(), "Phase 5: RAG Retrieval System")
    insert_after(
        nb,
        "# Inference Configuration",
        [code_cell(PHASE5_RAG_CONFIG_SRC)],
        cell_type="code",
    )
    insert_after(
        nb,
        "slices = build_walkforward_slices(price_df)",
        [
            markdown_cell("## Phase 5: Retrieval-Augmented Pattern Memory"),
            code_cell(PHASE5_RAG_SRC),
        ],
        cell_type="code",
    )

    model_src = cell_text(nb["cells"][find_cell_index(nb, "class Seq2SeqAttnGRU", cell_type="code")])
    model_src = model_src.replace(
        "        self.hidden_size = hidden_size\n",
        "        self.hidden_size = hidden_size\n        self.rag_retriever: Optional[RAGPatternRetriever] = None\n        self.last_rag_match: Optional[PatternMatch] = None\n",
    )
    model_src = model_src.replace(
        "    def forward(self, x, y_teacher=None, teacher_forcing_ratio=0.0, return_sigma=False):",
        """
    def attach_retriever(self, retriever: RAGPatternRetriever) -> None:
        self.rag_retriever = retriever

    def forward(self, x, y_teacher=None, teacher_forcing_ratio=0.0, return_sigma=False):
""",
    )
    model_src = model_src.replace(
        "            return torch.cat(generated, dim=1)",
        """
            generated_paths = torch.cat(generated, dim=1)
            if self.rag_retriever is None or not self.rag_retriever.ready():
                self.last_rag_match = None
                return generated_paths
            adjusted_paths = []
            last_match = None
            for batch_idx in range(generated_paths.size(0)):
                adjusted_path, last_match = self.rag_retriever.adjust_path(
                    query_sequence=x[batch_idx].detach(),
                    base_path=generated_paths[batch_idx].detach(),
                )
                adjusted_paths.append(adjusted_path.to(generated_paths.device))
            self.last_rag_match = last_match
            return torch.stack(adjusted_paths, dim=0)
""",
    )
    replace_cell_source(nb, "class Seq2SeqAttnGRU", model_src, cell_type="code")

    run_fold_src = cell_text(nb["cells"][find_cell_index(nb, "def run_fold(", cell_type="code")])
    run_fold_src = run_fold_src.replace(
        "    hist = train_model(model, train_loader, val_loader, max_epochs, patience)\n",
        """
    hist = train_model(model, train_loader, val_loader, max_epochs, patience)

    rag_retriever = RAGPatternRetriever(
        input_size=X_train.shape[-1],
        embedding_dim=RAG_EMBEDDING_DIM,
        k_retrieve=RAG_K_RETRIEVE,
        hidden_size=RAG_ENCODER_HIDDEN,
        num_layers=RAG_ENCODER_LAYERS,
    ).to(DEVICE)
    rag_limit = min(RAG_MAX_PATTERNS, len(X_train))
    rag_retriever.build_database(
        sequences=torch.from_numpy(X_train[:rag_limit]).float(),
        future_paths=torch.from_numpy(y_train_r[:rag_limit]).float(),
    )
    model.attach_retriever(rag_retriever)

""",
    )
    run_fold_src = run_fold_src.replace(
        "        'samples': {'train': len(X_train), 'val': len(X_val), 'test': len(X_test)},\n",
        "        'samples': {'train': len(X_train), 'val': len(X_val), 'test': len(X_test)},\n        'rag_database_size': rag_retriever.database.size() if rag_retriever.database is not None else 0,\n",
    )
    replace_cell_source(nb, "def run_fold(", run_fold_src, cell_type="code")

    rolling_train_src = cell_text(nb["cells"][find_cell_index(nb, "def train_v7_model_for_rolling(", cell_type="code")])
    rolling_train_src = rolling_train_src.replace(
        "    history_df = train_model(model, train_loader, val_loader, max_epochs=FINAL_MAX_EPOCHS, patience=FINAL_PATIENCE)\n    return model, feat_all, in_mean.astype(np.float32), in_std.astype(np.float32), history_df\n",
        """
    history_df = train_model(model, train_loader, val_loader, max_epochs=FINAL_MAX_EPOCHS, patience=FINAL_PATIENCE)

    rag_retriever = RAGPatternRetriever(
        input_size=X_all.shape[-1],
        embedding_dim=RAG_EMBEDDING_DIM,
        k_retrieve=RAG_K_RETRIEVE,
        hidden_size=RAG_ENCODER_HIDDEN,
        num_layers=RAG_ENCODER_LAYERS,
    ).to(DEVICE)
    rag_limit = min(RAG_MAX_PATTERNS, len(X_all))
    rag_retriever.build_database(
        sequences=torch.from_numpy(X_all[:rag_limit]).float(),
        future_paths=torch.from_numpy(y_all_r[:rag_limit]).float(),
    )
    model.attach_retriever(rag_retriever)
    print({'rag_database_size': rag_retriever.database.size() if rag_retriever.database is not None else 0})
    return model, feat_all, in_mean.astype(np.float32), in_std.astype(np.float32), history_df
""",
    )
    replace_cell_source(nb, "def train_v7_model_for_rolling(", rolling_train_src, cell_type="code")

    insert_after(
        nb,
        "def generate_rolling_frames(",
        [
            markdown_cell("### Phase 5 Validation"),
            code_cell(
                """
if getattr(rolling_model, 'rag_retriever', None) is not None and rolling_model.rag_retriever.database is not None:
    print({
        'rag_database_size': rolling_model.rag_retriever.database.size(),
        'rag_k_retrieve': RAG_K_RETRIEVE,
        'rag_blend_weight': RAG_BLEND_WEIGHT,
    })
"""
            ),
        ],
        cell_type="code",
    )
    patch_min_predicted_vol(nb)
    return nb


def build_v96() -> dict:
    nb = deepcopy(BASE_NOTEBOOK)
    title_md = phase_intro(
        "v9.6",
        "Experiment: MSFT 1-Minute GRU Forecast (Phase 6: RL Decision Layer v9.6)",
        [
            "Keeps the V8.5 price-forecasting and rolling-backtest pipeline intact.",
            "Adds an RL trading layer that consumes strictly causal rolling predictions as state.",
            "Implements a DSR-based trading environment plus PPO policy optimization in-notebook.",
        ],
        "Base architecture preserved:\n- Probabilistic GRU forecaster\n- Autoregressive recursive generation\n- Strict candle validity enforcement\n- Strictly causal rolling walk-forward backtest",
    )
    apply_common_patches(nb, "v9.6", title_md, data_config_source(), "Phase 6: RL Decision Layer")
    insert_after(
        nb,
        "# Inference Configuration",
        [code_cell(PHASE6_RL_CONFIG_SRC)],
        cell_type="code",
    )
    insert_after(
        nb,
        "def generate_rolling_frames(",
        [
            markdown_cell(PHASE6_RL_SECTION_MD),
            code_cell(PHASE6_RL_CODE),
        ],
        cell_type="code",
    )
    patch_min_predicted_vol(nb)
    return nb


BUILDERS = {
    "v9.1.ipynb": build_v91,
    "v9.2.ipynb": build_v92,
    "v9.3.ipynb": build_v93,
    "v9.4.ipynb": build_v94,
    "v9.5.ipynb": build_v95,
    "v9.6.ipynb": build_v96,
}


def write_notebook(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=1) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, builder in BUILDERS.items():
        notebook = builder()
        clear_notebook_outputs(notebook)
        write_notebook(OUTPUT_DIR / filename, notebook)
        print(f"wrote {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
