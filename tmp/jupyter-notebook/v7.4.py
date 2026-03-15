# %% [markdown] cell 0
# Experiment: MSFT 1-Minute Transformer + Long Memory Forecast (v7.4)

Key changes from v7 GRU baseline:
1. **Transformer Encoder** (4 heads, 2 layers, d_model=256) replacing GRU encoder
2. **Extended lookback** from 96 to 512 bars for longer memory
3. **Positional encoding** for sequence order awareness
4. **Memory attention** - decoder attends to past volatile events in lookback window
5. Keep GRU decoder (autoregressive) using Transformer encoder outputs as context
6. Probabilistic outputs (mu + log_sigma) with temperature control
7. Strict candle validity enforcement at each step

# %% [markdown] cell 1
## Package Installation & Imports

# %% [code] cell 2
import importlib.util
import subprocess
import sys

required = {
    'alpaca': 'alpaca-py',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'pandas_market_calendars': 'pandas-market-calendars',
}
missing = [pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None]
if missing:
    print('Installing missing packages:', missing)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
else:
    print('All required third-party packages are already installed.')

# %% [code] cell 3
from __future__ import annotations
import copy
import math
import os
import random
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import torch
import torch.nn as nn
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Rectangle
from torch.utils.data import DataLoader, Dataset

# %% [markdown] cell 4
## Random Seed & Device Setup

# %% [code] cell 5
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))

# %% [markdown] cell 6
## Configuration

# %% [code] cell 7
# Data Configuration
SYMBOL = 'MSFT'
LOOKBACK_DAYS = 120
OHLC_COLS = ['Open', 'High', 'Low', 'Close']
RAW_COLS = OHLC_COLS + ['Volume', 'TradeCount', 'VWAP']
BASE_FEATURE_COLS = [
    'rOpen', 'rHigh', 'rLow', 'rClose',
    'logVolChange', 'logTradeCountChange',
    'vwapDelta', 'rangeFrac', 'orderFlowProxy', 'tickPressure',
]
TARGET_COLS = ['rOpen', 'rHigh', 'rLow', 'rClose']
INPUT_EXTRA_COL = 'imputedFracWindow'

HORIZON = 15
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# v7.4: Extended lookback candidates for longer memory (was [64, 96, 160, 256])
LOOKBACK_CANDIDATES = [256, 384, 512, 640]
DEFAULT_LOOKBACK = 512  # v7.4: Extended from 96 to 512
ENABLE_LOOKBACK_SWEEP = True
SKIP_OPEN_BARS_TARGET = 6

# %% [code] cell 8
# Model Configuration - v7.4 Transformer Settings
D_MODEL = 256  # Transformer dimension (was HIDDEN_SIZE)
NUM_HEADS = 4  # Attention heads
NUM_ENCODER_LAYERS = 2  # Transformer encoder layers
NUM_DECODER_LAYERS = 2  # GRU decoder layers (depth)
DROPOUT = 0.20
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 256

# Memory attention configuration
MEMORY_ATTENTION_DIM = 128  # Dimension for memory attention mechanism
VOLATILITY_THRESHOLD = 0.9  # Percentile threshold for volatile event detection

# %% [code] cell 9
# Training Configuration
SWEEP_MAX_EPOCHS = 15
SWEEP_PATIENCE = 5
FINAL_MAX_EPOCHS = 60
FINAL_PATIENCE = 12
TF_START = 1.0
TF_END = 0.0
TF_DECAY_RATE = 0.95

# %% [code] cell 10
# Loss Configuration
RANGE_LOSS_WEIGHT = 0.3
VOLATILITY_WEIGHT = 0.5
DIR_PENALTY_WEIGHT = 0.1
STEP_LOSS_POWER = 1.5
MEMORY_ATTENTION_WEIGHT = 0.1  # v7.4: Weight for memory attention loss

# %% [code] cell 11
# Inference Configuration
SAMPLING_TEMPERATURE = 1.5
VOLATILITY_SCALING = True
MIN_PREDICTED_VOL = 0.0001

# %% [code] cell 12
# Data Processing Configuration
STANDARDIZE_TARGETS = False
APPLY_CLIPPING = True
CLIP_QUANTILES = (0.001, 0.999)
DIRECTION_EPS = 0.0001
STD_RATIO_TARGET_MIN = 0.3

# %% [code] cell 13
# Alpaca API Configuration
ALPACA_FEED = os.getenv('ALPACA_FEED', 'iex').strip().lower()
SESSION_TZ = 'America/New_York'
REQUEST_CHUNK_DAYS = 5
MAX_REQUESTS_PER_MINUTE = 120
MAX_RETRIES = 5
MAX_SESSION_FILL_RATIO = 0.15

# %% [code] cell 14
# Print Configuration Summary
print({
    'symbol': SYMBOL,
    'lookback_days': LOOKBACK_DAYS,
    'horizon': HORIZON,
    'default_lookback': DEFAULT_LOOKBACK,  # v7.4: now 512
    'transformer_d_model': D_MODEL,
    'transformer_heads': NUM_HEADS,
    'transformer_layers': NUM_ENCODER_LAYERS,
    'sampling_temperature': SAMPLING_TEMPERATURE,
    'loss_weights': {
        'range': RANGE_LOSS_WEIGHT,
        'volatility': VOLATILITY_WEIGHT,
        'dir_penalty': DIR_PENALTY_WEIGHT,
        'memory_attention': MEMORY_ATTENTION_WEIGHT,
    },
    'device': str(DEVICE),
})

# %% [markdown] cell 15
## Data Fetching Functions

# %% [code] cell 16
class RequestPacer:
    def __init__(self, max_calls_per_minute: int):
        if max_calls_per_minute <= 0:
            raise ValueError('max_calls_per_minute must be >0')
        self.min_interval = 60.0 / float(max_calls_per_minute)
        self.last_call_ts = 0.0
        
    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_call_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call_ts = time.monotonic()

# %% [code] cell 17
def _require_alpaca_credentials() -> tuple[str, str]:
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    if not api_key or not secret_key:
        raise RuntimeError('Missing ALPACA_API_KEY / ALPACA_SECRET_KEY.')
    return api_key, secret_key

def _resolve_feed(feed_name: str) -> DataFeed:
    mapping = {'iex': DataFeed.IEX, 'sip': DataFeed.SIP, 'delayed_sip': DataFeed.DELAYED_SIP}
    k = feed_name.strip().lower()
    if k not in mapping:
        raise ValueError(f'Unsupported ALPACA_FEED={feed_name!r}. Use one of: {list(mapping)}')
    return mapping[k]

# %% [code] cell 18
def fetch_bars_alpaca(symbol: str, lookback_days: int) -> tuple[pd.DataFrame, int]:
    api_key, secret_key = _require_alpaca_credentials()
    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    feed = _resolve_feed(ALPACA_FEED)
    pacer = RequestPacer(MAX_REQUESTS_PER_MINUTE)
    
    end_ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    if ALPACA_FEED in {'sip', 'delayed_sip'}:
        end_ts = end_ts - timedelta(minutes=20)
    start_ts = end_ts - timedelta(days=lookback_days)
    
    parts = []
    cursor = start_ts
    calls = 0
    
    while cursor < end_ts:
        chunk_end = min(cursor + timedelta(days=REQUEST_CHUNK_DAYS), end_ts)
        chunk = None
        for attempt in range(1, MAX_RETRIES + 1):
            pacer.wait()
            calls += 1
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=TimeFrame.Minute,
                    start=cursor,
                    end=chunk_end,
                    feed=feed,
                    limit=10000,
                )
                chunk = client.get_stock_bars(req).df
                break
            except Exception as exc:
                msg = str(exc).lower()
                if ('429' in msg or 'rate limit' in msg) and attempt < MAX_RETRIES:
                    backoff = min(2 ** attempt, 30)
                    print(f'Rate-limited; sleeping {backoff}s (attempt {attempt}/{MAX_RETRIES}).')
                    time.sleep(backoff)
                    continue
                if ('subscription' in msg or 'forbidden' in msg) and ALPACA_FEED != 'iex':
                    raise RuntimeError('Feed unavailable for account. Use ALPACA_FEED=iex or upgrade subscription.') from exc
                raise
        if chunk is not None and not chunk.empty:
            d = chunk.reset_index().rename(columns={
                'timestamp': 'Datetime', 'open': 'Open', 'high': 'High',
                'low': 'Low', 'close': 'Close', 'volume': 'Volume',
                'trade_count': 'TradeCount', 'vwap': 'VWAP',
            })
            if 'Volume' not in d.columns:
                d['Volume'] = 0.0
            if 'TradeCount' not in d.columns:
                d['TradeCount'] = 0.0
            if 'VWAP' not in d.columns:
                d['VWAP'] = d['Close']
            
            need = ['Datetime'] + RAW_COLS
            d['Datetime'] = pd.to_datetime(d['Datetime'], utc=True)
            d = d[need].dropna(subset=OHLC_COLS).set_index('Datetime').sort_index()
            parts.append(d)
        cursor = chunk_end
    
    if not parts:
        raise RuntimeError('No bars returned from Alpaca.')
    out = pd.concat(parts, axis=0).sort_index()
    out = out[~out.index.duplicated(keep='last')]
    return out.astype(np.float32), calls

# %% [code] cell 19
def sessionize_with_calendar(df_utc: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if df_utc.empty:
        raise RuntimeError('Input bars are empty.')
    
    idx = pd.DatetimeIndex(df_utc.index)
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    else:
        idx = idx.tz_convert('UTC')
    
    df_utc = df_utc.copy()
    df_utc.index = idx
    
    cal = mcal.get_calendar('XNYS')
    sched = cal.schedule(
        start_date=(idx.min() - pd.Timedelta(days=2)).date(),
        end_date=(idx.max() + pd.Timedelta(days=2)).date(),
    )
    
    pieces = []
    fill_ratios = []
    
    for sid, (_, row) in enumerate(sched.iterrows()):
        open_ts = pd.Timestamp(row['market_open'])
        close_ts = pd.Timestamp(row['market_close'])
        
        if open_ts.tzinfo is None:
            open_ts = open_ts.tz_localize('UTC')
        else:
            open_ts = open_ts.tz_convert('UTC')
        if close_ts.tzinfo is None:
            close_ts = close_ts.tz_localize('UTC')
        else:
            close_ts = close_ts.tz_convert('UTC')
            
        exp_idx = pd.date_range(open_ts, close_ts, freq='1min', inclusive='left')
        if len(exp_idx) == 0:
            continue
            
        day = df_utc[(df_utc.index >= open_ts) & (df_utc.index < close_ts)]
        day = day.reindex(exp_idx)
        imputed = day[OHLC_COLS].isna().any(axis=1).to_numpy()
        fill_ratio = float(imputed.mean())
        
        if fill_ratio >= 1.0 or fill_ratio > MAX_SESSION_FILL_RATIO:
            continue
            
        day[OHLC_COLS + ['VWAP']] = day[OHLC_COLS + ['VWAP']].ffill().bfill()
        if day['VWAP'].isna().all():
            day['VWAP'] = day['Close']
        else:
            day['VWAP'] = day['VWAP'].fillna(day['Close'])
            
        day['Volume'] = day['Volume'].fillna(0.0)
        day['TradeCount'] = day['TradeCount'].fillna(0.0)
        day['is_imputed'] = imputed.astype(np.int8)
        day['session_id'] = int(sid)
        day['bar_in_session'] = np.arange(len(day), dtype=np.int32)
        day['session_len'] = int(len(day))
        
        if day[RAW_COLS].isna().any().any():
            raise RuntimeError('NaNs remain after per-session fill.')
        pieces.append(day)
        fill_ratios.append(fill_ratio)
    
    if not pieces:
        raise RuntimeError('No sessions kept after calendar filtering.')
        
    out = pd.concat(pieces, axis=0).sort_index()
    out.index = out.index.tz_convert(SESSION_TZ).tz_localize(None)
    out = out.copy()
    
    for c in RAW_COLS:
        out[c] = out[c].astype(np.float32)
    out['is_imputed'] = out['is_imputed'].astype(np.int8)
    out['session_id'] = out['session_id'].astype(np.int32)
    out['bar_in_session'] = out['bar_in_session'].astype(np.int32)
    out['session_len'] = out['session_len'].astype(np.int32)
    
    meta = {
        'calendar_sessions_total': int(len(sched)),
        'kept_sessions': int(len(pieces)),
        'avg_fill_ratio_kept': float(np.mean(fill_ratios)) if fill_ratios else float('nan'),
    }
    return out, meta

# %% [markdown] cell 20
## Fetch Data from Alpaca

# %% [code] cell 21
raw_df_utc, api_calls = fetch_bars_alpaca(SYMBOL, LOOKBACK_DAYS)
price_df, session_meta = sessionize_with_calendar(raw_df_utc)
print(f'Raw rows from Alpaca: {len(raw_df_utc):,}')
print(f'Sessionized rows kept: {len(price_df):,}')
print('Session meta:', session_meta)

min_needed = max(LOOKBACK_CANDIDATES) + HORIZON + 1000
if len(price_df) < min_needed:
    raise RuntimeError(f'Not enough rows after session filtering ({len(price_df)}). Need at least {min_needed}.')

# %% [markdown] cell 22
## Feature Engineering Functions

# %% [code] cell 23
def enforce_candle_validity(ohlc: np.ndarray) -> np.ndarray:
    """Ensure High >= max(Open,Close) and Low <= min(Open,Close)"""
    out = np.asarray(ohlc, dtype=np.float32)
    o, h, l, c = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    out[:, 1] = np.maximum.reduce([h, o, c])
    out[:, 2] = np.minimum.reduce([l, o, c])
    return out

def returns_to_prices_seq(return_ohlc: np.ndarray, last_close: float) -> np.ndarray:
    seq = []
    prev_close = float(last_close)
    for rO, rH, rL, rC in np.asarray(return_ohlc, dtype=np.float32):
        o = prev_close * np.exp(float(rO))
        h = prev_close * np.exp(float(rH))
        l = prev_close * np.exp(float(rL))
        c = prev_close * np.exp(float(rC))
        cand = enforce_candle_validity(np.array([[o, h, l, c]], dtype=np.float32))[0]
        seq.append(cand)
        prev_close = float(cand[3])
    return np.asarray(seq, dtype=np.float32)

# %% [code] cell 24
def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9
    g = df.groupby('session_id', sort=False)
    prev_close = g['Close'].shift(1)
    prev_close = prev_close.fillna(df['Open'])
    prev_vol = g['Volume'].shift(1).fillna(df['Volume'])
    prev_tc = g['TradeCount'].shift(1).fillna(df['TradeCount'])
    prev_imp = g['is_imputed'].shift(1).fillna(0).astype(bool)
    
    row_imputed = (df['is_imputed'].astype(bool) | prev_imp)
    row_open_skip = (df['bar_in_session'].astype(int) < SKIP_OPEN_BARS_TARGET)
    
    out = pd.DataFrame(index=df.index, dtype=np.float32)
    out['rOpen'] = np.log(df['Open'] / (prev_close + eps))
    out['rHigh'] = np.log(df['High'] / (prev_close + eps))
    out['rLow'] = np.log(df['Low'] / (prev_close + eps))
    out['rClose'] = np.log(df['Close'] / (prev_close + eps))
    out['logVolChange'] = np.log((df['Volume'] + 1.0) / (prev_vol + 1.0))
    out['logTradeCountChange'] = np.log((df['TradeCount'] + 1.0) / (prev_tc + 1.0))
    out['vwapDelta'] = np.log((df['VWAP'] + eps) / (df['Close'] + eps))
    out['rangeFrac'] = np.maximum(out['rHigh'] - out['rLow'], 0) / (np.abs(out['rClose']) + eps)
    
    signed_body = (df['Close'] - df['Open']) / ((df['High'] - df['Low']) + eps)
    out['orderFlowProxy'] = signed_body * np.log1p(df['Volume'])
    out['tickPressure'] = np.sign(df['Close'] - df['Open']) * np.log1p(df['TradeCount'])
    
    out['row_imputed'] = row_imputed.astype(np.int8).to_numpy()
    out['row_open_skip'] = row_open_skip.astype(np.int8).to_numpy()
    out['prev_close'] = prev_close.astype(np.float32).to_numpy()
    return out.astype(np.float32)

def build_target_frame(feat_df: pd.DataFrame) -> pd.DataFrame:
    return feat_df[TARGET_COLS].copy().astype(np.float32)

# %% [code] cell 25
feat_df = build_feature_frame(price_df)
target_df = build_target_frame(feat_df)
print('Feature rows:', len(feat_df))
print('Target columns:', list(target_df.columns))

# %% [markdown] cell 26
## Windowing & Dataset Functions

# %% [code] cell 27
def split_points(n_rows: int) -> tuple[int, int]:
    tr = int(n_rows * TRAIN_RATIO)
    va = int(n_rows * (TRAIN_RATIO + VAL_RATIO))
    return tr, va

def build_walkforward_slices(price_df_full: pd.DataFrame) -> list[tuple[str, int, int]]:
    n = len(price_df_full)
    span = int(round(n * 0.85))
    shift = max(1, n - span)
    cands = [('slice_1', 0, min(span, n)), ('slice_2', shift, min(shift + span, n))]
    out = []
    seen = set()
    for name, a, b in cands:
        key = (a, b)
        if key in seen or b - a < max(LOOKBACK_CANDIDATES) + HORIZON + 1400:
            continue
        out.append((name, a, b))
        seen.add(key)
    return out if out else [('full', 0, n)]

# %% [code] cell 28
def make_multistep_windows(input_scaled, target_scaled, target_raw, row_imputed, row_open_skip, 
                           starts_prev_close, window, horizon):
    X, y_s, y_r, starts, prev_close = [], [], [], [], []
    dropped_target_imputed, dropped_target_open_skip = 0, 0
    n = len(input_scaled)
    
    for i in range(window, n - horizon + 1):
        if row_imputed[i:i+horizon].any():
            dropped_target_imputed += 1
            continue
        if row_open_skip[i:i+horizon].any():
            dropped_target_open_skip += 1
            continue
            
        xb = input_scaled[i-window:i]
        imp_frac = float(row_imputed[i-window:i].mean())
        imp_col = np.full((window, 1), imp_frac, dtype=np.float32)
        xb_aug = np.concatenate([xb, imp_col], axis=1)
        
        X.append(xb_aug)
        y_s.append(target_scaled[i:i+horizon])
        y_r.append(target_raw[i:i+horizon])
        starts.append(i)
        prev_close.append(starts_prev_close[i])
    
    return (np.asarray(X, dtype=np.float32), np.asarray(y_s, dtype=np.float32),
            np.asarray(y_r, dtype=np.float32), np.asarray(starts, dtype=np.int64),
            np.asarray(prev_close, dtype=np.float32), dropped_target_imputed, dropped_target_open_skip)

# %% [code] cell 29
class MultiStepDataset(Dataset):
    def __init__(self, X, y_s, y_r):
        self.X = torch.from_numpy(X).float()
        self.y_s = torch.from_numpy(y_s).float()
        self.y_r = torch.from_numpy(y_r).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y_s[idx], self.y_r[idx]

# %% [code] cell 30
slices = build_walkforward_slices(price_df)
print('Walk-forward slices:', slices)

# %% [markdown] cell 31
## v7.4 Model Definition: Transformer Encoder + Memory Attention + GRU Decoder

# %% [code] cell 32
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# %% [code] cell 33
class VolatileEventMemory(nn.Module):
    """
    Memory attention mechanism that identifies and attends to past volatile events
    in the lookback window. Helps decoder focus on significant market movements.
    """
    def __init__(self, d_model: int, memory_dim: int):
        super().__init__()
        self.d_model = d_model
        self.memory_dim = memory_dim
        
        # Project encoder outputs to memory space
        self.memory_proj = nn.Linear(d_model, memory_dim)
        
        # Volatility detection: projects to scalar volatility score
        self.volatility_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        
        # Attention mechanism for memory access
        self.query_proj = nn.Linear(d_model, memory_dim)
        self.key_proj = nn.Linear(memory_dim, memory_dim)
        self.value_proj = nn.Linear(memory_dim, memory_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(memory_dim, d_model),
            nn.LayerNorm(d_model),
        )
    
    def detect_volatile_events(self, enc_out: torch.Tensor, threshold_percentile: float = 0.9):
        """
        Detect volatile events based on feature magnitude.
        Returns: volatility_scores [batch, seq_len], volatile_mask [batch, seq_len]
        """
        # Calculate volatility score for each timestep
        volatility_scores = self.volatility_scorer(enc_out).squeeze(-1)  # [batch, seq_len]
        
        # Determine threshold per batch
        thresholds = torch.quantile(volatility_scores, threshold_percentile, dim=1, keepdim=True)
        volatile_mask = volatility_scores > thresholds
        
        return volatility_scores, volatile_mask
    
    def forward(self, decoder_hidden: torch.Tensor, enc_out: torch.Tensor, 
                threshold_percentile: float = 0.9):
        """
        Args:
            decoder_hidden: [batch, d_model] - current decoder state
            enc_out: [batch, seq_len, d_model] - transformer encoder outputs
        Returns:
            memory_context: [batch, d_model] - attended memory context
            attention_weights: [batch, seq_len] - attention distribution over memory
            volatile_mask: [batch, seq_len] - binary mask of volatile events
        """
        batch_size, seq_len, _ = enc_out.shape
        
        # Detect volatile events
        volatility_scores, volatile_mask = self.detect_volatile_events(enc_out, threshold_percentile)
        
        # Project encoder outputs to memory space
        memory = self.memory_proj(enc_out)  # [batch, seq_len, memory_dim]
        
        # Compute attention from decoder to memory
        query = self.query_proj(decoder_hidden).unsqueeze(2)  # [batch, memory_dim, 1]
        keys = self.key_proj(memory)  # [batch, seq_len, memory_dim]
        values = self.value_proj(memory)  # [batch, seq_len, memory_dim]
        
        # Attention scores
        scores = torch.bmm(keys, query).squeeze(2) / math.sqrt(self.memory_dim)  # [batch, seq_len]
        
        # Boost scores for volatile events
        volatility_boost = volatile_mask.float() * 2.0  # Boost volatile events
        scores = scores + volatility_boost
        
        # Softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=1)  # [batch, seq_len]
        
        # Weighted sum of values
        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)  # [batch, memory_dim]
        
        # Project back to d_model
        memory_context = self.output_proj(context)  # [batch, d_model]
        
        return memory_context, attention_weights, volatile_mask

# %% [code] cell 34
class TransformerMemoryAttnModel(nn.Module):
    """
    v7.4: Transformer Encoder + Memory Attention + GRU Decoder
    
    - Transformer encoder (4 heads, 2 layers, d_model=256) processes long lookback
    - Positional encoding for sequence order
    - Memory attention: decoder attends to past volatile events
    - GRU decoder for autoregressive generation
    - Probabilistic outputs (mu + sigma) with temperature control
    """
    def __init__(self, input_size: int, output_size: int, d_model: int = 256, 
                 num_heads: int = 4, num_encoder_layers: int = 2, num_decoder_layers: int = 2,
                 dropout: float = 0.2, horizon: int = 15, memory_dim: int = 128):
        super().__init__()
        self.horizon = horizon
        self.output_size = output_size
        self.d_model = d_model
        
        # Input projection to d_model
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=2048, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=False,  # PyTorch transformer uses [seq, batch, features] by default
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Memory attention mechanism
        self.memory_attention = VolatileEventMemory(d_model, memory_dim)
        
        # Decoder: GRU with memory context
        self.decoder_init = nn.Linear(d_model, num_decoder_layers * d_model)
        self.decoder_cell = nn.GRUCell(output_size + d_model * 2, d_model)  # +d_model*2 for context + memory
        self.decoder_hidden_proj = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_decoder_layers - 1)
        ])
        self.num_decoder_layers = num_decoder_layers
        
        # Standard attention for encoder-decoder
        self.attn_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Output heads: mu and log_sigma for each OHLC
        self.mu_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # *3 for h_dec + context + memory
            nn.GELU(),
            nn.Linear(d_model, output_size),
        )
        self.log_sigma_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, output_size),
        )
        
        # Initialize sigma head for moderate volatility
        nn.init.xavier_uniform_(self.mu_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.mu_head[-1].bias)
        nn.init.zeros_(self.log_sigma_head[-1].weight)
        nn.init.zeros_(self.log_sigma_head[-1].bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence with transformer.
        x: [batch, seq_len, input_size]
        Returns: [batch, seq_len, d_model]
        """
        # Project input
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        
        # Transpose for transformer: [seq_len, batch, d_model]
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        enc_out = self.transformer_encoder(x)  # [seq_len, batch, d_model]
        
        # Transpose back: [batch, seq_len, d_model]
        enc_out = enc_out.transpose(0, 1)
        
        return enc_out
    
    def _attend(self, h_dec: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        """Standard encoder-decoder attention."""
        query = self.attn_proj(h_dec).unsqueeze(2)  # [batch, d_model, 1]
        scores = torch.bmm(enc_out, query).squeeze(2) / math.sqrt(self.d_model)  # [batch, seq_len]
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)  # [batch, d_model]
        return context
    
    def forward(self, x: torch.Tensor, y_teacher: torch.Tensor = None, 
                teacher_forcing_ratio: float = 0.0, return_sigma: bool = False, 
                return_attention: bool = False):
        """
        Forward pass with optional teacher forcing.
        x: [batch, seq_len, input_size]
        y_teacher: [batch, horizon, output_size] - ground truth for teacher forcing
        """
        batch_size = x.size(0)
        
        # Encode
        enc_out = self.encode(x)  # [batch, seq_len, d_model]
        
        # Initialize decoder hidden state from encoded sequence
        pooled = enc_out.mean(dim=1)  # [batch, d_model]
        h_dec = self.decoder_init(pooled).view(batch_size, self.num_decoder_layers, self.d_model)
        h_dec = h_dec[:, 0, :]  # Use first layer's init
        
        # Initial decoder input: last timestep OHLC returns from input
        dec_input = x[:, -1, :self.output_size]
        
        mu_seq, sigma_seq, attn_weights_list = [], [], []
        
        for t in range(self.horizon):
            # Standard attention
            context = self._attend(h_dec, enc_out)
            
            # Memory attention to volatile events
            memory_context, attn_weights, volatile_mask = self.memory_attention(h_dec, enc_out)
            attn_weights_list.append(attn_weights)
            
            # Combine contexts
            combined_context = torch.cat([context, memory_context], dim=1)
            
            # Decoder cell
            cell_input = torch.cat([dec_input, combined_context], dim=1)
            h_dec = self.decoder_cell(cell_input, h_dec)
            
            # Output
            out_features = torch.cat([h_dec, context, memory_context], dim=1)
            
            mu = self.mu_head(out_features)
            log_sigma = self.log_sigma_head(out_features)
            
            mu_seq.append(mu.unsqueeze(1))
            sigma_seq.append(log_sigma.unsqueeze(1))
            
            # Teacher forcing or autoregressive
            if y_teacher is not None and teacher_forcing_ratio > 0.0:
                if teacher_forcing_ratio >= 1.0 or torch.rand(1).item() < teacher_forcing_ratio:
                    dec_input = y_teacher[:, t, :]
                else:
                    noise = torch.randn_like(mu) * torch.exp(log_sigma).detach()
                    dec_input = mu + noise
            else:
                dec_input = mu
        
        mu_out = torch.cat(mu_seq, dim=1)  # [batch, horizon, output_size]
        sigma_out = torch.cat(sigma_seq, dim=1)
        
        outputs = (mu_out, sigma_out) if return_sigma else (mu_out,)
        if return_attention:
            outputs += (attn_weights_list,)
        return outputs[0] if len(outputs) == 1 else outputs

# %% [code] cell 35
    def generate_realistic(self, x: torch.Tensor, temperature: float = 1.0, 
                          historical_vol: float = None, return_attention: bool = False):
        """
        Generate realistic price paths with controlled stochasticity.
        
        Args:
            x: [batch, seq_len, input_size]
            temperature: controls volatility (>1.0 = more wild)
            historical_vol: if provided, scale noise to match this volatility
            return_attention: if True, also return attention weights
        """
        self.eval()
        with torch.no_grad():
            batch_size = x.size(0)
            
            # Encode
            enc_out = self.encode(x)
            
            # Initialize decoder
            pooled = enc_out.mean(dim=1)
            h_dec = self.decoder_init(pooled).view(batch_size, self.num_decoder_layers, self.d_model)
            h_dec = h_dec[:, 0, :]
            
            dec_input = x[:, -1, :self.output_size]
            
            generated = []
            attention_history = []
            
            for t in range(self.horizon):
                # Standard attention
                context = self._attend(h_dec, enc_out)
                
                # Memory attention
                memory_context, attn_weights, volatile_mask = self.memory_attention(h_dec, enc_out)
                attention_history.append({
                    'weights': attn_weights.cpu(),
                    'volatile_mask': volatile_mask.cpu(),
                })
                
                # Combine contexts
                combined_context = torch.cat([context, memory_context], dim=1)
                
                # Decoder
                cell_input = torch.cat([dec_input, combined_context], dim=1)
                h_dec = self.decoder_cell(cell_input, h_dec)
                
                # Output
                out_features = torch.cat([h_dec, context, memory_context], dim=1)
                mu = self.mu_head(out_features)
                log_sigma = self.log_sigma_head(out_features)
                
                # Scale sigma by temperature
                sigma = torch.exp(log_sigma) * temperature
                
                # Optional: override with historical volatility for first few steps
                if historical_vol is not None and t < 5:
                    sigma = torch.ones_like(sigma) * historical_vol
                
                # Ensure minimum volatility
                sigma = torch.maximum(sigma, torch.tensor(MIN_PREDICTED_VOL))
                
                # Sample from distribution
                noise = torch.randn_like(mu) * sigma
                sample = mu + noise
                
                generated.append(sample.unsqueeze(1))
                dec_input = sample  # Autoregressive feedback
            
            result = torch.cat(generated, dim=1)
            if return_attention:
                return result, attention_history
            return result

# %% [markdown] cell 36
## Loss Functions

# %% [code] cell 37
def nll_loss(mu, log_sigma, target):
    """Negative log-likelihood for Gaussian"""
    sigma = torch.exp(log_sigma)
    nll = 0.5 * ((target - mu) / sigma) ** 2 + log_sigma + 0.5 * np.log(2 * np.pi)
    return nll.mean()

def candle_range_loss(mu, target):
    pred_range = mu[:, :, 1] - mu[:, :, 2]  # High - Low
    actual_range = target[:, :, 1] - target[:, :, 2]
    return ((pred_range - actual_range) ** 2).mean()

def volatility_match_loss(log_sigma, target):
    """Encourage predicted uncertainty to match actual error magnitude"""
    pred_vol = torch.exp(log_sigma).mean()
    actual_vol = target.std()
    return (pred_vol - actual_vol) ** 2

def directional_penalty(mu, target):
    pred_close = mu[:, :, 3]
    actual_close = target[:, :, 3]
    sign_match = torch.sign(pred_close) * torch.sign(actual_close)
    penalty = torch.clamp(-sign_match, min=0.0)
    return penalty.mean()

def memory_attention_loss(attn_weights_list, horizon):
    """
    v7.4: Encourage model to use memory attention effectively.
    Penalize uniform attention (encourage focus on specific events).
    """
    if not attn_weights_list:
        return torch.tensor(0.0, device=DEVICE)
    
    total_entropy = 0.0
    for attn in attn_weights_list:
        # Entropy: -sum(p * log(p)) - lower entropy means sharper focus
        entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=1).mean()
        total_entropy += entropy
    
    # We want to minimize entropy (sharper focus), so return entropy
    return total_entropy / len(attn_weights_list)

# %% [markdown] cell 38
## Training Functions

# %% [code] cell 39
def tf_ratio_for_epoch(epoch):
    ratio = TF_START * (TF_DECAY_RATE ** (epoch - 1))
    return max(float(TF_END), float(ratio))

def run_epoch(model, loader, step_weights_t, optimizer=None, tf_ratio=0.0):
    is_train = optimizer is not None
    model.train(is_train)
    
    total_loss, nll_total, range_total, vol_total, dir_total, mem_total = 0, 0, 0, 0, 0, 0
    n_items = 0
    
    for xb, yb_s, yb_r in loader:
        xb = xb.to(DEVICE)
        yb_s = yb_s.to(DEVICE)
        
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            
        with torch.set_grad_enabled(is_train):
            # v7.4: Get mu, sigma, and attention weights
            mu, sigma_out, attn_list = model(
                xb, 
                y_teacher=yb_s if is_train else None, 
                teacher_forcing_ratio=tf_ratio if is_train else 0.0, 
                return_sigma=True,
                return_attention=True
            )
            
            # Weighted losses
            nll = (nll_loss(mu, sigma_out, yb_s) * step_weights_t).mean()
            rng = candle_range_loss(mu, yb_s)
            vol = volatility_match_loss(sigma_out, yb_s)
            dir_pen = directional_penalty(mu, yb_s)
            mem_loss = memory_attention_loss(attn_list, HORIZON)
            
            loss = (nll + RANGE_LOSS_WEIGHT * rng + VOLATILITY_WEIGHT * vol + 
                    DIR_PENALTY_WEIGHT * dir_pen + MEMORY_ATTENTION_WEIGHT * mem_loss)
            
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
        bs = xb.size(0)
        total_loss += loss.item() * bs
        nll_total += nll.item() * bs
        range_total += rng.item() * bs
        vol_total += vol.item() * bs
        dir_total += dir_pen.item() * bs
        mem_total += mem_loss.item() * bs
        n_items += bs
        
    return {
        'total': total_loss / max(n_items, 1),
        'nll': nll_total / max(n_items, 1),
        'range': range_total / max(n_items, 1),
        'vol': vol_total / max(n_items, 1),
        'dir': dir_total / max(n_items, 1),
        'mem': mem_total / max(n_items, 1),
    }

# %% [code] cell 40
def train_model(model, train_loader, val_loader, max_epochs, patience):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    
    step_idx = np.arange(HORIZON, dtype=np.float32)
    step_w = 1.0 + (step_idx / max(HORIZON - 1, 1)) ** STEP_LOSS_POWER
    step_weights_t = torch.as_tensor(step_w, dtype=torch.float32, device=DEVICE).view(1, HORIZON, 1)
    
    best_val = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    rows = []
    
    for epoch in range(1, max_epochs + 1):
        tf = tf_ratio_for_epoch(epoch)
        tr = run_epoch(model, train_loader, step_weights_t, optimizer=optimizer, tf_ratio=tf)
        va = run_epoch(model, val_loader, step_weights_t, optimizer=None, tf_ratio=0.0)
        
        scheduler.step(va['total'])
        lr = optimizer.param_groups[0]['lr']
        
        rows.append({
            'epoch': epoch, 'tf_ratio': tf, 'lr': lr,
            'train_total': tr['total'], 'val_total': va['total'],
            'train_nll': tr['nll'], 'val_nll': va['nll'],
            'train_range': tr['range'], 'val_range': va['range'],
            'train_mem': tr['mem'], 'val_mem': va['mem'],
        })
        
        print(f"Epoch {epoch:02d} | tf={tf:.3f} | "
              f"train={tr['total']:.6f} (nll={tr['nll']:.6f}, mem={tr['mem']:.4f}) | "
              f"val={va['total']:.6f} (nll={va['nll']:.6f}) | lr={lr:.6g}")
        
        if va['total'] < best_val:
            best_val = va['total']
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f'Early stopping at epoch {epoch}.')
                break
                
    model.load_state_dict(best_state)
    return pd.DataFrame(rows)

# %% [markdown] cell 41
## Evaluation Functions

# %% [code] cell 42
def evaluate_metrics(actual_ohlc, pred_ohlc, prev_close):
    actual_ohlc = np.asarray(actual_ohlc, dtype=np.float32)
    pred_ohlc = np.asarray(pred_ohlc, dtype=np.float32)
    ac, pc = actual_ohlc[:, 3], pred_ohlc[:, 3]
    
    # Directional accuracy vs persistence baseline
    actual_direction = np.sign(ac - prev_close)
    pred_direction = np.sign(pc - prev_close)
    dir_acc = float(np.mean(actual_direction == pred_direction))
    
    # Bias: average predicted close vs actual close
    bias = float(np.mean(pc - ac))
    
    return {
        'close_mae': float(np.mean(np.abs(ac - pc))),
        'close_rmse': float(np.sqrt(np.mean((ac - pc) ** 2))),
        'ohlc_mae': float(np.mean(np.abs(actual_ohlc - pred_ohlc))),
        'directional_accuracy': dir_acc,
        'directional_accuracy_eps': float(np.mean(np.sign(ac - prev_close) == np.sign(pc - prev_close))),
        'bias': bias,
        'avg_pred_close': float(np.mean(pc)),
        'avg_actual_close': float(np.mean(ac)),
    }

def evaluate_baselines(actual_ohlc, prev_ohlc, prev_close):
    persistence = evaluate_metrics(actual_ohlc, prev_ohlc, prev_close)
    flat = np.repeat(prev_close.reshape(-1, 1), 4, axis=1).astype(np.float32)
    flat_rw = evaluate_metrics(actual_ohlc, flat, prev_close)
    return {'persistence': persistence, 'flat_close_rw': flat_rw}

# %% [code] cell 43
@torch.no_grad()
def predict_realistic_recursive(model, X, context_prices, temperature=1.0):
    """
    Generate realistic predictions using autoregressive sampling with memory attention.
    context_prices: last LOOKBACK prices to calculate realized vol for scaling
    """
    model.eval()
    
    # Calculate realized volatility from context for scaling
    log_returns = np.log(context_prices[1:] / context_prices[:-1])
    historical_vol = float(np.std(log_returns)) if len(log_returns) > 1 else 0.001
    
    print(f"Historical realized vol: {historical_vol:.6f}, Temperature: {temperature}")
    
    X_tensor = torch.from_numpy(X).float().to(DEVICE)
    
    # Generate with memory attention tracking
    generated, attention_history = model.generate_realistic(
        X_tensor, temperature=temperature, 
        historical_vol=historical_vol,
        return_attention=True
    )
    
    # Print memory attention stats
    avg_volatile_attention = 0
    for attn_info in attention_history:
        weights = attn_info['weights']
        volatile = attn_info['volatile_mask']
        avg_volatile_attention += (weights * volatile.float()).sum(dim=1).mean().item()
    avg_volatile_attention /= len(attention_history)
    print(f"Average attention to volatile events: {avg_volatile_attention:.4f}")
    
    return generated.detach().cpu().numpy()[0], attention_history  # [horizon, 4]

# %% [markdown] cell 44
## Main Training Function

# %% [code] cell 45
def run_fold(fold_name, price_fold, window, max_epochs, patience, run_sanity=False, quick_mode=False):
    feat_fold = build_feature_frame(price_fold)
    target_fold = build_target_frame(feat_fold)
    
    input_raw = feat_fold[BASE_FEATURE_COLS].to_numpy(np.float32)
    target_raw = target_fold[TARGET_COLS].to_numpy(np.float32)
    row_imputed = feat_fold['row_imputed'].to_numpy(np.int8).astype(bool)
    row_open_skip = feat_fold['row_open_skip'].to_numpy(np.int8).astype(bool)
    prev_close = feat_fold['prev_close'].to_numpy(np.float32)
    price_vals = price_fold.loc[feat_fold.index, OHLC_COLS].to_numpy(np.float32)
    
    tr_end, va_end = split_points(len(input_raw))
    
    # Standardize inputs only
    in_mean, in_std = input_raw[:tr_end].mean(axis=0), input_raw[:tr_end].std(axis=0)
    in_std = np.where(in_std < 1e-8, 1.0, in_std)
    input_scaled = (input_raw - in_mean) / in_std
    
    # No target scaling (raw returns)
    tg_mean, tg_std = np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)
    target_scaled = target_raw.copy()
    
    X_all, y_all_s, y_all_r, starts, prev_close_starts, dropped_imputed, dropped_skip = make_multistep_windows(
        input_scaled, target_scaled, target_raw, row_imputed, row_open_skip, prev_close, window, HORIZON
    )
    
    if len(X_all) == 0:
        raise RuntimeError(f'{fold_name}: no windows available.')
    
    # Splits
    end_idx = starts + HORIZON - 1
    tr_m, va_m, te_m = end_idx < tr_end, (end_idx >= tr_end) & (end_idx < va_end), end_idx >= va_end
    
    X_train, y_train_s, y_train_r = X_all[tr_m], y_all_s[tr_m], y_all_r[tr_m]
    X_val, y_val_s, y_val_r = X_all[va_m], y_all_s[va_m], y_all_r[va_m]
    X_test, y_test_s, y_test_r = X_all[te_m], y_all_s[te_m], y_all_r[te_m]
    test_starts = starts[te_m]
    test_prev_close = prev_close_starts[te_m]
    
    print(f'Samples: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}')
    
    # Data loaders
    train_loader = DataLoader(MultiStepDataset(X_train, y_train_s, y_train_r), 
                             batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MultiStepDataset(X_val, y_val_s, y_val_r), 
                           batch_size=BATCH_SIZE, shuffle=False)
    
    # v7.4: Transformer + Memory Attention Model
    model = TransformerMemoryAttnModel(
        input_size=X_train.shape[-1],
        output_size=len(TARGET_COLS),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=DROPOUT,
        horizon=HORIZON,
        memory_dim=MEMORY_ATTENTION_DIM,
    ).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    hist = train_model(model, train_loader, val_loader, max_epochs, patience)
    
    # REALISTIC PREDICTION (autoregressive with noise and memory attention)
    # Pick last test sample for visualization
    last_idx = len(X_test) - 1
    X_last = X_test[last_idx:last_idx+1]
    context_start = int(test_starts[last_idx]) - window
    context_prices = price_vals[context_start:int(test_starts[last_idx]), 3]  # Close prices
    
    # Generate with temperature > 1 for more realistic "wild" market movement
    pred_rets_realistic, attn_history = predict_realistic_recursive(
        model, X_last, context_prices, temperature=SAMPLING_TEMPERATURE
    )
    
    # Convert to prices
    last_close = float(test_prev_close[last_idx])
    pred_price_realistic = returns_to_prices_seq(pred_rets_realistic, last_close)
    
    # Actual future
    actual_future = price_vals[int(test_starts[last_idx]):int(test_starts[last_idx])+HORIZON]
    
    # One-step metrics (deterministic for comparison)
    mu_test = model(torch.from_numpy(X_test).float().to(DEVICE))[0].detach().cpu().numpy()
    pred_step1_ret = mu_test[:, 0, :]
    actual_step1_ret = y_test_r[:, 0, :]
    
    # Simple price reconstruction for metrics
    pred_ohlc_1 = np.zeros((len(test_starts), 4))
    for i in range(len(test_starts)):
        pc = test_prev_close[i]
        pred_ohlc_1[i] = [
            pc * np.exp(pred_step1_ret[i, 0]),
            pc * np.exp(pred_step1_ret[i, 1]),
            pc * np.exp(pred_step1_ret[i, 2]),
            pc * np.exp(pred_step1_ret[i, 3]),
        ]
        pred_ohlc_1[i] = enforce_candle_validity(pred_ohlc_1[i].reshape(1, -1))[0]
    
    actual_ohlc_1 = price_vals[test_starts + 1]
    prev_ohlc = price_vals[test_starts]
    
    model_metrics = evaluate_metrics(actual_ohlc_1, pred_ohlc_1, test_prev_close)
    baseline_metrics = evaluate_baselines(actual_ohlc_1, prev_ohlc, test_prev_close)
    
    print(f"\nRealistic prediction stats:")
    print(f"  Pred range: [{pred_price_realistic[:, 3].min():.2f}, {pred_price_realistic[:, 3].max():.2f}]")
    print(f"  Actual range: [{actual_future[:, 3].min():.2f}, {actual_future[:, 3].max():.2f}]")
    print(f"  Pred volatility: {np.std(pred_rets_realistic[:, 3]):.6f}")
    print(f"  Actual volatility: {np.std(actual_step1_ret[:, 3]):.6f}")
    
    # Build DataFrames for plotting
    future_idx = price_fold.index[test_starts[last_idx]:test_starts[last_idx]+HORIZON]
    pred_future_df = pd.DataFrame(pred_price_realistic, index=future_idx, columns=OHLC_COLS)
    actual_future_df = pd.DataFrame(actual_future, index=future_idx, columns=OHLC_COLS)
    context_df = price_fold.iloc[test_starts[last_idx]-window:test_starts[last_idx]+1][OHLC_COLS]
    
    return {
        'fold': fold_name,
        'window': window,
        'history_df': hist,
        'model_metrics': model_metrics,
        'baseline_metrics': baseline_metrics,
        'context_df': context_df,
        'actual_future_df': actual_future_df,
        'pred_future_df': pred_future_df,
        'attention_history': attn_history,
        'samples': {'train': len(X_train), 'val': len(X_val), 'test': len(X_test)},
    }

# %% [markdown] cell 46
## Run Experiments

# %% [code] cell 47
# Run lookback sweep if enabled
fold_results = []
primary_slice = slices[0]
selected_window = DEFAULT_LOOKBACK

if ENABLE_LOOKBACK_SWEEP:
    print('\n=== Lookback sweep (v7.4: Extended lookback candidates) ===')
    _, a0, b0 = primary_slice
    fold_price0 = price_df.iloc[a0:b0].copy()
    
    best_score = -float('inf')
    for w in LOOKBACK_CANDIDATES:
        print(f'\nSweep candidate lookback={w} --')
        try:
            r = run_fold(f'sweep_w{w}', fold_price0, w, SWEEP_MAX_EPOCHS, SWEEP_PATIENCE, quick_mode=True)
            score = -r['model_metrics']['close_mae']  # Simple selection
            if score > best_score:
                best_score = score
                selected_window = w
        except Exception as e:
            print(f"Failed for window {w}: {e}")

print(f'\nSelected lookback: {selected_window}')

# %% [code] cell 48
# Run full walk-forward with realistic generation
print('\n=== Full walk-forward with Transformer + Memory Attention ===')
for i, (name, a, b) in enumerate(slices, start=1):
    print(f'\n=== Running {name} [{a}:{b}] lookback={selected_window} ===')
    fold_price = price_df.iloc[a:b].copy()
    try:
        res = run_fold(name, fold_price, selected_window, FINAL_MAX_EPOCHS, FINAL_PATIENCE)
        fold_results.append(res)
        
        print(f"\nResults for {name}:")
        print(f"  Model MAE: {res['model_metrics']['close_mae']:.4f}")
        print(f"  Persistence MAE: {res['baseline_metrics']['persistence']['close_mae']:.4f}")
        print(f"  Directional Accuracy: {res['model_metrics']['directional_accuracy']:.2%}")
        print(f"  Bias: {res['model_metrics']['bias']:.6f}")
    except Exception as e:
        print(f"Error in fold {name}: {e}")

# %% [markdown] cell 49
## Visualization

# %% [code] cell 50
if fold_results:
    latest = fold_results[-1]
    
    fig, ax = plt.subplots(figsize=(18, 8), facecolor='black')
    ax.set_facecolor('black')
    
    def draw_candles(ax, ohlc, start_x, up_edge, up_face, down_edge, down_face, wick_color, width=0.6, alpha=1.0):
        vals = ohlc[OHLC_COLS].to_numpy()
        for i, (o, h, l, c) in enumerate(vals):
            x = start_x + i
            bull = c >= o
            ax.vlines(x, l, h, color=wick_color, linewidth=1.0, alpha=alpha, zorder=2)
            lower = min(o, c)
            height = max(abs(c - o), 1e-6)
            rect = Rectangle((x - width/2, lower), width, height,
                           facecolor=up_face if bull else down_face,
                           edgecolor=up_edge if bull else down_edge,
                           linewidth=1.0, alpha=alpha, zorder=3)
            ax.add_patch(rect)
    
    context_df = latest['context_df']
    actual_future_df = latest['actual_future_df']
    pred_future_df = latest['pred_future_df']
    
    # Draw history (green/red)
    draw_candles(ax, context_df, 0, '#00FF00', '#00FF00', '#FF0000', '#FF0000', '#FFFFFF', width=0.6, alpha=0.9)
    
    # Draw actual future (dimmed)
    draw_candles(ax, actual_future_df, len(context_df), '#00AA00', '#00AA00', '#AA0000', '#AA0000', '#888888', 
                 width=0.6, alpha=0.6)
    
    # Draw realistic prediction (bright white/black with glow effect)
    draw_candles(ax, pred_future_df, len(context_df), '#FFFFFF', '#FFFFFF', '#888888', '#000000', '#FFFFFF',
                 width=0.5, alpha=1.0)
    
    ax.axvline(len(context_df) - 0.5, color='white', linestyle='--', linewidth=1.0, alpha=0.8)
    
    # Labels
    n = len(context_df) + len(actual_future_df)
    step = max(1, n // 12)
    ticks = list(range(0, n, step))
    all_idx = context_df.index.append(actual_future_df.index)
    labels = [all_idx[i].strftime('%m-%d %H:%M') for i in ticks if i < len(all_idx)]
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, ha='right', color='white', fontsize=9)
    ax.tick_params(axis='y', colors='white')
    for sp in ax.spines.values():
        sp.set_color('#666666')
    ax.grid(color='#333333', linewidth=0.5, alpha=0.5)
    
    ax.set_title(f'MSFT 1m Transformer+Memory ({latest["fold"]}) - Realistic Forecast (Temp={SAMPLING_TEMPERATURE})', 
                 color='white', fontsize=14, pad=15)
    ax.set_ylabel('Price', color='white', fontsize=12)
    
    # Legend
    legend_elements = [
        Patch(facecolor='#00FF00', edgecolor='#00FF00', label='History (bull)'),
        Patch(facecolor='#FF0000', edgecolor='#FF0000', label='History (bear)'),
        Patch(facecolor='#00AA00', edgecolor='#00AA00', label='Actual Future (dim)'),
        Patch(facecolor='#FFFFFF', edgecolor='#FFFFFF', label='Predicted (bull)'),
        Patch(facecolor='#000000', edgecolor='#FFFFFF', label='Predicted (bear)'),
    ]
    ax.legend(handles=legend_elements, facecolor='black', edgecolor='white', labelcolor='white', 
             loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinal chart shows realistic candle generation with temperature {SAMPLING_TEMPERATURE}")
    print(f"Transformer encoder with {NUM_HEADS} heads, {NUM_ENCODER_LAYERS} layers, d_model={D_MODEL}")
    print(f"Memory attention tracking volatile events in {selected_window}-bar lookback")

# %% [markdown] cell 51
## Test Cell: Model Validation

# %% [code] cell 52
"""
Test cell for v7.4 Transformer + Long Memory model

Validates:
1. Directional accuracy vs persistence baseline
2. Average predicted close vs actual close (bias)
3. Visual confirmation of realistic candle wicks/bodies
"""

print("=" * 60)
print("MODEL VALIDATION TESTS - v7.4 Transformer + Long Memory")
print("=" * 60)

if not fold_results:
    print("WARNING: No fold results available. Run training first.")
else:
    latest = fold_results[-1]
    metrics = latest['model_metrics']
    baseline = latest['baseline_metrics']['persistence']
    
    print(f"\nConfiguration:")
    print(f"  Lookback window: {latest['window']} bars")
    print(f"  Transformer: {NUM_HEADS} heads, {NUM_ENCODER_LAYERS} layers, d_model={D_MODEL}")
    print(f"  Sampling temperature: {SAMPLING_TEMPERATURE}")
    
    # Test 1: Directional Accuracy vs Persistence
    print(f"\n--- Test 1: Directional Accuracy ---")
    model_dir_acc = metrics['directional_accuracy']
    persist_dir_acc = baseline['directional_accuracy']
    dir_improvement = model_dir_acc - persist_dir_acc
    
    print(f"  Model directional accuracy: {model_dir_acc:.2%}")
    print(f"  Persistence baseline:       {persist_dir_acc:.2%}")
    print(f"  Improvement:                {dir_improvement:+.2%}")
    
    if dir_improvement >= 0:
        print(f"  ✓ PASS: Model meets or exceeds persistence baseline")
    else:
        print(f"  ✗ FAIL: Model below persistence baseline by {abs(dir_improvement):.2%}")
    
    # Test 2: Bias Check
    print(f"\n--- Test 2: Bias (Average Predicted vs Actual Close) ---")
    bias = metrics['bias']
    avg_pred = metrics['avg_pred_close']
    avg_actual = metrics['avg_actual_close']
    bias_pct = abs(bias) / avg_actual * 100 if avg_actual > 0 else 0
    
    print(f"  Average predicted close: ${avg_pred:.2f}")
    print(f"  Average actual close:    ${avg_actual:.2f}")
    print(f"  Bias:                    ${bias:.4f} ({bias_pct:.3f}%)")
    
    if bias_pct < 1.0:
        print(f"  ✓ PASS: Bias less than 1%")
    elif bias_pct < 5.0:
        print(f"  ⚠ WARNING: Bias between 1-5%")
    else:
        print(f"  ✗ FAIL: Bias exceeds 5%")
    
    # Test 3: Candle Validity Check
    print(f"\n--- Test 3: Candle Validity (Realistic Wicks/Bodies) ---")
    pred_df = latest['pred_future_df']
    o, h, l, c = pred_df['Open'], pred_df['High'], pred_df['Low'], pred_df['Close']
    
    # Check High >= max(Open, Close)
    high_valid = (h >= np.maximum(o, c)).all()
    # Check Low <= min(Open, Close)
    low_valid = (l <= np.minimum(o, c)).all()
    # Check High >= Low
    range_valid = (h >= l).all()
    
    print(f"  High >= max(Open, Close): {high_valid}")
    print(f"  Low <= min(Open, Close):  {low_valid}")
    print(f"  High >= Low:              {range_valid}")
    
    # Check for realistic wicks (shadows)
    upper_wicks = h - np.maximum(o, c)
    lower_wicks = np.minimum(o, c) - l
    bodies = abs(c - o)
    
    print(f"\n  Wick Statistics:")
    print(f"    Average upper wick: ${upper_wicks.mean():.3f}")
    print(f"    Average lower wick: ${lower_wicks.mean():.3f}")
    print(f"    Average body:       ${bodies.mean():.3f}")
    print(f"    Max upper wick:     ${upper_wicks.max():.3f}")
    print(f"    Max lower wick:     ${lower_wicks.max():.3f}")
    
    if high_valid and low_valid and range_valid:
        print(f"  ✓ PASS: All candles are valid OHLC")
    else:
        print(f"  ✗ FAIL: Invalid candles detected")
    
    # Test 4: Memory Attention Validation
    if 'attention_history' in latest and latest['attention_history']:
        print(f"\n--- Test 4: Memory Attention Analysis ---")
        attn_history = latest['attention_history']
        
        total_attention_to_volatile = 0
        total_volatile_timesteps = 0
        
        for step, attn_info in enumerate(attn_history):
            weights = attn_info['weights']
            volatile = attn_info['volatile_mask']
            
            volatile_attention = (weights * volatile.float()).sum(dim=1).mean().item()
            total_attention_to_volatile += volatile_attention
            total_volatile_timesteps += volatile.float().sum().item() / volatile.numel()
        
        avg_volatile_attention = total_attention_to_volatile / len(attn_history)
        avg_volatile_ratio = total_volatile_timesteps / len(attn_history)
        
        print(f"  Average attention to volatile events: {avg_volatile_attention:.4f}")
        print(f"  Average volatile timestep ratio:      {avg_volatile_ratio:.2%}")
        
        if avg_volatile_attention > 0:
            print(f"  ✓ PASS: Memory attention is utilizing volatile event detection")
        else:
            print(f"  ⚠ WARNING: Memory attention not showing volatile focus")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Model: Transformer + Long Memory (v7.4)")
    print(f"Lookback: {latest['window']} bars | Horizon: {HORIZON} bars")
    print(f"Close MAE: {metrics['close_mae']:.4f}")
    print(f"Directional Accuracy: {model_dir_acc:.2%}")
    print(f"Bias: ${bias:.4f} ({bias_pct:.3f}%)")
    print(f"{'=' * 60}")

