# %% [markdown] cell 0
# Experiment: MSFT 1-Minute FTiTransformer (Phase 4: Time-Frequency Features v9.4)

Key changes for v9.4 - Time-Frequency iTransformer:
1. **FrequencyFeatureExtractor**: STFT-based frequency analysis using torch.stft
   - n_fft=16, hop_length=4 for frequency resolution
   - 1D-CNN to process magnitude spectrum
   - Captures periodic patterns in price data
2. **MultiScaleFrequencyExtractor**: Multiple n_ffts [8, 16, 32] for multi-resolution analysis
   - Fusion layer combining multi-scale frequency features
3. **FTiTransformerEncoder**: Time-Frequency iTransformer
   - Time-domain encoder (original iTransformer-style)
   - Frequency-domain encoder (MultiScaleFrequencyExtractor)
   - Fusion layer combining both domains
   - use_frequency flag to toggle frequency features
4. **Rolling backtest integration**: FTiTransformerEncoder as encoder, keep decoder unchanged
5. **Validation**: Test cell for STFT output shapes and periodic vs random signal distinction

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
import torch.nn.functional as F
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

HORIZON = 50
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
LOOKBACK_CANDIDATES = [64, 96, 160, 256]
DEFAULT_LOOKBACK = 96
ENABLE_LOOKBACK_SWEEP = True
SKIP_OPEN_BARS_TARGET = 6

# %% [code] cell 8
# Model Configuration
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.20
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 256

# Frequency Configuration (Phase 4)
USE_FREQUENCY = True  # Toggle for ablation study
FREQ_N_FFT = 16
FREQ_HOP_LENGTH = 4
FREQ_OUT_CHANNELS = 64
MULTISCALE_N_FFTS = [8, 16, 32]  # Multiple frequency resolutions

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

# %% [code] cell 11
# Inference Configuration
SAMPLING_TEMPERATURE = 1.5
ENSEMBLE_SIZE = 20
TREND_LOOKBACK_BARS = 20
STRONG_TREND_THRESHOLD = 0.002
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
    'ensemble_size': ENSEMBLE_SIZE,
    'use_frequency': USE_FREQUENCY,
    'freq_n_fft': FREQ_N_FFT,
    'multiscale_n_ffts': MULTISCALE_N_FFTS,
    'sampling_temperature': SAMPLING_TEMPERATURE,
    'loss_weights': {
        'range': RANGE_LOSS_WEIGHT,
        'volatility': VOLATILITY_WEIGHT,
        'dir_penalty': DIR_PENALTY_WEIGHT,
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
## Phase 4: Time-Frequency Feature Extractors

# %% [code] cell 27
class FrequencyFeatureExtractor(nn.Module):
    """
    STFT-based frequency feature extractor.
    
    Applies Short-Time Fourier Transform to time series data,
    extracts magnitude spectrum, and processes with 1D-CNN.
    
    Args:
        n_fft: FFT window size (default: 16)
        hop_length: Hop length for STFT (default: 4)
        out_channels: Output channels from CNN (default: 64)
    """
    def __init__(self, n_fft: int = 16, hop_length: int = 4, out_channels: int = 64):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Number of frequency bins = n_fft // 2 + 1
        self.n_freq_bins = n_fft // 2 + 1
        
        # 1D-CNN to process frequency features
        # Input: [batch, n_freq_bins, time_frames]
        self.conv1 = nn.Conv1d(self.n_freq_bins, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Global pooling to get fixed-size output
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, features]
        Returns:
            Frequency features [batch, out_channels]
        """
        batch_size, seq_len, n_features = x.shape
        
        # Process each feature dimension separately and aggregate
        freq_features = []
        
        for feat_idx in range(min(n_features, 4)):  # Process first 4 features (OHLC)
            # Extract single feature: [batch, seq_len]
            signal = x[:, :, feat_idx]
            
            # Pad sequence to accommodate n_fft
            if seq_len < self.n_fft:
                pad_len = self.n_fft - seq_len
                signal = F.pad(signal, (0, pad_len), mode='reflect')
            
            # Apply STFT: [batch, n_freq_bins, time_frames]
            # Using hanning window (default in torch.stft)
            stft_out = torch.stft(
                signal,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=torch.hann_window(self.n_fft, device=x.device),
                center=False,
                return_complex=True
            )
            
            # Get magnitude spectrum
            magnitude = torch.abs(stft_out)  # [batch, n_freq_bins, time_frames]
            
            # Apply log compression for numerical stability
            magnitude = torch.log1p(magnitude)
            
            freq_features.append(magnitude)
        
        # Stack and average across features: [batch, n_freq_bins, time_frames]
        freq_features = torch.stack(freq_features, dim=0).mean(dim=0)
        
        # Process with CNN
        out = self.conv1(freq_features)
        out = self.bn1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.gelu(out)
        
        # Global pooling: [batch, out_channels, 1] -> [batch, out_channels]
        out = self.global_pool(out).squeeze(-1)
        
        return out

# %% [code] cell 28
class MultiScaleFrequencyExtractor(nn.Module):
    """
    Multi-scale frequency extractor using multiple n_fft values.
    
    Captures different frequency resolutions for better periodic pattern detection.
    
    Args:
        n_ffts: List of FFT window sizes (default: [8, 16, 32])
        hop_length: Hop length for STFT (scales with n_fft)
        out_channels: Output channels per scale
    """
    def __init__(self, n_ffts: list[int] = None, hop_length: int = 4, out_channels: int = 64):
        super().__init__()
        self.n_ffts = n_ffts or [8, 16, 32]
        self.hop_length = hop_length
        
        # Create frequency extractors for each scale
        self.extractors = nn.ModuleList([
            FrequencyFeatureExtractor(
                n_fft=n_fft,
                hop_length=max(1, n_fft // 4),  # Scale hop with n_fft
                out_channels=out_channels
            )
            for n_fft in self.n_ffts
        ])
        
        # Fusion layer to combine multi-scale features
        total_channels = out_channels * len(self.n_ffts)
        self.fusion = nn.Sequential(
            nn.Linear(total_channels, out_channels * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels * 2, out_channels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, features]
        Returns:
            Multi-scale frequency features [batch, out_channels]
        """
        # Extract features at each scale
        scale_features = []
        for extractor in self.extractors:
            feat = extractor(x)
            scale_features.append(feat)
        
        # Concatenate all scales
        multi_scale = torch.cat(scale_features, dim=-1)
        
        # Fuse multi-scale features
        out = self.fusion(multi_scale)
        
        return out

# %% [code] cell 29
class FTiTransformerEncoder(nn.Module):
    """
    Time-Frequency iTransformer Encoder.
    
    Combines time-domain encoding (via GRU) with frequency-domain encoding
    for comprehensive time series representation.
    
    Args:
        input_size: Input feature dimension
        hidden_size: Hidden dimension for time encoder
        num_layers: Number of GRU layers
        dropout: Dropout rate
        use_frequency: Whether to use frequency features
        freq_out_channels: Output dimension for frequency encoder
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        use_frequency: bool = True,
        freq_out_channels: int = 64
    ):
        super().__init__()
        self.use_frequency = use_frequency
        self.hidden_size = hidden_size
        
        # Time-domain encoder (GRU-based, similar to original)
        self.time_encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        # Frequency-domain encoder
        if use_frequency:
            self.freq_encoder = MultiScaleFrequencyExtractor(
                n_ffts=MULTISCALE_N_FFTS,
                out_channels=freq_out_channels
            )
            
            # Fusion layer: combine time and frequency features
            self.fusion = nn.Sequential(
                nn.Linear(hidden_size + freq_out_channels, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            )
            
            # Project frequency features to match hidden size for attention
            self.freq_proj = nn.Linear(freq_out_channels, hidden_size)
        else:
            self.freq_encoder = None
            self.fusion = None
            self.freq_proj = None
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch, seq_len, input_size]
        Returns:
            enc_out: Encoder outputs [batch, seq_len, hidden_size]
            h: Final hidden state [num_layers, batch, hidden_size]
        """
        # Time-domain encoding
        time_out, h = self.time_encoder(x)  # [batch, seq_len, hidden_size]
        
        if self.use_frequency and self.freq_encoder is not None:
            # Frequency-domain encoding
            freq_features = self.freq_encoder(x)  # [batch, freq_out_channels]
            
            # Expand frequency features to match sequence length
            freq_expanded = self.freq_proj(freq_features).unsqueeze(1)  # [batch, 1, hidden_size]
            freq_expanded = freq_expanded.expand(-1, time_out.size(1), -1)  # [batch, seq_len, hidden_size]
            
            # Concatenate time and frequency features
            combined = torch.cat([time_out, freq_expanded], dim=-1)  # [batch, seq_len, hidden_size * 2]
            
            # Fuse features
            enc_out = self.fusion(combined)  # [batch, seq_len, hidden_size]
            
            # Augment hidden state with frequency information
            h_freq = self.freq_proj(freq_features).unsqueeze(0).expand(h.size(0), -1, -1)
            h = h + 0.1 * h_freq  # Residual connection with scaled frequency info
        else:
            enc_out = time_out
        
        return enc_out, h

# %% [markdown] cell 30
## Windowing & Dataset Functions

# %% [code] cell 31
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

# %% [code] cell 32
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

# %% [code] cell 33
class MultiStepDataset(Dataset):
    def __init__(self, X, y_s, y_r):
        self.X = torch.from_numpy(X).float()
        self.y_s = torch.from_numpy(y_s).float()
        self.y_r = torch.from_numpy(y_r).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y_s[idx], self.y_r[idx]

# %% [code] cell 34
slices = build_walkforward_slices(price_df)
print('Walk-forward slices:', slices)

# %% [markdown] cell 35
## Model Definition with FTiTransformer

# %% [code] cell 36
class Seq2SeqAttnFTiTransformer(nn.Module):
    """
    Sequence-to-sequence model with FTiTransformer encoder and attention decoder.
    """
    def __init__(
        self, 
        input_size, 
        output_size, 
        hidden_size, 
        num_layers, 
        dropout, 
        horizon,
        use_frequency=True,
        freq_out_channels=64
    ):
        super().__init__()
        self.horizon = horizon
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        # FTiTransformer Encoder
        self.encoder = FTiTransformerEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_frequency=use_frequency,
            freq_out_channels=freq_out_channels
        )
        
        # Decoder (unchanged from v8.5)
        self.decoder_cell = nn.GRUCell(output_size + hidden_size, hidden_size)
        self.attn_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Output heads for mu and log_sigma
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )
        self.log_sigma_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, output_size),
        )
        
        # Initialize
        nn.init.xavier_uniform_(self.mu_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.mu_head[-1].bias)
        nn.init.zeros_(self.log_sigma_head[-1].weight)
        nn.init.zeros_(self.log_sigma_head[-1].bias)
        
    def _attend(self, h_dec, enc_out):
        query = self.attn_proj(h_dec).unsqueeze(2)
        scores = torch.bmm(enc_out, query).squeeze(2)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)
        return context
    
    def forward(self, x, y_teacher=None, teacher_forcing_ratio=0.0, return_sigma=False):
        enc_out, h = self.encoder(x)
        h_dec = h[-1]
        dec_input = x[:, -1, :self.output_size]
        
        mu_seq, sigma_seq = [], []
        for t in range(self.horizon):
            context = self._attend(h_dec, enc_out)
            cell_input = torch.cat([dec_input, context], dim=1)
            h_dec = self.decoder_cell(cell_input, h_dec)
            out_features = torch.cat([h_dec, context], dim=1)
            
            mu = self.mu_head(out_features)
            log_sigma = self.log_sigma_head(out_features)
            
            mu_seq.append(mu.unsqueeze(1))
            sigma_seq.append(log_sigma.unsqueeze(1))
            
            if y_teacher is not None and teacher_forcing_ratio > 0.0:
                if teacher_forcing_ratio >= 1.0 or torch.rand(1).item() < teacher_forcing_ratio:
                    dec_input = y_teacher[:, t, :]
                else:
                    noise = torch.randn_like(mu) * torch.exp(log_sigma).detach()
                    dec_input = mu + noise
            else:
                dec_input = mu
        
        mu_out = torch.cat(mu_seq, dim=1)
        sigma_out = torch.cat(sigma_seq, dim=1)
        
        if return_sigma:
            return mu_out, sigma_out
        return mu_out
    
    def generate_realistic(self, x, temperature=1.0, historical_vol=None, manual_seed=None):
        """Generate realistic price paths with controlled stochasticity."""
        self.eval()
        with torch.no_grad():
            if manual_seed is not None:
                torch.manual_seed(manual_seed)
            
            enc_out, h = self.encoder(x)
            h_dec = h[-1]
            dec_input = x[:, -1, :self.output_size]
            
            generated = []
            for t in range(self.horizon):
                context = self._attend(h_dec, enc_out)
                cell_input = torch.cat([dec_input, context], dim=1)
                h_dec = self.decoder_cell(cell_input, h_dec)
                out_features = torch.cat([h_dec, context], dim=1)
                
                mu = self.mu_head(out_features)
                log_sigma = self.log_sigma_head(out_features)
                
                sigma = torch.exp(log_sigma) * temperature
                
                if historical_vol is not None and t < 5:
                    sigma = torch.ones_like(sigma) * historical_vol
                
                sigma = torch.maximum(sigma, torch.tensor(MIN_PREDICTED_VOL))
                
                noise = torch.randn_like(mu) * sigma
                sample = mu + noise
                
                generated.append(sample.unsqueeze(1))
                dec_input = sample
            
            return torch.cat(generated, dim=1)

# %% [markdown] cell 37
## Trend Injection Functions

# %% [code] cell 38
def calculate_trend_slope(prices: np.ndarray) -> float:
    prices = np.asarray(prices, dtype=np.float32)
    if len(prices) < 2:
        return 0.0
    log_prices = np.log(prices)
    x = np.arange(len(log_prices), dtype=np.float32)
    x_mean = np.mean(x)
    y_mean = np.mean(log_prices)
    numerator = np.sum((x - x_mean) * (log_prices - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    if denominator < 1e-10:
        return 0.0
    return float(numerator / denominator)

def calculate_path_trend(return_ohlc: np.ndarray) -> float:
    close_returns = return_ohlc[:, 3]
    close_prices = np.exp(np.cumsum(close_returns))
    return calculate_trend_slope(close_prices)

def select_best_path_by_trend(
    all_paths: list[np.ndarray], 
    historical_slope: float,
    strong_trend_threshold: float = 0.002
) -> tuple[int, np.ndarray, dict]:
    path_slopes = []
    path_directions = []
    
    for path in all_paths:
        slope = calculate_path_trend(path)
        path_slopes.append(slope)
        path_directions.append(np.sign(slope))
    
    path_slopes = np.array(path_slopes)
    path_directions = np.array(path_directions)
    
    historical_direction = np.sign(historical_slope)
    is_strong_trend = abs(historical_slope) > strong_trend_threshold
    
    if is_strong_trend:
        valid_mask = path_directions == historical_direction
        if not valid_mask.any():
            valid_mask = np.ones(len(all_paths), dtype=bool)
    else:
        valid_mask = np.ones(len(all_paths), dtype=bool)
    
    slope_distances = np.abs(path_slopes - historical_slope)
    slope_distances[~valid_mask] = np.inf
    
    best_idx = int(np.argmin(slope_distances))
    best_path = all_paths[best_idx]
    
    info = {
        'historical_slope': historical_slope,
        'historical_direction': historical_direction,
        'is_strong_trend': is_strong_trend,
        'strong_threshold': strong_trend_threshold,
        'path_slopes': path_slopes.tolist(),
        'path_directions': path_directions.tolist(),
        'valid_mask': valid_mask.tolist(),
        'selected_idx': best_idx,
        'selected_slope': path_slopes[best_idx],
        'rejected_count': int((~valid_mask).sum()),
        'slope_distances': slope_distances[valid_mask].tolist() if valid_mask.any() else [],
    }
    
    return best_idx, best_path, info

def generate_ensemble_with_trend_selection(
    model, 
    X: np.ndarray,
    context_prices: np.ndarray,
    temperature: float = 1.0,
    ensemble_size: int = 20,
    trend_lookback: int = 20
) -> tuple[np.ndarray, dict]:
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(DEVICE)
        
        log_returns = np.log(context_prices[1:] / context_prices[:-1])
        historical_vol = float(np.std(log_returns)) if len(log_returns) > 1 else 0.001
        
        trend_context = context_prices[-trend_lookback:]
        historical_slope = calculate_trend_slope(trend_context)
        
        print(f"Historical realized vol: {historical_vol:.6f}, Temperature: {temperature}")
        print(f"Historical trend slope (last {trend_lookback} bars): {historical_slope:.6f}")
        
        all_paths = []
        for i in range(ensemble_size):
            seed = SEED + i * 1000
            generated = model.generate_realistic(
                X_tensor, 
                temperature=temperature, 
                historical_vol=historical_vol,
                manual_seed=seed
            )
            all_paths.append(generated.detach().cpu().numpy()[0])
        
        best_idx, best_path, selection_info = select_best_path_by_trend(
            all_paths, 
            historical_slope,
            STRONG_TREND_THRESHOLD
        )
        
        print(f"Generated {ensemble_size} paths, selected path {best_idx}")
        print(f"  Selected path slope: {selection_info['selected_slope']:.6f}")
        print(f"  Strong trend: {selection_info['is_strong_trend']}, Rejected: {selection_info['rejected_count']}")
        
        ensemble_info = {
            'all_paths': all_paths,
            'best_idx': best_idx,
            'best_path': best_path,
            'selection_info': selection_info,
            'historical_vol': historical_vol,
            'temperature': temperature,
        }
        
        return best_path, ensemble_info

# %% [markdown] cell 39
## Loss Functions

# %% [code] cell 40
def nll_loss(mu, log_sigma, target):
    sigma = torch.exp(log_sigma)
    nll = 0.5 * ((target - mu) / sigma) ** 2 + log_sigma + 0.5 * np.log(2 * np.pi)
    return nll.mean()

def candle_range_loss(mu, target):
    pred_range = mu[:, :, 1] - mu[:, :, 2]
    actual_range = target[:, :, 1] - target[:, :, 2]
    return ((pred_range - actual_range) ** 2).mean()

def volatility_match_loss(log_sigma, target):
    pred_vol = torch.exp(log_sigma).mean()
    actual_vol = target.std()
    return (pred_vol - actual_vol) ** 2

def directional_penalty(mu, target):
    pred_close = mu[:, :, 3]
    actual_close = target[:, :, 3]
    sign_match = torch.sign(pred_close) * torch.sign(actual_close)
    penalty = torch.clamp(-sign_match, min=0.0)
    return penalty.mean()

# %% [markdown] cell 41
## Training Functions

# %% [code] cell 42
def tf_ratio_for_epoch(epoch):
    ratio = TF_START * (TF_DECAY_RATE ** (epoch - 1))
    return max(float(TF_END), float(ratio))

def run_epoch(model, loader, step_weights_t, optimizer=None, tf_ratio=0.0):
    is_train = optimizer is not None
    model.train(is_train)
    
    total_loss, nll_total, range_total, vol_total, dir_total = 0, 0, 0, 0, 0
    n_items = 0
    
    for xb, yb_s, yb_r in loader:
        xb = xb.to(DEVICE)
        yb_s = yb_s.to(DEVICE)
        
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            
        with torch.set_grad_enabled(is_train):
            mu, log_sigma = model(xb, y_teacher=yb_s if is_train else None, 
                                  teacher_forcing_ratio=tf_ratio if is_train else 0.0, 
                                  return_sigma=True)
            
            nll = (nll_loss(mu, log_sigma, yb_s) * step_weights_t).mean()
            rng = candle_range_loss(mu, yb_s)
            vol = volatility_match_loss(log_sigma, yb_s)
            dir_pen = directional_penalty(mu, yb_s)
            
            loss = nll + RANGE_LOSS_WEIGHT * rng + VOLATILITY_WEIGHT * vol + DIR_PENALTY_WEIGHT * dir_pen
            
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
        n_items += bs
        
    return {
        'total': total_loss / max(n_items, 1),
        'nll': nll_total / max(n_items, 1),
        'range': range_total / max(n_items, 1),
        'vol': vol_total / max(n_items, 1),
        'dir': dir_total / max(n_items, 1),
    }

# %% [code] cell 43
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
        })
        
        print(f"Epoch {epoch:02d} | tf={tf:.3f} | "
              f"train={tr['total']:.6f} (nll={tr['nll']:.6f}) | "
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

# %% [markdown] cell 44
## Evaluation Functions

# %% [code] cell 45
def evaluate_metrics(actual_ohlc, pred_ohlc, prev_close):
    actual_ohlc = np.asarray(actual_ohlc, dtype=np.float32)
    pred_ohlc = np.asarray(pred_ohlc, dtype=np.float32)
    ac, pc = actual_ohlc[:, 3], pred_ohlc[:, 3]
    
    return {
        'close_mae': float(np.mean(np.abs(ac - pc))),
        'close_rmse': float(np.sqrt(np.mean((ac - pc) ** 2))),
        'ohlc_mae': float(np.mean(np.abs(actual_ohlc - pred_ohlc))),
        'directional_accuracy_eps': float(np.mean(np.sign(ac - prev_close) == np.sign(pc - prev_close))),
    }

def evaluate_baselines(actual_ohlc, prev_ohlc, prev_close):
    persistence = evaluate_metrics(actual_ohlc, prev_ohlc, prev_close)
    flat = np.repeat(prev_close.reshape(-1, 1), 4, axis=1).astype(np.float32)
    flat_rw = evaluate_metrics(actual_ohlc, flat, prev_close)
    return {'persistence': persistence, 'flat_close_rw': flat_rw}

# %% [markdown] cell 46
## Main Training Function

# %% [code] cell 47
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
    
    in_mean, in_std = input_raw[:tr_end].mean(axis=0), input_raw[:tr_end].std(axis=0)
    in_std = np.where(in_std < 1e-8, 1.0, in_std)
    input_scaled = (input_raw - in_mean) / in_std
    
    tg_mean, tg_std = np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)
    target_scaled = target_raw.copy()
    
    X_all, y_all_s, y_all_r, starts, prev_close_starts, dropped_imputed, dropped_skip = make_multistep_windows(
        input_scaled, target_scaled, target_raw, row_imputed, row_open_skip, prev_close, window, HORIZON
    )
    
    if len(X_all) == 0:
        raise RuntimeError(f'{fold_name}: no windows available.')
    
    end_idx = starts + HORIZON - 1
    tr_m, va_m, te_m = end_idx < tr_end, (end_idx >= tr_end) & (end_idx < va_end), end_idx >= va_end
    
    X_train, y_train_s, y_train_r = X_all[tr_m], y_all_s[tr_m], y_all_r[tr_m]
    X_val, y_val_s, y_val_r = X_all[va_m], y_all_s[va_m], y_all_r[va_m]
    X_test, y_test_s, y_test_r = X_all[te_m], y_all_s[te_m], y_all_r[te_m]
    test_starts = starts[te_m]
    test_prev_close = prev_close_starts[te_m]
    
    print(f'Samples: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}')
    
    train_loader = DataLoader(MultiStepDataset(X_train, y_train_s, y_train_r), 
                             batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MultiStepDataset(X_val, y_val_s, y_val_r), 
                           batch_size=BATCH_SIZE, shuffle=False)
    
    # FTiTransformer Model
    model = Seq2SeqAttnFTiTransformer(
        input_size=X_train.shape[-1],
        output_size=len(TARGET_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        horizon=HORIZON,
        use_frequency=USE_FREQUENCY,
        freq_out_channels=FREQ_OUT_CHANNELS
    ).to(DEVICE)
    
    hist = train_model(model, train_loader, val_loader, max_epochs, patience)
    
    last_idx = len(X_test) - 1
    X_last = X_test[last_idx:last_idx+1]
    context_start = int(test_starts[last_idx]) - window
    context_prices = price_vals[context_start:int(test_starts[last_idx]), 3]
    
    best_rets, ensemble_info = generate_ensemble_with_trend_selection(
        model, X_last, context_prices, 
        temperature=SAMPLING_TEMPERATURE,
        ensemble_size=ENSEMBLE_SIZE,
        trend_lookback=TREND_LOOKBACK_BARS
    )
    
    last_close = float(test_prev_close[last_idx])
    pred_price_best = returns_to_prices_seq(best_rets, last_close)
    
    all_price_paths = []
    for path_rets in ensemble_info['all_paths']:
        path_prices = returns_to_prices_seq(path_rets, last_close)
        all_price_paths.append(path_prices)
    ensemble_info['all_price_paths'] = all_price_paths
    
    actual_future = price_vals[int(test_starts[last_idx]):int(test_starts[last_idx])+HORIZON]
    
    mu_test = model(torch.from_numpy(X_test).float().to(DEVICE)).detach().cpu().numpy()
    pred_step1_ret = mu_test[:, 0, :]
    actual_step1_ret = y_test_r[:, 0, :]
    
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
    
    print(f"\nEnsemble prediction stats:")
    print(f"  Pred range (best): [{pred_price_best[:, 3].min():.2f}, {pred_price_best[:, 3].max():.2f}]")
    print(f"  Actual range: [{actual_future[:, 3].min():.2f}, {actual_future[:, 3].max():.2f}]")
    print(f"  Pred volatility (best): {np.std(best_rets[:, 3]):.6f}")
    print(f"  Actual volatility: {np.std(actual_step1_ret[:, 3]):.6f}")
    
    future_idx = price_fold.index[test_starts[last_idx]:test_starts[last_idx]+HORIZON]
    pred_future_df = pd.DataFrame(pred_price_best, index=future_idx, columns=OHLC_COLS)
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
        'ensemble_info': ensemble_info,
        'samples': {'train': len(X_train), 'val': len(X_val), 'test': len(X_test)},
    }

# %% [markdown] cell 48
## Test Cell: STFT Output Validation

# %% [code] cell 49
def test_stft_output_shapes():
    """Test STFT output shapes and validate periodic vs random signal distinction."""
    print("="*60)
    print("STFT OUTPUT VALIDATION TEST")
    print("="*60)
    
    batch_size = 4
    seq_len = 96
    n_features = 4
    
    # Create test signals
    t = np.linspace(0, 4*np.pi, seq_len)
    
    # Periodic signal (sine wave)
    periodic_signal = np.sin(t).astype(np.float32)
    
    # Random signal
    np.random.seed(42)
    random_signal = np.random.randn(seq_len).astype(np.float32) * 0.5
    
    # Create batch with both signal types
    x_periodic = torch.from_numpy(np.tile(periodic_signal, (batch_size, n_features, 1))).transpose(1, 2)
    x_random = torch.from_numpy(np.tile(random_signal, (batch_size, n_features, 1))).transpose(1, 2)
    
    # Initialize frequency extractor
    freq_extractor = FrequencyFeatureExtractor(
        n_fft=FREQ_N_FFT, 
        hop_length=FREQ_HOP_LENGTH, 
        out_channels=FREQ_OUT_CHANNELS
    ).to(DEVICE)
    
    x_periodic = x_periodic.to(DEVICE)
    x_random = x_random.to(DEVICE)
    
    # Test forward pass
    with torch.no_grad():
        freq_periodic = freq_extractor(x_periodic)
        freq_random = freq_extractor(x_random)
    
    # Validate output shape
    expected_shape = (batch_size, FREQ_OUT_CHANNELS)
    assert freq_periodic.shape == expected_shape, f"Expected {expected_shape}, got {freq_periodic.shape}"
    assert freq_random.shape == expected_shape, f"Expected {expected_shape}, got {freq_random.shape}"
    
    print(f"✓ Output shape test passed: {freq_periodic.shape}")
    
    # Test periodic vs random signal distinction
    periodic_energy = torch.norm(freq_periodic, dim=1).mean().item()
    random_energy = torch.norm(freq_random, dim=1).mean().item()
    
    print(f"  Periodic signal frequency energy: {periodic_energy:.4f}")
    print(f"  Random signal frequency energy: {random_energy:.4f}")
    
    # Periodic signal should have more concentrated energy
    print(f"  Energy ratio (periodic/random): {periodic_energy/random_energy:.4f}")
    
    # Test MultiScaleFrequencyExtractor
    print("\n--- MultiScaleFrequencyExtractor Test ---")
    multi_extractor = MultiScaleFrequencyExtractor(
        n_ffts=MULTISCALE_N_FFTS,
        out_channels=FREQ_OUT_CHANNELS
    ).to(DEVICE)
    
    with torch.no_grad():
        multi_freq = multi_extractor(x_periodic)
    
    expected_multi_shape = (batch_size, FREQ_OUT_CHANNELS)
    assert multi_freq.shape == expected_multi_shape, f"Expected {expected_multi_shape}, got {multi_freq.shape}"
    
    print(f"✓ Multi-scale output shape test passed: {multi_freq.shape}")
    
    # Test FTiTransformerEncoder
    print("\n--- FTiTransformerEncoder Test ---")
    encoder = FTiTransformerEncoder(
        input_size=n_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        use_frequency=True,
        freq_out_channels=FREQ_OUT_CHANNELS
    ).to(DEVICE)
    
    with torch.no_grad():
        enc_out, h = encoder(x_periodic)
    
    expected_enc_shape = (batch_size, seq_len, HIDDEN_SIZE)
    expected_h_shape = (NUM_LAYERS, batch_size, HIDDEN_SIZE)
    
    assert enc_out.shape == expected_enc_shape, f"Expected enc_out {expected_enc_shape}, got {enc_out.shape}"
    assert h.shape == expected_h_shape, f"Expected h {expected_h_shape}, got {h.shape}"
    
    print(f"✓ Encoder output shape test passed")
    print(f"  enc_out: {enc_out.shape}")
    print(f"  h: {h.shape}")
    
    # Test without frequency features
    encoder_no_freq = FTiTransformerEncoder(
        input_size=n_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        use_frequency=False
    ).to(DEVICE)
    
    with torch.no_grad():
        enc_out_no_freq, h_no_freq = encoder_no_freq(x_periodic)
    
    assert enc_out_no_freq.shape == expected_enc_shape
    print(f"✓ Encoder (no frequency) test passed")
    
    print("\n" + "="*60)
    print("ALL STFT VALIDATION TESTS PASSED!")
    print("="*60)
    
    return True

# Run STFT validation test
test_stft_output_shapes()

# %% [markdown] cell 50
## Run Experiments

# %% [code] cell 51
fold_results = []
primary_slice = slices[0]
selected_window = DEFAULT_LOOKBACK

if ENABLE_LOOKBACK_SWEEP:
    print('\n=== Lookback sweep ===')
    _, a0, b0 = primary_slice
    fold_price0 = price_df.iloc[a0:b0].copy()
    
    best_score = -float('inf')
    for w in LOOKBACK_CANDIDATES:
        print(f'\nSweep candidate lookback={w} --')
        try:
            r = run_fold(f'sweep_w{w}', fold_price0, w, SWEEP_MAX_EPOCHS, SWEEP_PATIENCE, quick_mode=True)
            score = -r['model_metrics']['close_mae']
            if score > best_score:
                best_score = score
                selected_window = w
        except Exception as e:
            print(f"Failed for window {w}: {e}")

print(f'\nSelected lookback: {selected_window}')

# %% [code] cell 52
print('\n=== Full walk-forward with FTiTransformer ===')
for i, (name, a, b) in enumerate(slices, start=1):
    print(f'\n=== Running {name} [{a}:{b}] lookback={selected_window} ===')
    fold_price = price_df.iloc[a:b].copy()
    try:
        res = run_fold(name, fold_price, selected_window, FINAL_MAX_EPOCHS, FINAL_PATIENCE)
        fold_results.append(res)
        
        print(f"\nResults for {name}:")
        print(f"  Model MAE: {res['model_metrics']['close_mae']:.4f}")
        print(f"  Persistence MAE: {res['baseline_metrics']['persistence']['close_mae']:.4f}")
    except Exception as e:
        print(f"Error in fold {name}: {e}")

# %% [markdown] cell 53
## Visualization

# %% [code] cell 54
if fold_results:
    latest = fold_results[-1]
    ensemble_info = latest['ensemble_info']
    selection_info = ensemble_info['selection_info']
    
    fig, axes = plt.subplots(2, 1, figsize=(18, 12), facecolor='black', 
                            gridspec_kw={'height_ratios': [2, 1]})
    ax_main = axes[0]
    ax_trend = axes[1]
    ax_main.set_facecolor('black')
    ax_trend.set_facecolor('black')
    
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
    
    def draw_path_line(ax, price_path, start_x, color, alpha=0.3, linewidth=1, label=None):
        closes = price_path[:, 3]
        x_vals = np.arange(len(closes)) + start_x
        ax.plot(x_vals, closes, color=color, alpha=alpha, linewidth=linewidth, label=label)
    
    context_df = latest['context_df']
    actual_future_df = latest['actual_future_df']
    pred_future_df = latest['pred_future_df']
    
    draw_candles(ax_main, context_df, 0, '#00FF00', '#00FF00', '#FF0000', '#FF0000', '#FFFFFF', width=0.6, alpha=0.9)
    
    all_price_paths = ensemble_info['all_price_paths']
    best_idx = ensemble_info['best_idx']
    
    for i, path_prices in enumerate(all_price_paths):
        if i == best_idx:
            continue
        if selection_info['valid_mask'][i]:
            color = '#444444'
            alpha = 0.2
        else:
            color = '#330000'
            alpha = 0.15
        draw_path_line(ax_main, path_prices, len(context_df), color, alpha=alpha, linewidth=1)
    
    draw_path_line(ax_main, all_price_paths[best_idx], len(context_df), '#00FFFF', alpha=0.8, 
                   linewidth=2, label=f'Best Path (#{best_idx})')
    
    draw_candles(ax_main, actual_future_df, len(context_df), '#00AA00', '#00AA00', '#AA0000', '#AA0000', '#888888', 
                 width=0.6, alpha=0.6)
    
    draw_candles(ax_main, pred_future_df, len(context_df), '#00FFFF', '#00FFFF', '#0088AA', '#0088AA', '#00FFFF',
                 width=0.5, alpha=0.9)
    
    ax_main.axvline(len(context_df) - 0.5, color='white', linestyle='--', linewidth=1.0, alpha=0.8)
    
    n = len(context_df) + len(actual_future_df)
    step = max(1, n // 12)
    ticks = list(range(0, n, step))
    all_idx = context_df.index.append(actual_future_df.index)
    labels = [all_idx[i].strftime('%m-%d %H:%M') for i in ticks if i < len(all_idx)]
    
    ax_main.set_xticks(ticks)
    ax_main.set_xticklabels(labels, rotation=30, ha='right', color='white', fontsize=9)
    ax_main.tick_params(axis='y', colors='white')
    for sp in ax_main.spines.values():
        sp.set_color('#666666')
    ax_main.grid(color='#333333', linewidth=0.5, alpha=0.5)
    
    title_text = (f"MSFT 1m FTiTransformer ({latest['fold']}) | "
                  f"Temp={SAMPLING_TEMPERATURE}, Ensemble={ENSEMBLE_SIZE} | "
                  f"Selected Path: #{best_idx}")
    ax_main.set_title(title_text, color='white', fontsize=14, pad=15)
    ax_main.set_ylabel('Price', color='white', fontsize=12)
    
    legend_elements = [
        Patch(facecolor='#00FF00', edgecolor='#00FF00', label='History (bull)'),
        Patch(facecolor='#FF0000', edgecolor='#FF0000', label='History (bear)'),
        Patch(facecolor='#00AA00', edgecolor='#00AA00', label='Actual Future'),
        Patch(facecolor='#00FFFF', edgecolor='#00FFFF', label='Selected Prediction'),
    ]
    ax_main.legend(handles=legend_elements, facecolor='black', edgecolor='white', labelcolor='white', 
                  loc='upper left', fontsize=10)
    
    path_slopes = selection_info['path_slopes']
    historical_slope = selection_info['historical_slope']
    valid_mask = selection_info['valid_mask']
    
    valid_slopes = [s for s, v in zip(path_slopes, valid_mask) if v]
    rejected_slopes = [s for s, v in zip(path_slopes, valid_mask) if not v]
    
    if valid_slopes:
        ax_trend.hist(valid_slopes, bins=15, color='#00AA00', alpha=0.6, label='Valid Paths')
    if rejected_slopes:
        ax_trend.hist(rejected_slopes, bins=10, color='#AA0000', alpha=0.4, label='Rejected Paths')
    
    ax_trend.axvline(historical_slope, color='#FFFF00', linestyle='-', linewidth=2, 
                    label=f'Historical Slope: {historical_slope:.6f}')
    
    selected_slope = selection_info['selected_slope']
    ax_trend.axvline(selected_slope, color='#00FFFF', linestyle='--', linewidth=2,
                    label=f'Selected Slope: {selected_slope:.6f}')
    
    if selection_info['is_strong_trend']:
        thresh = selection_info['strong_threshold']
        if historical_slope > 0:
            ax_trend.axvspan(-thresh*2, -thresh, alpha=0.2, color='red', label='Rejection Zone')
        else:
            ax_trend.axvspan(thresh, thresh*2, alpha=0.2, color='red', label='Rejection Zone')
    
    ax_trend.set_xlabel('Trend Slope (log-return per step)', color='white', fontsize=11)
    ax_trend.set_ylabel('Path Count', color='white', fontsize=11)
    ax_trend.set_title('Momentum Confidence: Trend Slope Distribution', color='white', fontsize=12)
    ax_trend.tick_params(axis='x', colors='white')
    ax_trend.tick_params(axis='y', colors='white')
    for sp in ax_trend.spines.values():
        sp.set_color('#666666')
    ax_trend.legend(facecolor='black', edgecolor='white', labelcolor='white', loc='upper right')
    ax_trend.grid(color='#333333', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== Momentum Confidence Report ===")
    print(f"Historical trend (last {TREND_LOOKBACK_BARS} bars): {historical_slope:.6f}")
    print(f"Strong trend detected: {selection_info['is_strong_trend']}")
    print(f"Paths rejected (wrong direction): {selection_info['rejected_count']}/{ENSEMBLE_SIZE}")
    print(f"Selected path index: {best_idx}")
    print(f"Selected path slope: {selected_slope:.6f}")
    print(f"Slope difference: {abs(selected_slope - historical_slope):.6f}")
    print(f"\nAll path slopes: {[f'{s:.4f}' for s in path_slopes]}")

# %% [markdown] cell 55
## Summary Report

# %% [code] cell 56
print("="*70)
print("FTiTransformer (Phase 4: Time-Frequency Features) Summary")
print("="*70)

print(f"\nConfiguration:")
print(f"  Use Frequency Features: {USE_FREQUENCY}")
print(f"  Frequency n_fft: {FREQ_N_FFT}")
print(f"  Frequency hop_length: {FREQ_HOP_LENGTH}")
print(f"  Multi-scale n_ffts: {MULTISCALE_N_FFTS}")
print(f"  Frequency out channels: {FREQ_OUT_CHANNELS}")
print(f"  Hidden size: {HIDDEN_SIZE}")
print(f"  Num layers: {NUM_LAYERS}")

if fold_results:
    print(f"\nResults across {len(fold_results)} fold(s):")
    for res in fold_results:
        print(f"\n  {res['fold']}:")
        print(f"    Window: {res['window']}")
        print(f"    Samples: {res['samples']}")
        print(f"    Model MAE: {res['model_metrics']['close_mae']:.4f}")
        print(f"    Directional Acc: {res['model_metrics']['directional_accuracy_eps']:.2%}")

print("\n" + "="*70)

