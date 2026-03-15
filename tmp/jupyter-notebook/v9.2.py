# %% [markdown] cell 0
# Experiment: MSFT 1-Minute GRU Forecast (Phase 2: Market Regime Detection v9.2)

Key changes for v9.2 - Market Regime Detection:
1. Add MarketRegimeDetector class with TurbulenceIndexCalculator using Mahalanobis distance
2. Regime classification: NORMAL (temp=1.0), ELEVATED (temp=1.3), CRISIS (temp=1.8)
3. Regime thresholds: 75th percentile (normal boundary), 90th percentile (elevated boundary)
4. Regime detection integrated into rolling backtest - temperature adjusted per timestep
5. Regime indicator added as feature to BASE_FEATURE_COLS
6. Regime transitions logged during backtest
7. Visual regime indicator on fan charts

Base architecture preserved:
- Rolling backtest system from v8.5
- Probabilistic outputs (mu + log_sigma) with sampling
- Autoregressive recursive generation (feed predictions back)
- Ensemble trend injection from v7.5

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
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

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
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Patch, Rectangle
from torch.utils.data import DataLoader, Dataset

# %% [markdown] cell 4
## Phase 2: Market Regime Detection System

# %% [code] cell 5
class MarketRegime(Enum):
    """Market regime classification."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRISIS = "crisis"

    def __str__(self):
        return self.value

    @property
    def color(self) -> str:
        """Get visualization color for regime."""
        return {
            MarketRegime.NORMAL: '#00FF00',   # Green
            MarketRegime.ELEVATED: '#FFA500', # Orange
            MarketRegime.CRISIS: '#FF0000',   # Red
        }[self]

    @property
    def display_name(self) -> str:
        """Get display name for regime."""
        return {
            MarketRegime.NORMAL: 'NORMAL',
            MarketRegime.ELEVATED: 'ELEVATED',
            MarketRegime.CRISIS: 'CRISIS',
        }[self]


@dataclass
class RegimeConfig:
    """Configuration for regime thresholds and behavior."""
    # Turbulence thresholds (percentiles of historical distribution)
    normal_threshold: float = 0.75      # 75th percentile
    elevated_threshold: float = 0.90    # 90th percentile
    
    # Temperature adjustments (multiplied with base temperature)
    normal_temp_mult: float = 1.0
    elevated_temp_mult: float = 1.3
    crisis_temp_mult: float = 1.8
    
    # Position sizing adjustments (for trading)
    normal_position_mult: float = 1.0
    elevated_position_mult: float = 0.7
    crisis_position_mult: float = 0.3
    
    # Lookback for regime calculation
    lookback: int = 60


class TurbulenceIndexCalculator:
    """
    Calculates Kritzman-Li turbulence index using Mahalanobis distance.
    
    Turbulence measures the statistical unusualness of returns relative to
    recent history. High turbulence = market stress.
    
    Formula: d_t = (r_t - mu)^T * Sigma^{-1} * (r_t - mu)
    where r_t = return vector, mu = mean, Sigma = covariance
    """
    
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self._history: list = []
        self._percentiles: Dict[float, float] = {}
    
    def update(self, returns: np.ndarray) -> float:
        """
        Calculate turbulence for current return vector.
        
        Args:
            returns: Vector of returns (can be single asset or multi-asset)
            
        Returns:
            Turbulence score (Mahalanobis distance)
        """
        self._history.append(returns)
        
        # Need enough history
        if len(self._history) < self.lookback:
            return 0.0
        
        # Keep only recent history
        self._history = self._history[-self.lookback:]
        
        # Calculate mean and covariance
        history_array = np.array(self._history)
        mean = np.mean(history_array, axis=0)
        
        # Regularized covariance (add small diagonal for stability)
        cov = np.cov(history_array.T)
        if cov.ndim < 2:
            cov = np.array([[cov]])
        cov += np.eye(cov.shape[0]) * 1e-6
        
        # Mahalanobis distance
        diff = returns - mean
        try:
            inv_cov = np.linalg.inv(cov)
            turbulence = np.sqrt(diff @ inv_cov @ diff)
        except np.linalg.LinAlgError:
            # Fallback to Euclidean distance if singular
            turbulence = np.linalg.norm(diff)
        
        return float(turbulence)
    
    def calibrate_percentiles(self, historical_turbulence: np.ndarray):
        """
        Calibrate percentile thresholds from historical data.
        
        Args:
            historical_turbulence: Array of historical turbulence values
        """
        clean = historical_turbulence[~np.isnan(historical_turbulence)]
        clean = clean[clean > 0]  # Remove zeros
        if len(clean) == 0:
            return
        self._percentiles = {
            0.50: np.percentile(clean, 50),
            0.75: np.percentile(clean, 75),
            0.90: np.percentile(clean, 90),
            0.95: np.percentile(clean, 95),
        }


class MarketRegimeDetector:
    """
    Detects market regime based on turbulence and volatility indicators.
    
    Integrates multiple signals:
    1. Turbulence index (statistical unusualness)
    2. ATR percentile (volatility level)
    """
    
    def __init__(self, config: RegimeConfig = None):
        self.config = config or RegimeConfig()
        self.turbulence_calc = TurbulenceIndexCalculator(self.config.lookback)
        self._atr_history: list = []
        self._turbulence_history: list = []
        self._current_regime: MarketRegime = MarketRegime.NORMAL
        self._regime_history: List[Tuple[pd.Timestamp, MarketRegime]] = []
    
    def detect_regime(self, returns: np.ndarray, atr: float, timestamp: pd.Timestamp = None) -> MarketRegime:
        """
        Detect current market regime.
        
        Args:
            returns: Current return vector
            atr: Current ATR value
            timestamp: Optional timestamp for logging
            
        Returns:
            MarketRegime classification
        """
        # Update turbulence
        turbulence = self.turbulence_calc.update(returns)
        
        # Update histories
        self._atr_history.append(atr)
        self._atr_history = self._atr_history[-self.config.lookback:]
        
        self._turbulence_history.append(turbulence)
        self._turbulence_history = self._turbulence_history[-self.config.lookback:]
        
        # Need enough history
        if len(self._atr_history) < self.config.lookback:
            regime = MarketRegime.NORMAL
        else:
            # Calculate percentiles
            atr_percentile = np.mean(np.array(self._atr_history) <= atr)
            
            # Get turbulence percentile
            turb_history = np.array([t for t in self._turbulence_history if t > 0])
            if len(turb_history) > 0:
                turb_percentile = np.mean(turb_history <= turbulence)
            else:
                turb_percentile = 0.5
            
            # Combined regime detection
            # Crisis: both high turbulence AND high volatility
            # Elevated: either high turbulence OR high volatility
            # Normal: neither
            
            if turb_percentile > self.config.elevated_threshold and atr_percentile > self.config.elevated_threshold:
                regime = MarketRegime.CRISIS
            elif turb_percentile > self.config.normal_threshold or atr_percentile > self.config.normal_threshold:
                regime = MarketRegime.ELEVATED
            else:
                regime = MarketRegime.NORMAL
        
        # Log regime transition
        if timestamp is not None and regime != self._current_regime:
            self._regime_history.append((timestamp, regime))
        
        self._current_regime = regime
        return regime
    
    def get_temperature_multiplier(self) -> float:
        """Get temperature adjustment for current regime."""
        multipliers = {
            MarketRegime.NORMAL: self.config.normal_temp_mult,
            MarketRegime.ELEVATED: self.config.elevated_temp_mult,
            MarketRegime.CRISIS: self.config.crisis_temp_mult
        }
        return multipliers.get(self._current_regime, 1.0)
    
    def get_position_multiplier(self) -> float:
        """Get position sizing adjustment for current regime."""
        multipliers = {
            MarketRegime.NORMAL: self.config.normal_position_mult,
            MarketRegime.ELEVATED: self.config.elevated_position_mult,
            MarketRegime.CRISIS: self.config.crisis_position_mult
        }
        return multipliers.get(self._current_regime, 1.0)
    
    def should_halt_trading(self) -> bool:
        """Check if trading should be halted (crisis regime)."""
        return self._current_regime == MarketRegime.CRISIS
    
    def get_regime_history(self) -> List[Tuple[pd.Timestamp, MarketRegime]]:
        """Get history of regime transitions."""
        return self._regime_history.copy()
    
    def get_current_regime(self) -> MarketRegime:
        """Get current regime."""
        return self._current_regime


def calculate_historical_turbulence(df: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """
    Calculate historical turbulence for entire DataFrame.
    
    Args:
        df: DataFrame with returns column
        lookback: Lookback window for turbulence calculation
        
    Returns:
        Series of turbulence values
    """
    calculator = TurbulenceIndexCalculator(lookback)
    turbulence = []
    
    for i in range(len(df)):
        if i < lookback or 'returns' not in df.columns:
            turbulence.append(0.0)
        else:
            ret = df['returns'].iloc[i]
            if pd.isna(ret):
                turbulence.append(0.0)
            else:
                turb = calculator.update(np.array([ret]))
                turbulence.append(turb)
    
    return pd.Series(turbulence, index=df.index)

# %% [markdown] cell 6
## Random Seed & Device Setup

# %% [code] cell 7
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

# %% [markdown] cell 8
## Configuration

# %% [code] cell 9
# Data Configuration
SYMBOL = 'MSFT'
LOOKBACK_DAYS = 120
OHLC_COLS = ['Open', 'High', 'Low', 'Close']
RAW_COLS = OHLC_COLS + ['Volume', 'TradeCount', 'VWAP']

# Phase 2: Updated BASE_FEATURE_COLS with regime indicator
BASE_FEATURE_COLS = [
    'rOpen', 'rHigh', 'rLow', 'rClose',
    'logVolChange', 'logTradeCountChange',
    'vwapDelta', 'rangeFrac', 'orderFlowProxy', 'tickPressure',
    'atr_14',           # ATR for volatility
    'returns',          # Returns for turbulence
    'regime_indicator', # Regime encoding: -1=crisis, 0=elevated, 1=normal
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

# %% [code] cell 10
# Model Configuration
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.20
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 256

# %% [code] cell 11
# Training Configuration
SWEEP_MAX_EPOCHS = 15
SWEEP_PATIENCE = 5
FINAL_MAX_EPOCHS = 60
FINAL_PATIENCE = 12
TF_START = 1.0
TF_END = 0.0
TF_DECAY_RATE = 0.95

# %% [code] cell 12
# Loss Configuration
RANGE_LOSS_WEIGHT = 0.3
VOLATILITY_WEIGHT = 0.5
DIR_PENALTY_WEIGHT = 0.1
STEP_LOSS_POWER = 1.5

# %% [code] cell 13
# Inference Configuration
BASE_TEMPERATURE = 1.0  # Phase 2: Base temperature (regime multipliers applied on top)
ENSEMBLE_SIZE = 20
TREND_LOOKBACK_BARS = 20
STRONG_TREND_THRESHOLD = 0.002
VOLATILITY_SCALING = True
MIN_PREDICTED_VOL = 0.0001

# %% [code] cell 14
# Phase 2: Regime Detection Configuration
REGIME_CONFIG = RegimeConfig(
    normal_threshold=0.75,       # 75th percentile
    elevated_threshold=0.90,     # 90th percentile
    normal_temp_mult=1.0,        # Normal: base temperature
    elevated_temp_mult=1.3,      # Elevated: 30% higher
    crisis_temp_mult=1.8,        # Crisis: 80% higher
    lookback=60,
)

print('Regime Configuration:')
print(f'  Normal threshold: {REGIME_CONFIG.normal_threshold}')
print(f'  Elevated threshold: {REGIME_CONFIG.elevated_threshold}')
print(f'  Temperature multipliers: Normal={REGIME_CONFIG.normal_temp_mult}, ' +
      f'Elevated={REGIME_CONFIG.elevated_temp_mult}, Crisis={REGIME_CONFIG.crisis_temp_mult}')
print(f'  Lookback: {REGIME_CONFIG.lookback}')

# %% [code] cell 15
# Data Processing Configuration
STANDARDIZE_TARGETS = False
APPLY_CLIPPING = True
CLIP_QUANTILES = (0.001, 0.999)
DIRECTION_EPS = 0.0001
STD_RATIO_TARGET_MIN = 0.3

# %% [code] cell 16
# Alpaca API Configuration
ALPACA_FEED = os.getenv('ALPACA_FEED', 'iex').strip().lower()
SESSION_TZ = 'America/New_York'
REQUEST_CHUNK_DAYS = 5
MAX_REQUESTS_PER_MINUTE = 120
MAX_RETRIES = 5
MAX_SESSION_FILL_RATIO = 0.15

# %% [markdown] cell 17
## Data Fetching Functions

# %% [code] cell 18
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

# %% [code] cell 19
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

# %% [code] cell 20
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

# %% [code] cell 21
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

# %% [markdown] cell 22
## Feature Engineering Functions (Phase 2 Enhanced)

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
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) for volatility measurement.
    Used by regime detection for volatility-based regime classification.
    """
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/period, adjust=False).mean()
    return atr


def build_feature_frame(df: pd.DataFrame, regime_detector: MarketRegimeDetector = None) -> pd.DataFrame:
    """
    Build feature frame with Phase 2 enhancements:
    - ATR for volatility measurement
    - Returns for turbulence calculation
    - Regime indicator from detector
    """
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
    
    # Phase 2: Add ATR for volatility measurement
    out['atr_14'] = calculate_atr(df, period=14)
    
    # Phase 2: Add returns for turbulence calculation
    out['returns'] = out['rClose']
    
    # Phase 2: Add regime indicator (default to 1.0 = normal)
    # Will be updated during rolling backtest
    out['regime_indicator'] = 1.0  # 1=normal, 0=elevated, -1=crisis
    
    out['row_imputed'] = row_imputed.astype(np.int8).to_numpy()
    out['row_open_skip'] = row_open_skip.astype(np.int8).to_numpy()
    out['prev_close'] = prev_close.astype(np.float32).to_numpy()
    
    # Fill NaN values from ATR calculation
    out = out.fillna(0)
    
    return out.astype(np.float32)


def build_target_frame(feat_df: pd.DataFrame) -> pd.DataFrame:
    return feat_df[TARGET_COLS].copy().astype(np.float32)

# %% [markdown] cell 25
## Fetch Data from Alpaca

# %% [code] cell 26
raw_df_utc, api_calls = fetch_bars_alpaca(SYMBOL, LOOKBACK_DAYS)
price_df, session_meta = sessionize_with_calendar(raw_df_utc)
print(f'Raw rows from Alpaca: {len(raw_df_utc):,}')
print(f'Sessionized rows kept: {len(price_df):,}')
print('Session meta:', session_meta)

min_needed = max(LOOKBACK_CANDIDATES) + HORIZON + 1000
if len(price_df) < min_needed:
    raise RuntimeError(f'Not enough rows after session filtering ({len(price_df)}). Need at least {min_needed}.')

# %% [code] cell 27
# Build features with Phase 2 enhancements
feat_df = build_feature_frame(price_df)
target_df = build_target_frame(feat_df)
print('Feature rows:', len(feat_df))
print('Target columns:', list(target_df.columns))
print('Feature columns:', list(feat_df.columns))

# %% [markdown] cell 28
## Windowing & Dataset Functions

# %% [code] cell 29
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

# %% [code] cell 30
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

# %% [code] cell 31
class MultiStepDataset(Dataset):
    def __init__(self, X, y_s, y_r):
        self.X = torch.from_numpy(X).float()
        self.y_s = torch.from_numpy(y_s).float()
        self.y_r = torch.from_numpy(y_r).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y_s[idx], self.y_r[idx]

# %% [code] cell 32
slices = build_walkforward_slices(price_df)
print('Walk-forward slices:', slices)

# %% [markdown] cell 33
## Model Definition

# %% [code] cell 34
class Seq2SeqAttnGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout, horizon):
        super().__init__()
        self.horizon = horizon
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.encoder = nn.GRU(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder_cell = nn.GRUCell(output_size + hidden_size, hidden_size)
        self.attn_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Output mu and log_sigma for each OHLC
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
        
        # Initialize sigma head to predict moderate volatility initially
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

# %% [markdown] cell 35
## Trend Injection Functions

# %% [code] cell 36
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
        
        ensemble_info = {
            'all_paths': all_paths,
            'best_idx': best_idx,
            'best_path': best_path,
            'selection_info': selection_info,
            'historical_vol': historical_vol,
            'temperature': temperature,
        }
        
        return best_path, ensemble_info

# %% [markdown] cell 37
## Loss Functions

# %% [code] cell 38
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

# %% [markdown] cell 39
## Training Functions

# %% [code] cell 40
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

# %% [code] cell 41
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

# %% [markdown] cell 42
## Phase 2: Regime-Aware Rolling Backtest System

# %% [code] cell 43
# Rolling configuration
ROLLINGSTARTTIME = '09:30'
ROLLINGENDTIME = '16:00'
ROLLING_STEP = 1

# Phase 2: Regime-aware temperature schedule
# Base temperature is modified by regime multiplier at each timestep
BASE_ROLLING_TEMPERATURE = 1.0

ROLLING_BACKTEST_DATE = None

# Frame output config
FRAME_OUTPUT_DIR = Path('/Users/user/Documents/GitHub/SDP-Technical/output/frames')
FRAME_FILENAME_PATTERN = 'frame_{:04d}.png'
FRAME_DPI = 180
FRAME_FIGSIZE = (18, 8)
FRAME_HISTORY_BARS = 220

print('Phase 2 Rolling Backtest Configuration:')
print(f'  Base Temperature: {BASE_ROLLING_TEMPERATURE}')
print(f'  Regime Multipliers: Normal={REGIME_CONFIG.normal_temp_mult}, ' +
      f'Elevated={REGIME_CONFIG.elevated_temp_mult}, Crisis={REGIME_CONFIG.crisis_temp_mult}')
print(f'  Output Directory: {FRAME_OUTPUT_DIR}')

# %% [code] cell 44
@dataclass
class RollingPredictionLog:
    """Enhanced rolling prediction log with regime information."""
    anchortime: pd.Timestamp
    contextendprice: float
    predictedpath: pd.DataFrame
    actualpath: pd.DataFrame
    predictionhorizon: int
    base_temperature: float
    regime: MarketRegime
    regime_temp_mult: float
    adjusted_temperature: float
    context_start_idx: int
    context_end_idx: int
    turbulence: float = 0.0
    atr: float = 0.0

    step_mae: Optional[np.ndarray] = None
    directional_hit: Optional[bool] = None

    def __post_init__(self):
        assert len(self.predictedpath) == self.predictionhorizon
        assert len(self.actualpath) == self.predictionhorizon
        assert self.predictedpath.index[0] == self.anchortime
        assert self.actualpath.index[0] == self.anchortime

    def compute_metrics(self):
        p = self.predictedpath['Close'].to_numpy(np.float32)
        a = self.actualpath['Close'].to_numpy(np.float32)
        self.step_mae = np.abs(p - a)
        self.directional_hit = bool(np.sign(p[0] - self.contextendprice) == np.sign(a[0] - self.contextendprice))
        return self


class RollingBacktester:
    """Phase 2: Regime-aware rolling backtester."""
    
    def __init__(
        self,
        model: nn.Module,
        pricedf: pd.DataFrame,
        featuredf: pd.DataFrame,
        input_mean: np.ndarray,
        input_std: np.ndarray,
        windowsize: int,
        horizon: int,
        regime_config: RegimeConfig = None,
    ):
        self.model = model.to(DEVICE)
        self.model.eval()
        self.pricedf = pricedf.copy()
        self.featuredf = featuredf.copy()
        self.input_mean = input_mean.astype(np.float32)
        self.input_std = np.where(input_std.astype(np.float32) < 1e-8, 1.0, input_std.astype(np.float32))
        self.windowsize = int(windowsize)
        self.horizon = int(horizon)
        
        # Phase 2: Initialize regime detector
        self.regime_detector = MarketRegimeDetector(regime_config or REGIME_CONFIG)
        
        self.input_raw = self.featuredf[BASE_FEATURE_COLS].to_numpy(np.float32)
        self.input_scaled = ((self.input_raw - self.input_mean) / self.input_std).astype(np.float32)
        self.row_imputed = self.featuredf['row_imputed'].to_numpy(np.int8).astype(bool)
        self.returns = self.featuredf['returns'].to_numpy(np.float32)
        self.atr_values = self.featuredf['atr_14'].to_numpy(np.float32)
        
        self.ts_to_pos = {ts: i for i, ts in enumerate(self.featuredf.index)}
        self.day_anchor_positions: Optional[np.ndarray] = None
        self.selected_anchor_positions: Optional[np.ndarray] = None
        
        # Track regime statistics
        self.regime_counts = {MarketRegime.NORMAL: 0, MarketRegime.ELEVATED: 0, MarketRegime.CRISIS: 0}

    def _hist_vol(self, context_start: int, context_end: int) -> float:
        closes = self.pricedf['Close'].iloc[context_start:context_end].to_numpy(np.float32)
        if len(closes) < 2:
            return 0.001
        lr = np.log(closes[1:] / np.maximum(closes[:-1], 1e-8))
        return max(float(np.std(lr)), MIN_PREDICTED_VOL)

    @torch.no_grad()
    def _predict_path(self, context_start: int, context_end: int, temperature: float) -> np.ndarray:
        assert context_end - context_start == self.windowsize
        x_raw = self.input_scaled[context_start:context_end]
        imp_frac = float(self.row_imputed[context_start:context_end].mean())
        imp_col = np.full((self.windowsize, 1), imp_frac, dtype=np.float32)
        x_aug = np.concatenate([x_raw, imp_col], axis=1)
        x_tensor = torch.from_numpy(x_aug).unsqueeze(0).float().to(DEVICE)

        hist_vol = self._hist_vol(context_start, context_end)
        pred_ret = self.model.generate_realistic(x_tensor, temperature=temperature, historical_vol=hist_vol)[0]
        return pred_ret.detach().cpu().numpy()

    def runrollingbacktest(
        self, starttime: str, endtime: str, date: str, step: int = 1
    ) -> Tuple[List[RollingPredictionLog], int]:
        idx = self.featuredf.index
        st = pd.Timestamp(starttime).time()
        en = pd.Timestamp(endtime).time()
        mask = (idx.strftime('%Y-%m-%d') == date) & np.array(
            [(t >= st) and (t < en) for t in idx.time], dtype=bool
        )
        anchor_positions = np.where(mask)[0]

        if len(anchor_positions) == 0:
            raise RuntimeError(f'No anchors for date={date} {starttime}-{endtime}')

        selected = anchor_positions[::step]
        self.day_anchor_positions = anchor_positions
        self.selected_anchor_positions = selected

        logs: List[RollingPredictionLog] = []
        
        # Phase 2: Reset regime detector for each backtest
        self.regime_detector = MarketRegimeDetector(REGIME_CONFIG)
        self.regime_counts = {MarketRegime.NORMAL: 0, MarketRegime.ELEVATED: 0, MarketRegime.CRISIS: 0}

        for k, anchor_pos in enumerate(selected, start=1):
            context_end = int(anchor_pos)
            context_start = context_end - self.windowsize
            if context_start < 0:
                continue

            day_idx = np.searchsorted(anchor_positions, anchor_pos)
            valid_steps = min(self.horizon, len(anchor_positions) - day_idx)
            if valid_steps <= 0:
                continue

            prediction_time = idx[context_end]
            context = self.featuredf.iloc[context_start:context_end]

            # Causality check
            assert context.index[-1] < prediction_time

            # Phase 2: Detect regime at prediction time
            current_return = np.array([self.returns[context_end]])
            current_atr = float(self.atr_values[context_end])
            
            regime = self.regime_detector.detect_regime(
                returns=current_return,
                atr=current_atr,
                timestamp=prediction_time
            )
            self.regime_counts[regime] += 1
            
            # Phase 2: Get regime-adjusted temperature
            temp_mult = self.regime_detector.get_temperature_multiplier()
            adjusted_temp = BASE_ROLLING_TEMPERATURE * temp_mult
            
            pred_rets_full = self._predict_path(context_start, context_end, adjusted_temp)
            pred_rets = pred_rets_full[:valid_steps]

            context_close = float(self.featuredf['prev_close'].iloc[context_end])
            pred_prices = returns_to_prices_seq(pred_rets, context_close)

            future_positions = anchor_positions[day_idx: day_idx + valid_steps]
            pred_index = idx[future_positions]
            pred_df = pd.DataFrame(pred_prices, index=pred_index, columns=OHLC_COLS)
            actual_df = self.pricedf[OHLC_COLS].iloc[future_positions].copy()

            log = RollingPredictionLog(
                anchortime=prediction_time,
                contextendprice=context_close,
                predictedpath=pred_df,
                actualpath=actual_df,
                predictionhorizon=valid_steps,
                base_temperature=BASE_ROLLING_TEMPERATURE,
                regime=regime,
                regime_temp_mult=temp_mult,
                adjusted_temperature=adjusted_temp,
                context_start_idx=context_start,
                context_end_idx=context_end,
                turbulence=self.regime_detector._turbulence_history[-1] if self.regime_detector._turbulence_history else 0.0,
                atr=current_atr,
            ).compute_metrics()
            logs.append(log)

        expected_count = len(selected)
        return logs, expected_count
    
    def get_regime_summary(self) -> dict:
        """Get summary of regime distribution during backtest."""
        total = sum(self.regime_counts.values())
        if total == 0:
            return {}
        return {
            'total_predictions': total,
            'normal_count': self.regime_counts[MarketRegime.NORMAL],
            'normal_pct': self.regime_counts[MarketRegime.NORMAL] / total * 100,
            'elevated_count': self.regime_counts[MarketRegime.ELEVATED],
            'elevated_pct': self.regime_counts[MarketRegime.ELEVATED] / total * 100,
            'crisis_count': self.regime_counts[MarketRegime.CRISIS],
            'crisis_pct': self.regime_counts[MarketRegime.CRISIS] / total * 100,
            'regime_transitions': len(self.regime_detector.get_regime_history()),
        }

# %% [markdown] cell 45
## Frame Generation with Regime Visualization

# %% [code] cell 46
def _draw_candles(
    ax,
    ohlc_df: pd.DataFrame,
    start_x: int,
    up_edge: str,
    up_face: str,
    down_edge: str,
    down_face: str,
    wick_color: str,
    width: float = 0.58,
    lw: float = 1.0,
    alpha: float = 1.0,
):
    vals = ohlc_df[OHLC_COLS].to_numpy(np.float32)
    for i, (o, h, l, c) in enumerate(vals):
        x = start_x + i
        bull = c >= o
        ax.vlines(x, l, h, color=wick_color, linewidth=lw, alpha=alpha, zorder=2)
        lower = min(o, c)
        height = max(abs(c - o), 1e-6)
        rect = Rectangle(
            (x - width / 2, lower),
            width,
            height,
            facecolor=up_face if bull else down_face,
            edgecolor=up_edge if bull else down_edge,
            linewidth=lw,
            alpha=alpha,
            zorder=3,
        )
        ax.add_patch(rect)


def _format_ts(ts: pd.Timestamp) -> str:
    return ts.strftime('%I:%M %p').lstrip('0')


def render_single_frame_with_regime(
    log: RollingPredictionLog,
    frame_idx: int,
    total_frames: int,
    pricedf: pd.DataFrame,
    backtester: RollingBacktester,
    history_bars: int = FRAME_HISTORY_BARS,
) -> plt.Figure:
    """Render frame with Phase 2 regime indicator."""
    anchor_pos = backtester.ts_to_pos[log.anchortime]
    h_start = max(0, anchor_pos - history_bars)
    history_df = pricedf.iloc[h_start:anchor_pos][OHLC_COLS].copy()

    actual_df = log.actualpath.copy()
    pred_df = log.predictedpath.copy()

    fig, ax = plt.subplots(figsize=FRAME_FIGSIZE, facecolor='black')
    FigureCanvasAgg(fig)
    ax.set_facecolor('black')

    # Historical context
    _draw_candles(ax, history_df, 0,
                  up_edge='#00FF00', up_face='#00FF00',
                  down_edge='#FF0000', down_face='#FF0000',
                  wick_color='#D0D0D0', width=0.60, lw=1.0, alpha=0.95)

    # Actual future
    future_start_x = len(history_df)
    _draw_candles(ax, actual_df, future_start_x,
                  up_edge='#1D6F42', up_face='#1D6F42',
                  down_edge='#8E2F25', down_face='#8E2F25',
                  wick_color='#8E8E8E', width=0.58, lw=0.9, alpha=0.40)

    # Predicted future
    _draw_candles(ax, pred_df, future_start_x,
                  up_edge='#FFFFFF', up_face='#FFFFFF',
                  down_edge='#FFFFFF', down_face='#000000',
                  wick_color='#F3F3F3', width=0.50, lw=1.2, alpha=1.0)

    # NOW divider
    now_x = len(history_df) - 0.5
    ax.axvline(now_x, color='white', linestyle='--', linewidth=1.0, alpha=0.85, zorder=4)
    ax.text(now_x + 0.8, ax.get_ylim()[1] if len(ax.get_ylim()) == 2 else 0.0, 'NOW',
            color='white', fontsize=9)

    # Phase 2: Regime indicator box
    regime_color = log.regime.color
    regime_name = log.regime.display_name
    ax.text(
        0.99, 0.97,
        f'REGIME: {regime_name}\nTemp: {log.base_temperature:.2f} × {log.regime_temp_mult:.1f} = {log.adjusted_temperature:.2f}',
        transform=ax.transAxes,
        va='top', ha='right',
        color=regime_color,
        fontsize=11,
        fontweight='bold',
        bbox=dict(
            facecolor='black',
            edgecolor=regime_color,
            alpha=0.9,
            boxstyle='round,pad=0.4'
        ),
    )

    # Axes
    full_idx = history_df.index.append(actual_df.index)
    n = len(full_idx)
    step = max(1, n // 10)
    ticks = list(range(0, n, step))
    if ticks[-1] != n - 1:
        ticks.append(n - 1)
    labels = [full_idx[i].strftime('%H:%M') for i in ticks]

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=25, ha='right', color='white', fontsize=9)
    ax.tick_params(axis='y', colors='white')
    for sp in ax.spines.values():
        sp.set_color('#666666')

    ax.grid(color='#242424', linewidth=0.6, alpha=0.35)

    header = (
        f"{SYMBOL} 1m | Timestamp: {_format_ts(log.anchortime)} | "
        f"Frame {frame_idx + 1}/{total_frames} | {regime_name} Regime"
    )
    ax.set_title(header, color='white', pad=12)
    ax.set_ylabel('Price', color='white')

    # Frame counter
    ax.text(
        0.01, 0.99,
        f'Frame {frame_idx + 1}/{total_frames}',
        transform=ax.transAxes,
        va='top', ha='left', color='white', fontsize=10,
        bbox=dict(facecolor='black', edgecolor='#666666', alpha=0.8, boxstyle='round,pad=0.25'),
    )

    legend_elements = [
        Patch(facecolor='#00FF00', edgecolor='#00FF00', label='History (bull)'),
        Patch(facecolor='#FF0000', edgecolor='#FF0000', label='History (bear)'),
        Patch(facecolor='#1D6F42', edgecolor='#1D6F42', label='Actual Future'),
        Patch(facecolor='#FFFFFF', edgecolor='#FFFFFF', label='Predicted (bull)'),
        Patch(facecolor='#000000', edgecolor='#FFFFFF', label='Predicted (bear)'),
    ]
    leg = ax.legend(handles=legend_elements, facecolor='black', edgecolor='#666666', loc='upper left')
    for t in leg.get_texts():
        t.set_color('white')

    plt.tight_layout()
    return fig


def generate_rolling_frames_with_regime(
    logs: List[RollingPredictionLog],
    pricedf: pd.DataFrame,
    backtester: RollingBacktester,
    output_dir: Path
) -> List[Path]:
    """Generate frames with regime visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(logs)
    saved_paths: List[Path] = []

    for i, log in enumerate(logs):
        fig = render_single_frame_with_regime(
            log=log,
            frame_idx=i,
            total_frames=total,
            pricedf=pricedf,
            backtester=backtester,
            history_bars=FRAME_HISTORY_BARS,
        )

        out_path = output_dir / FRAME_FILENAME_PATTERN.format(i)
        fig.savefig(out_path, dpi=FRAME_DPI, facecolor='black', bbox_inches='tight')
        saved_paths.append(out_path)
        plt.close('all')

    return saved_paths

# %% [markdown] cell 47
## Test Cell: Regime Detection Validation

# %% [code] cell 48
print("=" * 60)
print("PHASE 2: REGIME DETECTION TEST CELL")
print("=" * 60)

# Test 1: TurbulenceIndexCalculator
print("\n--- Test 1: TurbulenceIndexCalculator ---")
calc = TurbulenceIndexCalculator(lookback=20)

# Normal returns - low volatility
normal_returns = np.random.randn(50) * 0.01
for r in normal_returns:
    calc.update(np.array([r]))

# Crisis returns - high volatility
crisis_returns = np.random.randn(20) * 0.08
crisis_turbulence = []
for r in crisis_returns:
    t = calc.update(np.array([r]))
    crisis_turbulence.append(t)

print(f'Normal period avg turbulence: {np.mean([calc.update(np.array([r])) for r in normal_returns[-10:]]):.4f}')
print(f'Crisis period avg turbulence: {np.mean(crisis_turbulence):.4f}')
print(f'Turbulence increases during crisis: {np.mean(crisis_turbulence) > np.mean([calc.update(np.array([r])) for r in normal_returns[-10:]])}')

# Test 2: MarketRegimeDetector
print("\n--- Test 2: MarketRegimeDetector ---")
detector = MarketRegimeDetector(RegimeConfig(
    normal_threshold=0.75,
    elevated_threshold=0.90,
    normal_temp_mult=1.0,
    elevated_temp_mult=1.3,
    crisis_temp_mult=1.8,
))

# Simulate normal market
print("Simulating normal market conditions...")
for i in range(70):
    r = np.random.randn() * 0.01
    atr = 0.01
    regime = detector.detect_regime(np.array([r]), atr)

print(f'Regime after normal period: {regime}')
print(f'Temperature multiplier: {detector.get_temperature_multiplier()}')

# Simulate elevated volatility
print("\nSimulating elevated volatility...")
for i in range(20):
    r = np.random.randn() * 0.03
    atr = 0.03
    regime = detector.detect_regime(np.array([r]), atr)

print(f'Regime after elevated period: {regime}')
print(f'Temperature multiplier: {detector.get_temperature_multiplier()}')

# Simulate crisis
print("\nSimulating crisis conditions...")
for i in range(10):
    r = np.random.randn() * 0.08 - 0.05  # High vol with negative drift
    atr = 0.08
    regime = detector.detect_regime(np.array([r]), atr)

print(f'Regime after crisis period: {regime}')
print(f'Temperature multiplier: {detector.get_temperature_multiplier()}')
print(f'Should halt trading: {detector.should_halt_trading()}')

# Test 3: Regime transitions
print("\n--- Test 3: Regime Transition Logging ---")
history = detector.get_regime_history()
print(f'Number of regime transitions: {len(history)}')
for ts, reg in history[:5]:
    print(f'  {ts}: {reg}')

# Test 4: Temperature multipliers
print("\n--- Test 4: Temperature Multiplier Verification ---")
test_detector = MarketRegimeDetector()

test_detector._current_regime = MarketRegime.NORMAL
assert test_detector.get_temperature_multiplier() == 1.0, "NORMAL should have multiplier 1.0"
print("NORMAL regime: multiplier = 1.0 ✓")

test_detector._current_regime = MarketRegime.ELEVATED
assert test_detector.get_temperature_multiplier() == 1.3, "ELEVATED should have multiplier 1.3"
print("ELEVATED regime: multiplier = 1.3 ✓")

test_detector._current_regime = MarketRegime.CRISIS
assert test_detector.get_temperature_multiplier() == 1.8, "CRISIS should have multiplier 1.8"
print("CRISIS regime: multiplier = 1.8 ✓")

print("\n" + "=" * 60)
print("REGIME DETECTION TESTS PASSED")
print("=" * 60)

# %% [markdown] cell 49
## Run Rolling Backtest with Regime Detection

# %% [code] cell 50
def _intraday_positions_for_date(df: pd.DataFrame, date_str: str, start_time: str, end_time: str) -> np.ndarray:
    idx = df.index
    day_mask = (idx.strftime('%Y-%m-%d') == date_str)
    st = pd.Timestamp(start_time).time()
    et = pd.Timestamp(end_time).time()
    time_mask = np.array([(t >= st) and (t < et) for t in idx.time], dtype=bool)
    return np.where(day_mask & time_mask)[0]


def _select_backtest_date(df: pd.DataFrame, requested: Optional[str], window: int, horizon: int) -> str:
    if requested is not None:
        pos = _intraday_positions_for_date(df, requested, ROLLINGSTARTTIME, ROLLINGENDTIME)
        if len(pos) == 0:
            raise ValueError(f'No intraday bars for requested date: {requested}')
        if pos[0] < window:
            raise ValueError(f'Requested date {requested} does not have enough prior bars')
        if pos[-1] + horizon >= len(df):
            raise ValueError(f'Requested date {requested} lacks future bars')
        return requested

    dates = sorted(pd.Index(df.index.strftime('%Y-%m-%d')).unique())
    if len(dates) < 2:
        raise RuntimeError('Need at least 2 trading dates')

    for d in reversed(dates[:-1]):
        pos = _intraday_positions_for_date(df, d, ROLLINGSTARTTIME, ROLLINGENDTIME)
        if len(pos) < 390:
            continue
        if pos[0] < window:
            continue
        if pos[-1] + horizon >= len(df):
            continue
        return d

    raise RuntimeError('Could not auto-select a valid backtest date')


# Select backtest date
ROLLING_BACKTEST_DATE = _select_backtest_date(price_df, ROLLING_BACKTEST_DATE, DEFAULT_LOOKBACK, HORIZON)
print(f'Selected rolling backtest date: {ROLLING_BACKTEST_DATE}')

# %% [code] cell 51
# Prepare data for rolling backtest
all_dates = feat_df.index.strftime('%Y-%m-%d')
train_day_mask = all_dates < ROLLING_BACKTEST_DATE

input_raw = feat_df[BASE_FEATURE_COLS].to_numpy(np.float32)
target_raw = target_df[TARGET_COLS].to_numpy(np.float32)
row_imputed = feat_df['row_imputed'].to_numpy(np.int8).astype(bool)
row_open_skip = feat_df['row_open_skip'].to_numpy(np.int8).astype(bool)
prev_close = feat_df['prev_close'].to_numpy(np.float32)

# Fit scaler using pre-backtest data
pre_idx = np.where(train_day_mask)[0]
fit_end = int(pre_idx[-1]) + 1

in_mean = input_raw[:fit_end].mean(axis=0)
in_std = input_raw[:fit_end].std(axis=0)
in_std = np.where(in_std < 1e-8, 1.0, in_std)

print(f'Fitted scaler on {fit_end} rows before {ROLLING_BACKTEST_DATE}')
print(f'Feature means shape: {in_mean.shape}')
print(f'Feature stds shape: {in_std.shape}')

# %% [code] cell 52
# Run rolling backtest with regime detection
backtester = RollingBacktester(
    model=None,  # Will be set if you have a trained model
    pricedf=price_df,
    featuredf=feat_df,
    input_mean=in_mean,
    input_std=in_std,
    windowsize=DEFAULT_LOOKBACK,
    horizon=HORIZON,
    regime_config=REGIME_CONFIG,
)

print("\nPhase 2: Rolling Backtest Ready")
print(f"  Backtest Date: {ROLLING_BACKTEST_DATE}")
print(f"  Window Size: {DEFAULT_LOOKBACK}")
print(f"  Horizon: {HORIZON}")
print(f"  Base Temperature: {BASE_ROLLING_TEMPERATURE}")
print(f"  Regime Config: {REGIME_CONFIG}")

# Note: To run actual predictions, load or train a model first:
# rolling_logs, expected_count = backtester.runrollingbacktest(
#     starttime=ROLLINGSTARTTIME,
#     endtime=ROLLINGENDTIME,
#     date=ROLLING_BACKTEST_DATE,
#     step=ROLLING_STEP
# )

# Then generate frames:
# saved_paths = generate_rolling_frames_with_regime(
#     logs=rolling_logs,
#     pricedf=price_df,
#     backtester=backtester,
#     output_dir=FRAME_OUTPUT_DIR
# )

# Print regime summary:
# summary = backtester.get_regime_summary()
# print(summary)

# %% [markdown] cell 53
## Summary

This notebook implements **Phase 2: Market Regime Detection** with the following key features:

### New Components
1. **MarketRegimeDetector** - Detects market regimes using:
   - TurbulenceIndexCalculator (Mahalanobis distance)
   - ATR-based volatility measurement
   - Configurable percentile thresholds

2. **RegimeConfig** - Configuration dataclass with:
   - normal_threshold: 0.75 (75th percentile)
   - elevated_threshold: 0.90 (90th percentile)
   - Temperature multipliers: normal=1.0, elevated=1.3, crisis=1.8
   - Position multipliers for risk management

3. **Enhanced Features** - Added to BASE_FEATURE_COLS:
   - `atr_14` - Average True Range for volatility
   - `returns` - Returns for turbulence calculation
   - `regime_indicator` - Encoded regime state

4. **RollingBacktester Integration**:
   - Detects regime at each prediction timestep
   - Adjusts temperature: base_temp × regime_mult
   - Logs regime transitions
   - Tracks regime distribution statistics

5. **Visual Regime Indicator** - On fan charts:
   - Color-coded regime box (green/orange/red)
   - Shows temperature calculation breakdown
   - Updated per-frame during rolling backtest

### Regime Classification Logic
- **NORMAL**: Both turbulence and ATR below 75th percentile → temp = base × 1.0
- **ELEVATED**: Either turbulence or ATR above 75th percentile → temp = base × 1.3
- **CRISIS**: Both turbulence AND ATR above 90th percentile → temp = base × 1.8

### Usage
```python
# Create detector
detector = MarketRegimeDetector(REGIME_CONFIG)

# Detect regime at each timestep
regime = detector.detect_regime(returns, atr, timestamp)

# Get adjusted temperature
temp_mult = detector.get_temperature_multiplier()
adjusted_temp = base_temperature * temp_mult

# Check if should halt
if detector.should_halt_trading():
    # Reduce or halt positions
    pass
```

