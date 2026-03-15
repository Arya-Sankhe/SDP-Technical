# %% [markdown] cell 0
# Phase 3: iTransformer Architecture with Hybrid Encoder (v9.3)

Key changes for v9.3 - iTransformer Architecture:
1. **iTransformerEncoderLayer**: Variable attention across features (NOT time)
   - Multi-head attention across feature dimension
   - Time FFN: Feed-forward network across time dimension
   - Key innovation: [batch, time, features] → [batch, features, time, d_model]

2. **iTransformerEncoder**: Stack of iTransformerEncoderLayer with:
   - Input projection to d_model
   - Sinusoidal positional encoding
   - Global average pooling over time

3. **HybridEncoder**: Combines iTransformer + GRU
   - iTransformer for cross-variable relationships
   - GRU for temporal dynamics
   - Fusion layer combining both outputs

4. **Preserve all v7.5/v8 features**:
   - Rolling backtest (v8)
   - Ensemble trend injection (v7.5)
   - Autoregressive generation
   - Probabilistic outputs (mu + log_sigma)

Configuration:
- d_model: 128
- n_heads: 8
- n_layers: 2
- lookback: 60

# %% [code] cell 1
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

# %% [code] cell 2
from __future__ import annotations
import copy
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple

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
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Patch, Rectangle
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# %% [code] cell 3
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

# %% [markdown] cell 4
## Data Configuration

# %% [code] cell 5
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
DEFAULT_LOOKBACK = 60  # Phase 3: Changed to 60 for iTransformer
ENABLE_LOOKBACK_SWEEP = True
SKIP_OPEN_BARS_TARGET = 6

# %% [markdown] cell 6
## Model Configuration - Phase 3: iTransformer

# %% [code] cell 7
# iTransformer specific
D_MODEL = 128      # iTransformer embedding dimension
N_HEADS = 8        # Number of attention heads
N_LAYERS = 2       # Number of iTransformer encoder layers
DROPOUT = 0.20     # Dropout rate

# GRU component (kept for temporal dynamics)
HIDDEN_SIZE = 256  # GRU hidden size
NUM_LAYERS = 2     # GRU layers

# Training
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 256

# %% [markdown] cell 8
## Training Configuration

# %% [code] cell 9
SWEEP_MAX_EPOCHS = 15
SWEEP_PATIENCE = 5
FINAL_MAX_EPOCHS = 60
FINAL_PATIENCE = 12
TF_START = 1.0
TF_END = 0.0
TF_DECAY_RATE = 0.95

# %% [markdown] cell 10
## Loss Configuration

# %% [code] cell 11
RANGE_LOSS_WEIGHT = 0.3
VOLATILITY_WEIGHT = 0.5
DIR_PENALTY_WEIGHT = 0.1
STEP_LOSS_POWER = 1.5

# %% [markdown] cell 12
## Inference Configuration

# %% [code] cell 13
SAMPLING_TEMPERATURE = 1.5
ENSEMBLE_SIZE = 20
TREND_LOOKBACK_BARS = 20
STRONG_TREND_THRESHOLD = 0.002
VOLATILITY_SCALING = True
MIN_PREDICTED_VOL = 0.0001

# %% [markdown] cell 14
## Data Processing Configuration

# %% [code] cell 15
STANDARDIZE_TARGETS = False
APPLY_CLIPPING = True
CLIP_QUANTILES = (0.001, 0.999)
DIRECTION_EPS = 0.0001
STD_RATIO_TARGET_MIN = 0.3

# %% [markdown] cell 16
## Alpaca API Configuration

# %% [code] cell 17
ALPACA_FEED = os.getenv('ALPACA_FEED', 'iex').strip().lower()
SESSION_TZ = 'America/New_York'
REQUEST_CHUNK_DAYS = 5
MAX_REQUESTS_PER_MINUTE = 120
MAX_RETRIES = 5
MAX_SESSION_FILL_RATIO = 0.15

# %% [markdown] cell 18
## V8 Rolling Configuration

# %% [code] cell 19
ROLLINGSTARTTIME = '09:30'
ROLLINGENDTIME = '16:00'
ROLLING_STEP = 1
DEFAULT_ROLLING_TEMPERATURE = 1.5
USE_TEMPERATURE_SCHEDULE = True
TEMPERATURESCHEDULE = [
    ('09:30', '10:15', 1.25),
    ('10:15', '14:00', 1.45),
    ('14:00', '16:00', 1.60),
]
ROLLING_BACKTEST_DATE = None
FRAME_OUTPUT_DIR = Path('.')
FRAME_FILENAME_PATTERN = 'frame_{:04d}.png'
FRAME_DPI = 180
FRAME_FIGSIZE = (18, 8)
FRAME_HISTORY_BARS = 220

# %% [code] cell 20
print({
    'symbol': SYMBOL,
    'lookback_days': LOOKBACK_DAYS,
    'horizon': HORIZON,
    'ensemble_size': ENSEMBLE_SIZE,
    'trend_lookback_bars': TREND_LOOKBACK_BARS,
    'strong_trend_threshold': STRONG_TREND_THRESHOLD,
    'sampling_temperature': SAMPLING_TEMPERATURE,
    # iTransformer config
    'd_model': D_MODEL,
    'n_heads': N_HEADS,
    'n_layers': N_LAYERS,
    'hidden_size': HIDDEN_SIZE,
    'num_layers': NUM_LAYERS,
    'device': str(DEVICE),
})

# %% [markdown] cell 21
## Phase 3: iTransformer Architecture Components

# %% [code] cell 22
class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time dimension."""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * 
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) if d_model % 2 == 0 else torch.cos(position * div_term[:-1])
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class iTransformerEncoderLayer(nn.Module):
    """
    iTransformer Encoder Layer with variable attention (across features, NOT time).
    Key innovation: [batch, time, features] -> [batch, features, time, d_model]
    """
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int = 512, 
                 dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args: src: [batch, num_features, time, d_model] - Inverted format!
        Returns: out: [batch, num_features, time, d_model]
        """
        b, n_var, t, d = src.shape
        src_reshaped = src.permute(0, 2, 1, 3).reshape(b * t, n_var, d)
        src2 = self.norm1(src_reshaped)
        attn_out, _ = self.self_attn(src2, src2, src2, 
                                      attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask,
                                      need_weights=False)
        src_reshaped = src_reshaped + self.dropout1(attn_out)
        src = src_reshaped.reshape(b, t, n_var, d).permute(0, 2, 1, 3)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class iTransformerEncoder(nn.Module):
    """
    iTransformer Encoder with input projection, positional encoding,
    stack of iTransformerEncoderLayer, and global average pooling.
    """
    def __init__(self, input_size: int, d_model: int = 128, n_heads: int = 8, 
                 n_layers: int = 2, dim_feedforward: int = 512, dropout: float = 0.1,
                 max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.input_size = input_size
        self.input_projection = nn.Linear(1, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            iTransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args: src: [batch, time, input_size]
        Returns: out: [batch, input_size * d_model]
        """
        batch_size, time_len, n_features = src.shape
        src = src.transpose(1, 2).unsqueeze(-1)  # [batch, n_features, time, 1]
        src = self.input_projection(src)  # [batch, n_features, time, d_model]
        src = src.reshape(batch_size * n_features, time_len, self.d_model)
        src = self.pos_encoder(src)
        src = src.reshape(batch_size, n_features, time_len, self.d_model)
        for layer in self.layers:
            src = layer(src)
        src = src.mean(dim=2)  # [batch, n_features, d_model]
        src = src.reshape(batch_size, -1)  # [batch, n_features * d_model]
        return src


class HybridEncoder(nn.Module):
    """Hybrid Encoder combining iTransformer + GRU."""
    def __init__(self, input_size: int, d_model: int = 128, n_heads: int = 8, 
                 n_layers: int = 2, hidden_size: int = 256, num_gru_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.itransformer = iTransformerEncoder(
            input_size=input_size, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, dropout=dropout
        )
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_gru_layers, batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0
        )
        fusion_input_size = input_size * d_model + hidden_size
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.output_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args: src: [batch, time, input_size]
        Returns: fused: [batch, hidden_size], gru_out: [batch, time, hidden_size]
        """
        itrans_out = self.itransformer(src)
        gru_out, gru_hidden = self.gru(src)
        gru_last = gru_hidden[-1]
        combined = torch.cat([itrans_out, gru_last], dim=-1)
        fused = self.fusion(combined)
        fused = self.output_norm(fused)
        return fused, gru_out


print('✓ Phase 3: iTransformer architecture components defined')
print(f'  - d_model: {D_MODEL}')
print(f'  - n_heads: {N_HEADS}')
print(f'  - n_layers: {N_LAYERS}')
print(f'  - hidden_size: {HIDDEN_SIZE}')

# %% [markdown] cell 23
## Model Definition: HybridSeq2Seq with iTransformer Encoder

# %% [code] cell 24
class HybridSeq2Seq(nn.Module):
    """
    Seq2Seq with Hybrid Encoder (iTransformer + GRU) and GRU Decoder.
    Replaces Seq2SeqAttnGRU encoder with HybridEncoder.
    """
    def __init__(self, input_size, output_size, d_model, n_heads, n_layers, 
                 hidden_size, num_gru_layers, dropout, horizon):
        super().__init__()
        self.horizon = horizon
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.d_model = d_model
        self.encoder = HybridEncoder(
            input_size=input_size, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, hidden_size=hidden_size,
            num_gru_layers=num_gru_layers, dropout=dropout
        )
        self.decoder_cell = nn.GRUCell(output_size + hidden_size, hidden_size)
        self.attn_proj = nn.Linear(hidden_size, hidden_size, bias=False)
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
        h_dec, enc_out = self.encoder(x)
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
            h_dec, enc_out = self.encoder(x)
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


print('✓ HybridSeq2Seq model defined with iTransformer encoder')

# %% [markdown] cell 25
## Test Cell: Validate iTransformer Architecture

# %% [code] cell 26
print('=' * 60)
print('TEST CELL: iTransformer Architecture Validation')
print('=' * 60)

test_batch_size = 4
test_seq_len = 60
test_input_size = 11
test_output_size = 4
test_horizon = 50

test_model = HybridSeq2Seq(
    input_size=test_input_size, output_size=test_output_size,
    d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
    hidden_size=HIDDEN_SIZE, num_gru_layers=NUM_LAYERS,
    dropout=DROPOUT, horizon=test_horizon,
).to(DEVICE)

test_input = torch.randn(test_batch_size, test_seq_len, test_input_size).to(DEVICE)
test_target = torch.randn(test_batch_size, test_horizon, test_output_size).to(DEVICE)

print('\n1. Forward Pass Test:')
mu_out = test_model(test_input)
assert mu_out.shape == (test_batch_size, test_horizon, test_output_size)
print(f'   ✓ Mu output shape: {mu_out.shape}')

mu_out, sigma_out = test_model(test_input, return_sigma=True)
assert sigma_out.shape == (test_batch_size, test_horizon, test_output_size)
print(f'   ✓ Sigma output shape: {sigma_out.shape}')

print('\n2. generate_realistic() Test:')
gen_out = test_model.generate_realistic(test_input, temperature=1.0)
assert gen_out.shape == (test_batch_size, test_horizon, test_output_size)
print(f'   ✓ Generated output shape: {gen_out.shape}')

gen_out1 = test_model.generate_realistic(test_input, temperature=1.0, manual_seed=42)
gen_out2 = test_model.generate_realistic(test_input, temperature=1.0, manual_seed=42)
assert torch.allclose(gen_out1, gen_out2)
print(f'   ✓ Manual seed reproducibility: OK')

print('\n3. Gradient Flow Test:')
test_model.train()
mu_out, sigma_out = test_model(test_input, y_teacher=test_target, teacher_forcing_ratio=0.5, return_sigma=True)
loss = F.mse_loss(mu_out, test_target) + torch.exp(sigma_out).mean()
loss.backward()

has_grads = [name for name, param in test_model.named_parameters() if param.grad is not None]
print(f'   ✓ Parameters with gradients: {len(has_grads)}')
itransformer_grads = [n for n in has_grads if 'itransformer' in n]
print(f'   ✓ iTransformer parameters with gradients: {len(itransformer_grads)}')

print('\n4. Architecture Component Test:')
test_layer = iTransformerEncoderLayer(d_model=D_MODEL, n_heads=N_HEADS).to(DEVICE)
test_layer_input = torch.randn(test_batch_size, test_input_size, test_seq_len, D_MODEL).to(DEVICE)
test_layer_output = test_layer(test_layer_input)
assert test_layer_output.shape == test_layer_input.shape
print(f'   ✓ iTransformerEncoderLayer output shape: {test_layer_output.shape}')

test_enc = iTransformerEncoder(input_size=test_input_size, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS).to(DEVICE)
test_enc_input = torch.randn(test_batch_size, test_seq_len, test_input_size).to(DEVICE)
test_enc_output = test_enc(test_enc_input)
expected_enc_out = test_input_size * D_MODEL
assert test_enc_output.shape == (test_batch_size, expected_enc_out)
print(f'   ✓ iTransformerEncoder output shape: {test_enc_output.shape}')

test_hybrid = HybridEncoder(
    input_size=test_input_size, d_model=D_MODEL, n_heads=N_HEADS,
    n_layers=N_LAYERS, hidden_size=HIDDEN_SIZE, num_gru_layers=NUM_LAYERS
).to(DEVICE)
test_hybrid_fused, test_hybrid_gru = test_hybrid(test_enc_input)
assert test_hybrid_fused.shape == (test_batch_size, HIDDEN_SIZE)
assert test_hybrid_gru.shape == (test_batch_size, test_seq_len, HIDDEN_SIZE)
print(f'   ✓ HybridEncoder fused output shape: {test_hybrid_fused.shape}')
print(f'   ✓ HybridEncoder GRU output shape: {test_hybrid_gru.shape}')

print('\n5. Parameter Count:')
total_params = sum(p.numel() for p in test_model.parameters())
print(f'   Total parameters: {total_params:,}')
itrans_params = sum(p.numel() for p in test_model.encoder.itransformer.parameters())
gru_enc_params = sum(p.numel() for p in test_model.encoder.gru.parameters())
print(f'   iTransformer encoder: {itrans_params:,}')
print(f'   GRU encoder: {gru_enc_params:,}')

print('\n' + '=' * 60)
print('ALL TESTS PASSED ✓')
print('=' * 60)

# %% [markdown] cell 27
## Data Fetching Functions

# %% [code] cell 28
class RequestPacer:
    def __init__(self, max_calls_per_minute: int):
        self.min_interval = 60.0 / float(max_calls_per_minute)
        self.last_call_ts = 0.0
    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_call_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call_ts = time.monotonic()

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
        raise ValueError(f'Unsupported ALPACA_FEED={feed_name!r}')
    return mapping[k]

def fetch_bars_alpaca(symbol: str, lookback_days: int) -> tuple[pd.DataFrame, int]:
    api_key, secret_key = _require_alpaca_credentials()
    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    feed = _resolve_feed(ALPACA_FEED)
    pacer = RequestPacer(MAX_REQUESTS_PER_MINUTE)
    end_ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    if ALPACA_FEED in {'sip', 'delayed_sip'}:
        end_ts = end_ts - timedelta(minutes=20)
    start_ts = end_ts - timedelta(days=lookback_days)
    parts, cursor, calls = [], start_ts, 0
    while cursor < end_ts:
        chunk_end = min(cursor + timedelta(days=REQUEST_CHUNK_DAYS), end_ts)
        chunk = None
        for attempt in range(1, MAX_RETRIES + 1):
            pacer.wait(); calls += 1
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=[symbol], timeframe=TimeFrame.Minute,
                    start=cursor, end=chunk_end, feed=feed, limit=10000)
                chunk = client.get_stock_bars(req).df
                break
            except Exception as exc:
                if attempt < MAX_RETRIES:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise
        if chunk is not None and not chunk.empty:
            d = chunk.reset_index().rename(columns={
                'timestamp': 'Datetime', 'open': 'Open', 'high': 'High',
                'low': 'Low', 'close': 'Close', 'volume': 'Volume',
                'trade_count': 'TradeCount', 'vwap': 'VWAP'})
            for col in ['Volume', 'TradeCount', 'VWAP']:
                if col not in d.columns:
                    d[col] = 0.0 if col != 'VWAP' else d['Close']
            d['Datetime'] = pd.to_datetime(d['Datetime'], utc=True)
            d = d[['Datetime'] + RAW_COLS].dropna(subset=OHLC_COLS).set_index('Datetime').sort_index()
            parts.append(d)
        cursor = chunk_end
    if not parts:
        raise RuntimeError('No bars returned from Alpaca.')
    out = pd.concat(parts).sort_index()
    return out[~out.index.duplicated(keep='last')].astype(np.float32), calls

# %% [code] cell 29
def sessionize_with_calendar(df_utc: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if df_utc.empty:
        raise RuntimeError('Input bars are empty.')
    idx = pd.DatetimeIndex(df_utc.index)
    idx = idx.tz_localize('UTC') if idx.tz is None else idx.tz_convert('UTC')
    df_utc = df_utc.copy(); df_utc.index = idx
    cal = mcal.get_calendar('XNYS')
    sched = cal.schedule(
        start_date=(idx.min() - pd.Timedelta(days=2)).date(),
        end_date=(idx.max() + pd.Timedelta(days=2)).date())
    pieces, fill_ratios = [], []
    for sid, (_, row) in enumerate(sched.iterrows()):
        open_ts = pd.Timestamp(row['market_open'])
        close_ts = pd.Timestamp(row['market_close'])
        open_ts = open_ts.tz_localize('UTC') if open_ts.tzinfo is None else open_ts.tz_convert('UTC')
        close_ts = close_ts.tz_localize('UTC') if close_ts.tzinfo is None else close_ts.tz_convert('UTC')
        exp_idx = pd.date_range(open_ts, close_ts, freq='1min', inclusive='left')
        if len(exp_idx) == 0: continue
        day = df_utc[(df_utc.index >= open_ts) & (df_utc.index < close_ts)].reindex(exp_idx)
        imputed = day[OHLC_COLS].isna().any(axis=1).to_numpy()
        fill_ratio = float(imputed.mean())
        if fill_ratio >= 1.0 or fill_ratio > MAX_SESSION_FILL_RATIO: continue
        day[OHLC_COLS + ['VWAP']] = day[OHLC_COLS + ['VWAP']].ffill().bfill()
        day['VWAP'] = day['VWAP'].fillna(day['Close'])
        day['Volume'] = day['Volume'].fillna(0.0)
        day['TradeCount'] = day['TradeCount'].fillna(0.0)
        day['is_imputed'] = imputed.astype(np.int8)
        day['session_id'] = int(sid)
        day['bar_in_session'] = np.arange(len(day), dtype=np.int32)
        day['session_len'] = int(len(day))
        pieces.append(day)
        fill_ratios.append(fill_ratio)
    if not pieces:
        raise RuntimeError('No sessions kept after calendar filtering.')
    out = pd.concat(pieces).sort_index()
    out.index = out.index.tz_convert(SESSION_TZ).tz_localize(None)
    for c in RAW_COLS: out[c] = out[c].astype(np.float32)
    out['is_imputed'] = out['is_imputed'].astype(np.int8)
    out['session_id'] = out['session_id'].astype(np.int32)
    out['bar_in_session'] = out['bar_in_session'].astype(np.int32)
    out['session_len'] = out['session_len'].astype(np.int32)
    meta = {'calendar_sessions_total': int(len(sched)), 'kept_sessions': int(len(pieces)),
            'avg_fill_ratio_kept': float(np.mean(fill_ratios)) if fill_ratios else float('nan')}
    return out, meta

# %% [markdown] cell 30
## Feature Engineering Functions

# %% [code] cell 31
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
        seq.append(cand); prev_close = float(cand[3])
    return np.asarray(seq, dtype=np.float32)

def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9
    g = df.groupby('session_id', sort=False)
    prev_close = g['Close'].shift(1).fillna(df['Open'])
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

# %% [markdown] cell 32
## Windowing & Dataset Functions

# %% [code] cell 33
def split_points(n_rows: int) -> tuple[int, int]:
    tr = int(n_rows * TRAIN_RATIO)
    va = int(n_rows * (TRAIN_RATIO + VAL_RATIO))
    return tr, va

def build_walkforward_slices(price_df_full: pd.DataFrame) -> list[tuple[str, int, int]]:
    n = len(price_df_full)
    span = int(round(n * 0.85))
    shift = max(1, n - span)
    cands = [('slice_1', 0, min(span, n)), ('slice_2', shift, min(shift + span, n))]
    out, seen = [], set()
    for name, a, b in cands:
        key = (a, b)
        if key in seen or b - a < max(LOOKBACK_CANDIDATES) + HORIZON + 1400: continue
        out.append((name, a, b)); seen.add(key)
    return out if out else [('full', 0, n)]

def make_multistep_windows(input_scaled, target_scaled, target_raw, row_imputed, row_open_skip, 
                           starts_prev_close, window, horizon):
    X, y_s, y_r, starts, prev_close = [], [], [], [], []
    dropped_target_imputed, dropped_target_open_skip = 0, 0
    n = len(input_scaled)
    for i in range(window, n - horizon + 1):
        if row_imputed[i:i+horizon].any(): dropped_target_imputed += 1; continue
        if row_open_skip[i:i+horizon].any(): dropped_target_open_skip += 1; continue
        xb = input_scaled[i-window:i]
        imp_frac = float(row_imputed[i-window:i].mean())
        imp_col = np.full((window, 1), imp_frac, dtype=np.float32)
        xb_aug = np.concatenate([xb, imp_col], axis=1)
        X.append(xb_aug); y_s.append(target_scaled[i:i+horizon])
        y_r.append(target_raw[i:i+horizon]); starts.append(i)
        prev_close.append(starts_prev_close[i])
    return (np.asarray(X, dtype=np.float32), np.asarray(y_s, dtype=np.float32),
            np.asarray(y_r, dtype=np.float32), np.asarray(starts, dtype=np.int64),
            np.asarray(prev_close, dtype=np.float32), dropped_target_imputed, dropped_target_open_skip)

class MultiStepDataset(Dataset):
    def __init__(self, X, y_s, y_r):
        self.X = torch.from_numpy(X).float()
        self.y_s = torch.from_numpy(y_s).float()
        self.y_r = torch.from_numpy(y_r).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y_s[idx], self.y_r[idx]

# %% [markdown] cell 34
## Trend Injection Functions (v7.5)

# %% [code] cell 35
def calculate_trend_slope(prices: np.ndarray) -> float:
    prices = np.asarray(prices, dtype=np.float32)
    if len(prices) < 2: return 0.0
    log_prices = np.log(prices)
    x = np.arange(len(log_prices), dtype=np.float32)
    x_mean, y_mean = np.mean(x), np.mean(log_prices)
    numerator = np.sum((x - x_mean) * (log_prices - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    if denominator < 1e-10: return 0.0
    return float(numerator / denominator)

def calculate_path_trend(return_ohlc: np.ndarray) -> float:
    close_returns = return_ohlc[:, 3]
    close_prices = np.exp(np.cumsum(close_returns))
    return calculate_trend_slope(close_prices)

def select_best_path_by_trend(all_paths: list[np.ndarray], historical_slope: float, 
                              strong_trend_threshold: float = 0.002) -> tuple[int, np.ndarray, dict]:
    path_slopes = [calculate_path_trend(path) for path in all_paths]
    path_directions = [np.sign(s) for s in path_slopes]
    path_slopes = np.array(path_slopes)
    path_directions = np.array(path_directions)
    historical_direction = np.sign(historical_slope)
    is_strong_trend = abs(historical_slope) > strong_trend_threshold
    if is_strong_trend:
        valid_mask = path_directions == historical_direction
        if not valid_mask.any(): valid_mask = np.ones(len(all_paths), dtype=bool)
    else:
        valid_mask = np.ones(len(all_paths), dtype=bool)
    slope_distances = np.abs(path_slopes - historical_slope)
    slope_distances[~valid_mask] = np.inf
    best_idx = int(np.argmin(slope_distances))
    best_path = all_paths[best_idx]
    info = {'historical_slope': historical_slope, 'is_strong_trend': is_strong_trend,
            'selected_idx': best_idx, 'selected_slope': path_slopes[best_idx],
            'rejected_count': int((~valid_mask).sum())}
    return best_idx, best_path, info

def generate_ensemble_with_trend_selection(model, X: np.ndarray, context_prices: np.ndarray,
                                           temperature: float = 1.0, ensemble_size: int = 20,
                                           trend_lookback: int = 20) -> tuple[np.ndarray, dict]:
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(DEVICE)
        log_returns = np.log(context_prices[1:] / context_prices[:-1])
        historical_vol = float(np.std(log_returns)) if len(log_returns) > 1 else 0.001
        historical_slope = calculate_trend_slope(context_prices[-trend_lookback:])
        print(f'Historical vol: {historical_vol:.6f}, Temperature: {temperature}')
        print(f'Historical trend slope: {historical_slope:.6f}')
        all_paths = []
        for i in range(ensemble_size):
            seed = SEED + i * 1000
            generated = model.generate_realistic(X_tensor, temperature=temperature,
                                                 historical_vol=historical_vol, manual_seed=seed)
            all_paths.append(generated.detach().cpu().numpy()[0])
        best_idx, best_path, selection_info = select_best_path_by_trend(all_paths, historical_slope, STRONG_TREND_THRESHOLD)
        print(f'Generated {ensemble_size} paths, selected path {best_idx}')
        ensemble_info = {'all_paths': all_paths, 'best_idx': best_idx, 'best_path': best_path,
                        'selection_info': selection_info, 'historical_vol': historical_vol}
        return best_path, ensemble_info

# %% [markdown] cell 36
## Loss Functions

# %% [code] cell 37
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

# %% [markdown] cell 38
## Training Functions

# %% [code] cell 39
def tf_ratio_for_epoch(epoch):
    ratio = TF_START * (TF_DECAY_RATE ** (epoch - 1))
    return max(float(TF_END), float(ratio))

def run_epoch(model, loader, step_weights_t, optimizer=None, tf_ratio=0.0):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, nll_total, range_total, vol_total, dir_total = 0, 0, 0, 0, 0
    n_items = 0
    for xb, yb_s, yb_r in loader:
        xb, yb_s = xb.to(DEVICE), yb_s.to(DEVICE)
        if is_train: optimizer.zero_grad(set_to_none=True)
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
        nll_total += nll.item() * bs; range_total += rng.item() * bs
        vol_total += vol.item() * bs; dir_total += dir_pen.item() * bs
        n_items += bs
    return {'total': total_loss / max(n_items, 1), 'nll': nll_total / max(n_items, 1),
            'range': range_total / max(n_items, 1), 'vol': vol_total / max(n_items, 1),
            'dir': dir_total / max(n_items, 1)}

def train_model(model, train_loader, val_loader, max_epochs, patience):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    step_idx = np.arange(HORIZON, dtype=np.float32)
    step_w = 1.0 + (step_idx / max(HORIZON - 1, 1)) ** STEP_LOSS_POWER
    step_weights_t = torch.as_tensor(step_w, dtype=torch.float32, device=DEVICE).view(1, HORIZON, 1)
    best_val, wait, rows = float('inf'), 0, []
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, max_epochs + 1):
        tf = tf_ratio_for_epoch(epoch)
        tr = run_epoch(model, train_loader, step_weights_t, optimizer=optimizer, tf_ratio=tf)
        va = run_epoch(model, val_loader, step_weights_t, optimizer=None, tf_ratio=0.0)
        scheduler.step(va['total'])
        lr = optimizer.param_groups[0]['lr']
        rows.append({'epoch': epoch, 'tf_ratio': tf, 'lr': lr,
                     'train_total': tr['total'], 'val_total': va['total']})
        print(f'Epoch {epoch:02d} | tf={tf:.3f} | train={tr["total"]:.6f} | val={va["total"]:.6f} | lr={lr:.6g}')
        if va['total'] < best_val:
            best_val = va['total']; best_state = copy.deepcopy(model.state_dict()); wait = 0
        else:
            wait += 1
            if wait >= patience: print(f'Early stopping at epoch {epoch}.'); break
    model.load_state_dict(best_state)
    return pd.DataFrame(rows)

# %% [markdown] cell 40
## Evaluation Functions

# %% [code] cell 41
def evaluate_metrics(actual_ohlc, pred_ohlc, prev_close):
    actual_ohlc = np.asarray(actual_ohlc, dtype=np.float32)
    pred_ohlc = np.asarray(pred_ohlc, dtype=np.float32)
    ac, pc = actual_ohlc[:, 3], pred_ohlc[:, 3]
    return {'close_mae': float(np.mean(np.abs(ac - pc))),
            'close_rmse': float(np.sqrt(np.mean((ac - pc) ** 2))),
            'ohlc_mae': float(np.mean(np.abs(actual_ohlc - pred_ohlc))),
            'directional_accuracy_eps': float(np.mean(np.sign(ac - prev_close) == np.sign(pc - prev_close)))}

def evaluate_baselines(actual_ohlc, prev_ohlc, prev_close):
    persistence = evaluate_metrics(actual_ohlc, prev_ohlc, prev_close)
    flat = np.repeat(prev_close.reshape(-1, 1), 4, axis=1).astype(np.float32)
    flat_rw = evaluate_metrics(actual_ohlc, flat, prev_close)
    return {'persistence': persistence, 'flat_close_rw': flat_rw}

# %% [markdown] cell 42
## Main Training Function

# %% [code] cell 43
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
    in_mean = input_raw[:tr_end].mean(axis=0); in_std = input_raw[:tr_end].std(axis=0)
    in_std = np.where(in_std < 1e-8, 1.0, in_std)
    input_scaled = (input_raw - in_mean) / in_std
    target_scaled = target_raw.copy()
    X_all, y_all_s, y_all_r, starts, prev_close_starts, _, _ = make_multistep_windows(
        input_scaled, target_scaled, target_raw, row_imputed, row_open_skip, prev_close, window, HORIZON)
    if len(X_all) == 0: raise RuntimeError(f'{fold_name}: no windows available.')
    end_idx = starts + HORIZON - 1
    tr_m = end_idx < tr_end; va_m = (end_idx >= tr_end) & (end_idx < va_end); te_m = end_idx >= va_end
    X_train, y_train_s, y_train_r = X_all[tr_m], y_all_s[tr_m], y_all_r[tr_m]
    X_val, y_val_s, y_val_r = X_all[va_m], y_all_s[va_m], y_all_r[va_m]
    X_test, y_test_s, y_test_r = X_all[te_m], y_all_s[te_m], y_all_r[te_m]
    test_starts, test_prev_close = starts[te_m], prev_close_starts[te_m]
    print(f'Samples: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}')
    train_loader = DataLoader(MultiStepDataset(X_train, y_train_s, y_train_r), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MultiStepDataset(X_val, y_val_s, y_val_r), batch_size=BATCH_SIZE, shuffle=False)
    # Phase 3: Use HybridSeq2Seq with iTransformer encoder
    model = HybridSeq2Seq(
        input_size=X_train.shape[-1], output_size=len(TARGET_COLS),
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        hidden_size=HIDDEN_SIZE, num_gru_layers=NUM_LAYERS,
        dropout=DROPOUT, horizon=HORIZON).to(DEVICE)
    hist = train_model(model, train_loader, val_loader, max_epochs, patience)
    # Ensemble trend injection
    last_idx = len(X_test) - 1
    X_last = X_test[last_idx:last_idx+1]
    context_start = int(test_starts[last_idx]) - window
    context_prices = price_vals[context_start:int(test_starts[last_idx]), 3]
    best_rets, ensemble_info = generate_ensemble_with_trend_selection(
        model, X_last, context_prices, temperature=SAMPLING_TEMPERATURE,
        ensemble_size=ENSEMBLE_SIZE, trend_lookback=TREND_LOOKBACK_BARS)
    last_close = float(test_prev_close[last_idx])
    pred_price_best = returns_to_prices_seq(best_rets, last_close)
    all_price_paths = [returns_to_prices_seq(path_rets, last_close) for path_rets in ensemble_info['all_paths']]
    ensemble_info['all_price_paths'] = all_price_paths
    actual_future = price_vals[int(test_starts[last_idx]):int(test_starts[last_idx])+HORIZON]
    mu_test = model(torch.from_numpy(X_test).float().to(DEVICE)).detach().cpu().numpy()
    pred_step1_ret = mu_test[:, 0, :]
    pred_ohlc_1 = np.zeros((len(test_starts), 4))
    for i in range(len(test_starts)):
        pc = test_prev_close[i]
        pred_ohlc_1[i] = [pc * np.exp(pred_step1_ret[i, j]) for j in range(4)]
        pred_ohlc_1[i] = enforce_candle_validity(pred_ohlc_1[i].reshape(1, -1))[0]
    actual_ohlc_1 = price_vals[test_starts + 1]
    model_metrics = evaluate_metrics(actual_ohlc_1, pred_ohlc_1, test_prev_close)
    baseline_metrics = evaluate_baselines(actual_ohlc_1, price_vals[test_starts], test_prev_close)
    print(f"\nPred range: [{pred_price_best[:, 3].min():.2f}, {pred_price_best[:, 3].max():.2f}]")
    future_idx = price_fold.index[test_starts[last_idx]:test_starts[last_idx]+HORIZON]
    return {'fold': fold_name, 'window': window, 'history_df': hist,
            'model_metrics': model_metrics, 'baseline_metrics': baseline_metrics,
            'context_df': price_fold.iloc[test_starts[last_idx]-window:test_starts[last_idx]+1][OHLC_COLS],
            'actual_future_df': pd.DataFrame(actual_future, index=future_idx, columns=OHLC_COLS),
            'pred_future_df': pd.DataFrame(pred_price_best, index=future_idx, columns=OHLC_COLS),
            'ensemble_info': ensemble_info}

# %% [markdown] cell 44
## V8 Extension: Rolling Walk-Forward Backtest

# %% [code] cell 45
@dataclass
class RollingPredictionLog:
    anchortime: pd.Timestamp
    contextendprice: float
    predictedpath: pd.DataFrame
    actualpath: pd.DataFrame
    predictionhorizon: int
    temperature: float
    context_start_idx: int
    context_end_idx: int
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
    def __init__(self, model: nn.Module, pricedf: pd.DataFrame, featuredf: pd.DataFrame,
                 input_mean: np.ndarray, input_std: np.ndarray, windowsize: int, horizon: int):
        self.model = model.to(DEVICE); self.model.eval()
        self.pricedf = pricedf.copy(); self.featuredf = featuredf.copy()
        self.input_mean = input_mean.astype(np.float32)
        self.input_std = np.where(input_std.astype(np.float32) < 1e-8, 1.0, input_std.astype(np.float32))
        self.windowsize = int(windowsize); self.horizon = int(horizon)
        self.input_raw = self.featuredf[BASE_FEATURE_COLS].to_numpy(np.float32)
        self.input_scaled = ((self.input_raw - self.input_mean) / self.input_std).astype(np.float32)
        self.row_imputed = self.featuredf['row_imputed'].to_numpy(np.int8).astype(bool)
        self.ts_to_pos = {ts: i for i, ts in enumerate(self.featuredf.index)}

    def _temperature_for_time(self, ts: pd.Timestamp) -> float:
        if not USE_TEMPERATURE_SCHEDULE: return float(DEFAULT_ROLLING_TEMPERATURE)
        t = ts.time()
        for st_s, en_s, temp in TEMPERATURESCHEDULE:
            st = pd.Timestamp(st_s).time(); en = pd.Timestamp(en_s).time()
            if st <= t < en: return float(temp)
        return float(DEFAULT_ROLLING_TEMPERATURE)

    def _hist_vol(self, context_start: int, context_end: int) -> float:
        closes = self.pricedf['Close'].iloc[context_start:context_end].to_numpy(np.float32)
        if len(closes) < 2: return 0.001
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

    def run_rolling_backtest(self, starttime: str, endtime: str, date: str, step: int = 1) -> Tuple[List[RollingPredictionLog], int]:
        idx = self.featuredf.index
        st = pd.Timestamp(starttime).time(); en = pd.Timestamp(endtime).time()
        mask = (idx.strftime('%Y-%m-%d') == date) & np.array([(t >= st) and (t < en) for t in idx.time], dtype=bool)
        anchor_positions = np.where(mask)[0]
        if len(anchor_positions) == 0: raise RuntimeError(f'No anchors for date={date}')
        selected = anchor_positions[::step]
        logs: List[RollingPredictionLog] = []
        pbar = tqdm(total=len(selected), desc=f'Processing minute 0/{len(selected)}')
        for k, anchor_pos in enumerate(selected, start=1):
            context_end = int(anchor_pos); context_start = context_end - self.windowsize
            if context_start < 0: continue
            day_idx = np.searchsorted(anchor_positions, anchor_pos)
            valid_steps = min(self.horizon, len(anchor_positions) - day_idx)
            if valid_steps <= 0: continue
            prediction_time = idx[context_end]
            context = self.featuredf.iloc[context_start:context_end]
            assert context.index[-1] < prediction_time
            temp = self._temperature_for_time(prediction_time)
            pred_rets_full = self._predict_path(context_start, context_end, temp)
            pred_rets = pred_rets_full[:valid_steps]
            context_close = float(self.featuredf['prev_close'].iloc[context_end])
            pred_prices = returns_to_prices_seq(pred_rets, context_close)
            future_positions = anchor_positions[day_idx: day_idx + valid_steps]
            pred_index = idx[future_positions]
            pred_df = pd.DataFrame(pred_prices, index=pred_index, columns=OHLC_COLS)
            actual_df = self.pricedf[OHLC_COLS].iloc[future_positions].copy()
            log = RollingPredictionLog(
                anchortime=prediction_time, contextendprice=context_close,
                predictedpath=pred_df, actualpath=actual_df,
                predictionhorizon=valid_steps, temperature=temp,
                context_start_idx=context_start, context_end_idx=context_end).compute_metrics()
            logs.append(log)
            pbar.set_description(f'Processing minute {k}/{len(selected)}'); pbar.update(1)
        pbar.close()
        return logs, len(selected)


def run_rolling_backtest(model, pricedf, windowsize, starttime, endtime, backtest_date):
    rb = RollingBacktester(model=model, pricedf=pricedf, featuredf=rolling_feat_df,
                           input_mean=rolling_in_mean, input_std=rolling_in_std,
                           windowsize=windowsize, horizon=HORIZON)
    return rb.run_rolling_backtest(starttime=starttime, endtime=endtime, date=backtest_date, step=ROLLING_STEP), rb

# %% [code] cell 46
def _intraday_positions_for_date(df: pd.DataFrame, date_str: str, start_time: str, end_time: str) -> np.ndarray:
    idx = df.index
    day_mask = (idx.strftime('%Y-%m-%d') == date_str)
    st = pd.Timestamp(start_time).time(); et = pd.Timestamp(end_time).time()
    time_mask = np.array([(t >= st) and (t < et) for t in idx.time], dtype=bool)
    return np.where(day_mask & time_mask)[0]

def _select_backtest_date(df: pd.DataFrame, requested: Optional[str], window: int, horizon: int) -> str:
    if requested is not None:
        pos = _intraday_positions_for_date(df, requested, ROLLINGSTARTTIME, ROLLINGENDTIME)
        if len(pos) == 0: raise ValueError(f'No intraday bars for requested date: {requested}')
        if pos[0] < window: raise ValueError(f'Not enough prior bars for window={window}')
        if pos[-1] + horizon >= len(df): raise ValueError(f'Lacks future bars for horizon={horizon}')
        return requested
    dates = sorted(pd.Index(df.index.strftime('%Y-%m-%d')).unique())
    if len(dates) < 2: raise RuntimeError('Need at least 2 trading dates')
    for d in reversed(dates[:-1]):
        pos = _intraday_positions_for_date(df, d, ROLLINGSTARTTIME, ROLLINGENDTIME)
        if len(pos) < 390 or pos[0] < window or pos[-1] + horizon >= len(df): continue
        return d
    raise RuntimeError('Could not auto-select valid backtest date')

def train_v9_model_for_rolling(price_df_full: pd.DataFrame, window: int, horizon: int, backtest_date: str):
    feat_all = build_feature_frame(price_df_full)
    target_all = build_target_frame(feat_all)
    all_dates = feat_all.index.strftime('%Y-%m-%d')
    train_day_mask = all_dates < backtest_date
    if train_day_mask.sum() < (window + horizon + 500):
        raise RuntimeError(f'Not enough pre-backtest bars: {train_day_mask.sum()}')
    input_raw = feat_all[BASE_FEATURE_COLS].to_numpy(np.float32)
    target_raw = target_all[TARGET_COLS].to_numpy(np.float32)
    row_imputed = feat_all['row_imputed'].to_numpy(np.int8).astype(bool)
    row_open_skip = feat_all['row_open_skip'].to_numpy(np.int8).astype(bool)
    prev_close = feat_all['prev_close'].to_numpy(np.float32)
    pre_idx = np.where(train_day_mask)[0]; fit_end = int(pre_idx[-1]) + 1
    in_mean = input_raw[:fit_end].mean(axis=0); in_std = input_raw[:fit_end].std(axis=0)
    in_std = np.where(in_std < 1e-8, 1.0, in_std)
    input_scaled = ((input_raw - in_mean) / in_std).astype(np.float32)
    target_scaled = target_raw.copy()
    X_all, y_all_s, y_all_r, starts, _, drop_imp, drop_skip = make_multistep_windows(
        input_scaled, target_scaled, target_raw, row_imputed, row_open_skip, prev_close, window, horizon)
    if len(X_all) == 0: raise RuntimeError('No windows available')
    end_idx = starts + horizon - 1; end_dates = feat_all.index[end_idx].strftime('%Y-%m-%d')
    usable = end_dates < backtest_date; X_all = X_all[usable]; y_all_s = y_all_s[usable]; y_all_r = y_all_r[usable]
    if len(X_all) < 1000: raise RuntimeError(f'Not enough usable windows: {len(X_all)}')
    split = int(len(X_all) * 0.85)
    X_train, y_train_s, y_train_r = X_all[:split], y_all_s[:split], y_all_r[:split]
    X_val, y_val_s, y_val_r = X_all[split:], y_all_s[split:], y_all_r[split:]
    train_loader = DataLoader(MultiStepDataset(X_train, y_train_s, y_train_r), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MultiStepDataset(X_val, y_val_s, y_val_r), batch_size=BATCH_SIZE, shuffle=False)
    model = HybridSeq2Seq(
        input_size=X_train.shape[-1], output_size=len(TARGET_COLS),
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        hidden_size=HIDDEN_SIZE, num_gru_layers=NUM_LAYERS,
        dropout=DROPOUT, horizon=horizon).to(DEVICE)
    print({'train_windows': len(X_train), 'val_windows': len(X_val),
           'dropped_target_imputed': int(drop_imp), 'dropped_target_open_skip': int(drop_skip)})
    history_df = train_model(model, train_loader, val_loader, max_epochs=FINAL_MAX_EPOCHS, patience=FINAL_PATIENCE)
    return model, feat_all, in_mean.astype(np.float32), in_std.astype(np.float32), history_df

# %% [markdown] cell 47
## Visualization Functions

# %% [code] cell 48
def _draw_candles(ax, ohlc_df: pd.DataFrame, start_x: int,
                  up_edge: str, up_face: str, down_edge: str, down_face: str,
                  wick_color: str, width: float = 0.58, lw: float = 1.0, alpha: float = 1.0):
    vals = ohlc_df[OHLC_COLS].to_numpy(np.float32)
    for i, (o, h, l, c) in enumerate(vals):
        x = start_x + i; bull = c >= o
        ax.vlines(x, l, h, color=wick_color, linewidth=lw, alpha=alpha, zorder=2)
        lower = min(o, c); height = max(abs(c - o), 1e-6)
        rect = Rectangle((x - width / 2, lower), width, height,
                         facecolor=up_face if bull else down_face,
                         edgecolor=up_edge if bull else down_edge,
                         linewidth=lw, alpha=alpha, zorder=3)
        ax.add_patch(rect)

def _format_ts(ts: pd.Timestamp) -> str: return ts.strftime('%I:%M %p').lstrip('0')

def render_single_frame(log: RollingPredictionLog, frame_idx: int, total_frames: int,
                       pricedf: pd.DataFrame, backtester: RollingBacktester,
                       history_bars: int = FRAME_HISTORY_BARS) -> plt.Figure:
    anchor_pos = backtester.ts_to_pos[log.anchortime]
    h_start = max(0, anchor_pos - history_bars)
    history_df = pricedf.iloc[h_start:anchor_pos][OHLC_COLS].copy()
    actual_df = log.actualpath.copy(); pred_df = log.predictedpath.copy()
    fig, ax = plt.subplots(figsize=FRAME_FIGSIZE, facecolor='black')
    FigureCanvasAgg(fig); ax.set_facecolor('black')
    _draw_candles(ax, history_df, 0, up_edge='#00FF00', up_face='#00FF00',
                  down_edge='#FF0000', down_face='#FF0000',
                  wick_color='#D0D0D0', width=0.60, lw=1.0, alpha=0.95)
    future_start_x = len(history_df)
    _draw_candles(ax, actual_df, future_start_x, up_edge='#1D6F42', up_face='#1D6F42',
                  down_edge='#8E2F25', down_face='#8E2F25',
                  wick_color='#8E8E8E', width=0.58, lw=0.9, alpha=0.40)
    _draw_candles(ax, pred_df, future_start_x, up_edge='#FFFFFF', up_face='#FFFFFF',
                  down_edge='#FFFFFF', down_face='#000000',
                  wick_color='#F3F3F3', width=0.50, lw=1.2, alpha=1.0)
    now_x = len(history_df) - 0.5
    ax.axvline(now_x, color='white', linestyle='--', linewidth=1.0, alpha=0.85, zorder=4)
    ax.text(now_x + 0.8, ax.get_ylim()[1], 'NOW', color='white', fontsize=9)
    full_idx = history_df.index.append(actual_df.index); n = len(full_idx)
    step = max(1, n // 10); ticks = list(range(0, n, step))
    if ticks[-1] != n - 1: ticks.append(n - 1)
    labels = [full_idx[i].strftime('%H:%M') for i in ticks]
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=25, ha='right', color='white', fontsize=9)
    ax.tick_params(axis='y', colors='white')
    for sp in ax.spines.values(): sp.set_color('#666666')
    ax.grid(color='#242424', linewidth=0.6, alpha=0.35)
    header = f'{SYMBOL} 1m | Timestamp: {_format_ts(log.anchortime)} | Frame {frame_idx + 1}/{total_frames} | Temp={log.temperature:.2f}'
    ax.set_title(header, color='white', pad=12); ax.set_ylabel('Price', color='white')
    ax.text(0.01, 0.99, f'Frame {frame_idx + 1}/{total_frames}', transform=ax.transAxes,
            va='top', ha='left', color='white', fontsize=10,
            bbox=dict(facecolor='black', edgecolor='#666666', alpha=0.8, boxstyle='round,pad=0.25'))
    legend_elements = [
        Patch(facecolor='#00FF00', edgecolor='#00FF00', label='History (bull)'),
        Patch(facecolor='#FF0000', edgecolor='#FF0000', label='History (bear)'),
        Patch(facecolor='#1D6F42', edgecolor='#1D6F42', label='Actual Future (dim)'),
        Patch(facecolor='#8E2F25', edgecolor='#8E2F25', label='Actual Future (dim bear)'),
        Patch(facecolor='#FFFFFF', edgecolor='#FFFFFF', label='Predicted (bull)'),
        Patch(facecolor='#000000', edgecolor='#FFFFFF', label='Predicted (bear)')]
    leg = ax.legend(handles=legend_elements, facecolor='black', edgecolor='#666666', loc='upper left')
    for t in leg.get_texts(): t.set_color('white')
    plt.tight_layout(); return fig

def generate_rolling_frames(logs: List[RollingPredictionLog], pricedf: pd.DataFrame,
                           backtester: RollingBacktester, output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True); total = len(logs); saved_paths: List[Path] = []
    pbar = tqdm(total=total, desc=f'Saving frame 0/{total}')
    for i, log in enumerate(logs):
        fig = render_single_frame(log, i, total, pricedf, backtester, FRAME_HISTORY_BARS)
        out_path = output_dir / FRAME_FILENAME_PATTERN.format(i)
        fig.savefig(out_path, dpi=FRAME_DPI, facecolor='black', bbox_inches='tight')
        saved_paths.append(out_path); plt.close('all')
        pbar.set_description(f'Saving frame {i + 1}/{total}'); pbar.update(1)
    pbar.close(); return saved_paths

# %% [markdown] cell 49
## Main Execution

# %% [code] cell 50
# Fetch data
raw_df_utc, api_calls = fetch_bars_alpaca(SYMBOL, LOOKBACK_DAYS)
price_df, session_meta = sessionize_with_calendar(raw_df_utc)
print(f'Raw rows: {len(raw_df_utc):,}')
print(f'Sessionized rows: {len(price_df):,}')
print(f'Session meta: {session_meta}')

min_needed = max(LOOKBACK_CANDIDATES) + HORIZON + 1000
if len(price_df) < min_needed:
    raise RuntimeError(f'Not enough rows: {len(price_df)}, need {min_needed}')

# %% [code] cell 51
# Build features
feat_df = build_feature_frame(price_df)
target_df = build_target_frame(feat_df)
print(f'Feature rows: {len(feat_df)}')
print(f'Target columns: {list(target_df.columns)}')

# %% [code] cell 52
# Build walk-forward slices
slices = build_walkforward_slices(price_df)
print('Walk-forward slices:', slices)

# %% [code] cell 53
# Lookback sweep
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
                best_score = score; selected_window = w
        except Exception as e:
            print(f'Failed for window {w}: {e}')
print(f'\nSelected lookback: {selected_window}')

# %% [code] cell 54
# Full walk-forward training
print('\n=== Full walk-forward with iTransformer ===')
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
        print(f'Error in fold {name}: {e}')

# %% [markdown] cell 55
## Rolling Backtest Execution

# %% [code] cell 56
# Train model for rolling backtest
ROLLING_BACKTEST_DATE = _select_backtest_date(price_df, ROLLING_BACKTEST_DATE, DEFAULT_LOOKBACK, HORIZON)
print(f'Selected rolling backtest date: {ROLLING_BACKTEST_DATE}')

rolling_model, rolling_feat_df, rolling_in_mean, rolling_in_std, rolling_train_history = train_v9_model_for_rolling(
    price_df_full=price_df, window=DEFAULT_LOOKBACK, horizon=HORIZON, backtest_date=ROLLING_BACKTEST_DATE)
print('Rolling model trained and ready.')

# %% [code] cell 57
# Run rolling backtest
(rolling_logs, expected_prediction_count), rolling_backtester = run_rolling_backtest(
    model=rolling_model, pricedf=price_df, windowsize=DEFAULT_LOOKBACK,
    starttime=ROLLINGSTARTTIME, endtime=ROLLINGENDTIME, backtest_date=ROLLING_BACKTEST_DATE)

if len(rolling_logs) == 0: raise RuntimeError('No rolling logs produced.')

# Validation asserts
assert rolling_logs[0].predictedpath.index[0] == rolling_logs[0].anchortime
for log in rolling_logs:
    anchor_pos = rolling_backtester.ts_to_pos[log.anchortime]
    assert log.context_end_idx == anchor_pos
assert len(rolling_logs) == expected_prediction_count

print({'rolling_date': ROLLING_BACKTEST_DATE,
       'predictions_generated': len(rolling_logs),
       'expected_prediction_count': expected_prediction_count})

# %% [code] cell 58
# Generate rolling frames
saved_frame_paths = generate_rolling_frames(rolling_logs, price_df, rolling_backtester, FRAME_OUTPUT_DIR)
print({'frames_dir': str(FRAME_OUTPUT_DIR.resolve()),
       'frames_saved': len(saved_frame_paths),
       'first_frame': saved_frame_paths[0].name if saved_frame_paths else None})

# Rolling metrics
hit_rate = float(np.mean([lg.directional_hit for lg in rolling_logs if lg.directional_hit is not None]))
def _mae_at(step):
    vals = [float(lg.step_mae[step - 1]) for lg in rolling_logs if lg.step_mae is not None and len(lg.step_mae) >= step]
    return float(np.mean(vals)) if vals else float('nan')
metrics_table = pd.DataFrame([
    ('Directional hit rate (t+1)', f'{hit_rate:.2%}'),
    ('Path MAE @ step 1', f'${_mae_at(1):.4f}'),
    ('Path MAE @ step 5', f'${_mae_at(5):.4f}'),
    ('Path MAE @ step 10', f'${_mae_at(10):.4f}'),
    ('Path MAE @ step 15', f'${_mae_at(15):.4f}'),
], columns=['Metric', 'Value'])
display(metrics_table)

# %% [markdown] cell 59
### Validation Checklist
- iTransformerEncoderLayer variable attention across features: **implemented**
- iTransformerEncoder with positional encoding and pooling: **implemented**
- HybridEncoder combining iTransformer + GRU: **implemented**
- HybridSeq2Seq with GRU decoder and generate_realistic(): **implemented**
- Rolling backtest with strict causality: **implemented**
- Ensemble trend injection: **preserved from v7.5**
- Output shape validation test: **included**
- Gradient flow through attention: **validated**

