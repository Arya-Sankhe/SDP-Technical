from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import build_v9_notebooks as base


V92 = base.load_notebook(base.OUTPUT_DIR / "v9.2.ipynb")


def code_src(nb: dict, needle: str) -> str:
    idx = base.find_cell_index(nb, needle, cell_type="code")
    return base.cell_text(nb["cells"][idx])


def markdown_src(nb: dict, needle: str) -> str:
    idx = base.find_cell_index(nb, needle, cell_type="markdown")
    return base.cell_text(nb["cells"][idx])


V10_FEATURE_ENGINEERING_SRC = f"""
def encode_regime_state(turbulence: pd.Series, atr_pct: pd.Series, lookback: int) -> pd.Series:
    labels = np.ones(len(turbulence), dtype=np.float32)
    turb_values = turbulence.fillna(0.0).to_numpy(np.float32)
    atr_values = atr_pct.fillna(0.0).to_numpy(np.float32)

    for i in range(len(labels)):
        start = max(0, i - lookback + 1)
        turb_window = turb_values[start : i + 1]
        atr_window = atr_values[start : i + 1]
        if len(turb_window) < lookback:
            labels[i] = 1.0
            continue
        turb_pct = float(np.mean(turb_window <= turb_window[-1]))
        atr_pct_rank = float(np.mean(atr_window <= atr_window[-1]))
        if turb_pct > REGIME_CONFIG.elevated_threshold and atr_pct_rank > REGIME_CONFIG.elevated_threshold:
            labels[i] = -1.0
        elif turb_pct > REGIME_CONFIG.normal_threshold or atr_pct_rank > REGIME_CONFIG.normal_threshold:
            labels[i] = 0.0
        else:
            labels[i] = 1.0
    return pd.Series(labels, index=turbulence.index, dtype=np.float32)


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

    out['returns'] = out['rClose']
    out['turbulence_60'] = calculate_historical_turbulence(
        pd.DataFrame({{'returns': out['returns']}}, index=df.index),
        lookback=REGIME_CONFIG.lookback,
    )
    out['regime_indicator'] = encode_regime_state(
        turbulence=out['turbulence_60'],
        atr_pct=out['atr_14_pct'],
        lookback=REGIME_CONFIG.lookback,
    )

    out['row_imputed'] = row_imputed.astype(np.int8).to_numpy()
    out['row_open_skip'] = row_open_skip.astype(np.int8).to_numpy()
    out['prev_close'] = prev_close.astype(np.float32).to_numpy()

    out = out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return out.astype(np.float32)


def build_target_frame(feat_df: pd.DataFrame) -> pd.DataFrame:
    return feat_df[TARGET_COLS].copy().astype(np.float32)
"""


V10_MODEL_SRC = """
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class iTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        batch, n_var, time_len, d_model = src.shape
        reshaped = src.permute(0, 2, 1, 3).reshape(batch * time_len, n_var, d_model)
        normed = self.norm1(reshaped)
        attn_out, _ = self.self_attn(normed, normed, normed, need_weights=False)
        reshaped = reshaped + self.dropout1(attn_out)
        src = reshaped.reshape(batch, time_len, n_var, d_model).permute(0, 2, 1, 3)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src2))))
        return src + self.dropout2(src2)


class iTransformerEncoder(nn.Module):
    def __init__(self, input_size: int, d_model: int = 128, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=5000, dropout=dropout)
        self.layers = nn.ModuleList([iTransformerEncoderLayer(d_model, n_heads, dropout=dropout) for _ in range(n_layers)])

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        batch_size, time_len, n_features = src.shape
        src = src.transpose(1, 2).unsqueeze(-1)
        src = self.input_projection(src)
        src = src.reshape(batch_size * n_features, time_len, -1)
        src = self.pos_encoder(src)
        src = src.reshape(batch_size, n_features, time_len, -1)
        for layer in self.layers:
            src = layer(src)
        return src.mean(dim=2).mean(dim=1)


class FrequencyFeatureExtractor(nn.Module):
    def __init__(self, n_fft: int = 16, hop_length: int = 4, out_channels: int = 64):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_freq_bins = n_fft // 2 + 1
        self.conv1 = nn.Conv1d(self.n_freq_bins, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_features = x.shape
        freq_features = []
        for feat_idx in range(min(n_features, 4)):
            signal = x[:, :, feat_idx]
            if seq_len < self.n_fft:
                signal = F.pad(signal, (0, self.n_fft - seq_len), mode='reflect')
            stft_out = torch.stft(
                signal,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=torch.hann_window(self.n_fft, device=x.device),
                center=False,
                return_complex=True,
            )
            freq_features.append(torch.log1p(torch.abs(stft_out)))
        freq_features = torch.stack(freq_features, dim=0).mean(dim=0)
        out = self.conv1(freq_features)
        out = self.bn1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.gelu(out)
        return self.global_pool(out).squeeze(-1)


class MultiScaleFrequencyExtractor(nn.Module):
    def __init__(self, n_ffts: list[int] | None = None, out_channels: int = 64):
        super().__init__()
        self.n_ffts = n_ffts or [8, 16, 32]
        self.extractors = nn.ModuleList(
            [
                FrequencyFeatureExtractor(n_fft=n_fft, hop_length=max(1, n_fft // 4), out_channels=out_channels)
                for n_fft in self.n_ffts
            ]
        )
        self.fusion = nn.Sequential(
            nn.Linear(out_channels * len(self.n_ffts), out_channels * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels * 2, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale = torch.cat([extractor(x) for extractor in self.extractors], dim=-1)
        return self.fusion(multi_scale)


class HybridFTEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        num_gru_layers: int = 2,
        dropout: float = 0.1,
        use_frequency: bool = True,
        freq_out_channels: int = 64,
    ):
        super().__init__()
        self.use_frequency = use_frequency
        self.itransformer = iTransformerEncoder(
            input_size=input_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
        )
        self.freq_encoder = MultiScaleFrequencyExtractor(
            n_ffts=MULTISCALE_N_FFTS,
            out_channels=freq_out_channels,
        ) if use_frequency else None
        fusion_input = d_model + hidden_size + (freq_out_channels if use_frequency else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        itrans_out = self.itransformer(src)
        gru_out, gru_hidden = self.gru(src)
        parts = [itrans_out, gru_hidden[-1]]
        if self.use_frequency and self.freq_encoder is not None:
            parts.append(self.freq_encoder(src))
        fused = self.fusion(torch.cat(parts, dim=-1))
        fused = self.output_norm(fused)
        return fused, gru_out


class Seq2SeqAttnGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout, horizon):
        super().__init__()
        self.horizon = horizon
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.rag_retriever: Optional[RAGPatternRetriever] = None
        self.last_rag_match: Optional[PatternMatch] = None

        self.encoder = HybridFTEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            num_gru_layers=num_layers,
            dropout=dropout,
            use_frequency=USE_FREQUENCY,
            freq_out_channels=FREQ_OUT_CHANNELS,
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

    def attach_retriever(self, retriever: RAGPatternRetriever) -> None:
        self.rag_retriever = retriever

    def _attend(self, h_dec, enc_out):
        query = self.attn_proj(h_dec).unsqueeze(2)
        scores = torch.bmm(enc_out, query).squeeze(2)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)
        return context

    def forward(self, x, y_teacher=None, teacher_forcing_ratio=0.0, return_sigma=False):
        h_dec, enc_out = self.encoder(x)
        dec_input = x[:, -1, : self.output_size]
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
            dec_input = x[:, -1, : self.output_size]
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
                sigma = torch.maximum(sigma, torch.full_like(sigma, MIN_PREDICTED_VOL))
                noise = torch.randn_like(mu) * sigma
                sample = mu + noise
                generated.append(sample.unsqueeze(1))
                dec_input = sample
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
"""


V10_MODEL_TEST_SRC = """
print('=' * 60)
print('V10 MODEL SMOKE TEST')
print('=' * 60)

test_device = RAG_DEVICE if LOW_VRAM_MODE else DEVICE
test_batch_size = 2 if LOW_VRAM_MODE else 4
test_seq_len = 96
test_input_size = len(BASE_FEATURE_COLS) + 1
test_horizon = 12

test_model = Seq2SeqAttnGRU(
    input_size=test_input_size,
    output_size=len(TARGET_COLS),
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    horizon=test_horizon,
).to(test_device)

test_input = torch.randn(test_batch_size, test_seq_len, test_input_size).to(test_device)
test_target = torch.randn(test_batch_size, test_horizon, len(TARGET_COLS)).to(test_device)

mu_out = test_model(test_input)
assert mu_out.shape == (test_batch_size, test_horizon, len(TARGET_COLS))
mu_out, sigma_out = test_model(test_input, return_sigma=True)
assert sigma_out.shape == (test_batch_size, test_horizon, len(TARGET_COLS))
gen_out = test_model.generate_realistic(test_input, temperature=1.0)
assert gen_out.shape == (test_batch_size, test_horizon, len(TARGET_COLS))

print({
    'forward_shape': tuple(mu_out.shape),
    'sigma_shape': tuple(sigma_out.shape),
    'generated_shape': tuple(gen_out.shape),
    'test_device': str(test_device),
})

test_model = test_model.to('cpu')
del test_model, test_input, test_target, mu_out, sigma_out, gen_out
cuda_cleanup()
"""


ROLLING_ENGINE_SRC = code_src(V92, "class RollingPredictionLog")
if "def runrollingbacktest(" not in ROLLING_ENGINE_SRC:
    ROLLING_ENGINE_SRC += """

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


FRAME_SRC = code_src(V92, "def _draw_candles(")
FRAME_SRC = FRAME_SRC.replace("render_single_frame_with_regime", "render_single_frame")
FRAME_SRC = FRAME_SRC.replace("generate_rolling_frames_with_regime", "generate_rolling_frames")

REGIME_TEST_SRC = code_src(V92, "PHASE 2: REGIME DETECTION TEST CELL")


V10_DEVICE_SETUP_SRC = """
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU_TOTAL_MEM_GB = 0.0
if torch.cuda.is_available():
    GPU_TOTAL_MEM_GB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
LOW_VRAM_MODE = bool(torch.cuda.is_available() and GPU_TOTAL_MEM_GB <= 10.0)
ROLLING_TRAIN_DEVICE = torch.device('cpu') if LOW_VRAM_MODE else DEVICE
RAG_DEVICE = torch.device('cpu') if LOW_VRAM_MODE else DEVICE

def cuda_cleanup() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f'Using device: {DEVICE}')
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
print({
    'low_vram_mode': LOW_VRAM_MODE,
    'gpu_total_mem_gb': round(GPU_TOTAL_MEM_GB, 2),
    'rolling_train_device': str(ROLLING_TRAIN_DEVICE),
    'rag_device': str(RAG_DEVICE),
})
"""


V10_MODEL_CONFIG_SRC = """
# Model Configuration
HIDDEN_SIZE = 256  # Increased for better generation capacity
NUM_LAYERS = 2
DROPOUT = 0.20     # Slightly higher for stochasticity
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
BASE_BATCH_SIZE = 256
BATCH_SIZE = 32 if LOW_VRAM_MODE else BASE_BATCH_SIZE
ROLLING_TRAIN_BATCH_SIZE = 16 if LOW_VRAM_MODE else min(128, BASE_BATCH_SIZE)
"""


V10_TRAINING_CONFIG_SRC = """
# Training Configuration
SWEEP_MAX_EPOCHS = 15
SWEEP_PATIENCE = 5
FINAL_MAX_EPOCHS = 60  # More epochs for convergence
FINAL_PATIENCE = 12
TF_START = 1.0
TF_END = 0.0
TF_DECAY_RATE = 0.95
"""


V10_INFERENCE_CONFIG_SRC = """
# Inference Configuration - tuned for realistic 1-minute bars
SAMPLING_TEMPERATURE = 0.50
BASE_ENSEMBLE_SIZE = 16
ENSEMBLE_SIZE = 8 if LOW_VRAM_MODE else BASE_ENSEMBLE_SIZE
TREND_LOOKBACK_BARS = 20
STRONG_TREND_THRESHOLD = 0.002
VOLATILITY_SCALING = True
MIN_PREDICTED_VOL = 0.0001
LOG_SIGMA_MIN = math.log(MIN_PREDICTED_VOL)
LOG_SIGMA_MAX = math.log(0.01)
"""


V10_PHASE5_RAG_CONFIG_SRC = """
# Phase 5: Retrieval-Augmented Pattern Memory
RAG_EMBEDDING_DIM = 64
RAG_K_RETRIEVE = 5
RAG_BLEND_WEIGHT = 0.25
RAG_MAX_PATTERNS = 3000 if LOW_VRAM_MODE else 4000
ROLLING_RAG_MAX_PATTERNS = 1500 if LOW_VRAM_MODE else RAG_MAX_PATTERNS
RAG_ENCODER_HIDDEN = 128
RAG_ENCODER_LAYERS = 2
RAG_BUILD_BATCH_SIZE = 32 if LOW_VRAM_MODE else 128
"""


V10_ROLLING_CONFIG_SRC = """
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

FRAME_OUTPUT_DIR = Path('output/jupyter-notebook/frames/v10')
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
"""


V10_LOSS_SRC = """
def clamp_log_sigma(log_sigma):
    return torch.clamp(log_sigma, min=LOG_SIGMA_MIN, max=LOG_SIGMA_MAX)


def nll_loss(mu, log_sigma, target):
    \"\"\"Per-step Gaussian negative log-likelihood.\"\"\"
    bounded_log_sigma = clamp_log_sigma(log_sigma)
    sigma = torch.exp(bounded_log_sigma)
    return 0.5 * ((target - mu) / sigma) ** 2 + bounded_log_sigma + 0.5 * np.log(2 * np.pi)


def candle_range_loss(mu, target):
    pred_range = mu[:, :, 1] - mu[:, :, 2]  # High - Low
    actual_range = target[:, :, 1] - target[:, :, 2]
    return ((pred_range - actual_range) ** 2).mean()


def volatility_match_loss(mu, log_sigma, target):
    \"\"\"Calibrate sigma to realized autoregressive forecast error.\"\"\"
    pred_vol = torch.exp(clamp_log_sigma(log_sigma))
    realized_abs_error = (target - mu).detach().abs()
    return ((pred_vol - realized_abs_error) ** 2).mean()


def directional_penalty(mu, target):
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
"""


V10_TRAIN_EPOCH_SRC = """
def tf_ratio_for_epoch(epoch):
    ratio = TF_START * (TF_DECAY_RATE ** (epoch - 1))
    return max(float(TF_END), float(ratio))


def run_epoch(model, loader, step_weights_t, optimizer=None, tf_ratio=0.0, device=None):
    is_train = optimizer is not None
    model.train(is_train)
    device = DEVICE if device is None else torch.device(device)

    total_loss, nll_total, range_total, vol_total, dir_total = 0, 0, 0, 0, 0
    n_items = 0

    for xb, yb_s, yb_r in loader:
        xb = xb.to(device)
        yb_s = yb_s.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            mu, log_sigma = model(
                xb,
                y_teacher=yb_s if is_train else None,
                teacher_forcing_ratio=tf_ratio if is_train else 0.0,
                return_sigma=True,
            )

            nll = (nll_loss(mu, log_sigma, yb_s) * step_weights_t).mean()
            rng = candle_range_loss(mu, yb_s)
            vol = volatility_match_loss(mu, log_sigma, yb_s)
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
"""


V10_TRAIN_MODEL_SRC = """
def train_model(model, train_loader, val_loader, max_epochs, patience, device=None):
    device = DEVICE if device is None else torch.device(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    step_idx = np.arange(HORIZON, dtype=np.float32)
    step_w = 1.0 + (step_idx / max(HORIZON - 1, 1)) ** STEP_LOSS_POWER
    step_weights_t = torch.as_tensor(step_w, dtype=torch.float32, device=device).view(1, HORIZON, 1)

    best_val = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    rows = []

    for epoch in range(1, max_epochs + 1):
        tf = tf_ratio_for_epoch(epoch)
        tr = run_epoch(model, train_loader, step_weights_t, optimizer=optimizer, tf_ratio=tf, device=device)
        va = run_epoch(model, val_loader, step_weights_t, optimizer=None, tf_ratio=0.0, device=device)

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
              f"val={va['total']:.6f} (nll={va['nll']:.6f}) | lr={lr:.6g} | device={device}")

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
"""


def apply_v10_stability_patches(nb: dict) -> None:
    base.replace_cell_source(nb, "SEED = 42", V10_DEVICE_SETUP_SRC, cell_type="code")
    base.replace_cell_source(nb, "# Model Configuration", V10_MODEL_CONFIG_SRC, cell_type="code")
    base.replace_cell_source(nb, "# Training Configuration", V10_TRAINING_CONFIG_SRC, cell_type="code")
    base.replace_cell_source(nb, "# Inference Configuration - v7.5 Ensemble Settings", V10_INFERENCE_CONFIG_SRC, cell_type="code")
    base.replace_cell_source(nb, "# Phase 5: Retrieval-Augmented Pattern Memory", V10_PHASE5_RAG_CONFIG_SRC, cell_type="code")
    base.replace_cell_source(nb, "# V8 rolling configuration (frame generator mode)", V10_ROLLING_CONFIG_SRC, cell_type="code")
    base.replace_cell_source(nb, "print('=' * 60)\nprint('V10 MODEL SMOKE TEST')", V10_MODEL_TEST_SRC, cell_type="code")
    base.replace_cell_source(nb, "def nll_loss(mu, log_sigma, target):", V10_LOSS_SRC, cell_type="code")
    base.replace_cell_source(nb, "def tf_ratio_for_epoch(epoch):", V10_TRAIN_EPOCH_SRC, cell_type="code")
    base.replace_cell_source(nb, "def train_model(model, train_loader, val_loader, max_epochs, patience):", V10_TRAIN_MODEL_SRC, cell_type="code")

    rag_src = base.cell_text(nb["cells"][base.find_cell_index(nb, "class RAGPatternRetriever", cell_type="code")])
    rag_src = rag_src.replace(
        "    def build_database(self, sequences: torch.Tensor, future_paths: torch.Tensor, batch_size: int = 256) -> None:\n",
        "    def build_database(self, sequences: torch.Tensor, future_paths: torch.Tensor, batch_size: int = RAG_BUILD_BATCH_SIZE) -> None:\n",
    )
    base.replace_cell_source(nb, "class RAGPatternRetriever", rag_src, cell_type="code")

    model_src = base.cell_text(nb["cells"][base.find_cell_index(nb, "class Seq2SeqAttnGRU", cell_type="code")])
    model_src = model_src.replace(
        "        self.hidden_size = hidden_size\n        self.rag_retriever: Optional[RAGPatternRetriever] = None\n",
        "        self.hidden_size = hidden_size\n        self.return_clip_low: Optional[torch.Tensor] = None\n        self.return_clip_high: Optional[torch.Tensor] = None\n        self.sigma_cap: Optional[torch.Tensor] = None\n        self.rag_retriever: Optional[RAGPatternRetriever] = None\n",
    )
    model_src = model_src.replace(
        "        nn.init.zeros_(self.log_sigma_head[-1].weight)\n        nn.init.zeros_(self.log_sigma_head[-1].bias)\n\n    def attach_retriever(self, retriever: RAGPatternRetriever) -> None:\n        self.rag_retriever = retriever\n",
        """        nn.init.zeros_(self.log_sigma_head[-1].weight)\n        nn.init.zeros_(self.log_sigma_head[-1].bias)\n\n    def _bound_log_sigma(self, log_sigma: torch.Tensor) -> torch.Tensor:\n        return torch.clamp(log_sigma, min=LOG_SIGMA_MIN, max=LOG_SIGMA_MAX)\n\n    def set_prediction_constraints(self, clip_low: np.ndarray, clip_high: np.ndarray, sigma_cap: np.ndarray) -> None:\n        self.return_clip_low = torch.as_tensor(clip_low, dtype=torch.float32)\n        self.return_clip_high = torch.as_tensor(clip_high, dtype=torch.float32)\n        self.sigma_cap = torch.as_tensor(sigma_cap, dtype=torch.float32)\n\n    def _clip_returns(self, values: torch.Tensor, widen: float = 1.0) -> torch.Tensor:\n        if self.return_clip_low is None or self.return_clip_high is None:\n            return values\n        low = self.return_clip_low.to(values.device).view(1, -1)\n        high = self.return_clip_high.to(values.device).view(1, -1)\n        if widen != 1.0:\n            center = 0.5 * (low + high)\n            half = 0.5 * (high - low) * widen\n            low = center - half\n            high = center + half\n        return torch.maximum(torch.minimum(values, high), low)\n\n    def _cap_sigma(self, sigma: torch.Tensor, historical_vol: Optional[float]) -> torch.Tensor:\n        if self.sigma_cap is not None:\n            sigma = torch.minimum(sigma, self.sigma_cap.to(sigma.device).view(1, -1))\n        if historical_vol is not None:\n            hist_cap = max(float(historical_vol) * 3.0, MIN_PREDICTED_VOL)\n            sigma = torch.minimum(sigma, torch.full_like(sigma, hist_cap))\n        return torch.maximum(sigma, torch.full_like(sigma, MIN_PREDICTED_VOL))\n\n    def attach_retriever(self, retriever: RAGPatternRetriever) -> None:\n        self.rag_retriever = retriever\n""",
    )
    model_src = model_src.replace(
        "            mu = self.mu_head(out_features)\n            log_sigma = self.log_sigma_head(out_features)\n",
        "            mu = self._clip_returns(self.mu_head(out_features), widen=1.0)\n            log_sigma = self._bound_log_sigma(self.log_sigma_head(out_features))\n",
    )
    model_src = model_src.replace(
        "                else:\n                    noise = torch.randn_like(mu) * torch.exp(log_sigma).detach()\n                    dec_input = mu + noise\n",
        "                else:\n                    dec_input = mu.detach()\n",
    )
    model_src = model_src.replace(
        "                mu = self.mu_head(out_features)\n                log_sigma = self.log_sigma_head(out_features)\n                sigma = torch.exp(log_sigma) * temperature\n                if historical_vol is not None and t < 5:\n                    sigma = torch.ones_like(sigma) * historical_vol\n                sigma = torch.maximum(sigma, torch.full_like(sigma, MIN_PREDICTED_VOL))\n                noise = torch.randn_like(mu) * sigma\n                sample = mu + noise\n",
        "                mu = self._clip_returns(self.mu_head(out_features), widen=1.10)\n                log_sigma = self._bound_log_sigma(self.log_sigma_head(out_features))\n                sigma = self._cap_sigma(torch.exp(log_sigma) * max(float(temperature), 0.0), historical_vol)\n                if float(temperature) <= 0.0:\n                    noise = torch.zeros_like(mu)\n                else:\n                    noise = torch.randn_like(mu) * sigma\n                sample = self._clip_returns(mu + noise, widen=1.15)\n",
    )
    model_src = model_src.replace(
        "                adjusted_paths.append(adjusted_path.to(generated_paths.device))\n",
        "                adjusted_paths.append(self._clip_returns(adjusted_path.to(generated_paths.device), widen=1.15))\n",
    )
    base.replace_cell_source(nb, "class Seq2SeqAttnGRU", model_src, cell_type="code")

    run_fold_src = base.cell_text(nb["cells"][base.find_cell_index(nb, "def run_fold(", cell_type="code")])
    run_fold_src = run_fold_src.replace(
        "    X_train, y_train_s, y_train_r = X_all[tr_m], y_all_s[tr_m], y_all_r[tr_m]\n    X_val, y_val_s, y_val_r = X_all[va_m], y_all_s[va_m], y_all_r[va_m]\n    X_test, y_test_s, y_test_r = X_all[te_m], y_all_s[te_m], y_all_r[te_m]\n    test_starts = starts[te_m]\n    test_prev_close = prev_close_starts[te_m]\n",
        "    X_train, y_train_s, y_train_r = X_all[tr_m], y_all_s[tr_m], y_all_r[tr_m]\n    X_val, y_val_s, y_val_r = X_all[va_m], y_all_s[va_m], y_all_r[va_m]\n    X_test, y_test_s, y_test_r = X_all[te_m], y_all_s[te_m], y_all_r[te_m]\n    test_starts = starts[te_m]\n    test_prev_close = prev_close_starts[te_m]\n\n    target_constraints = compute_target_constraints(y_train_r)\n    if APPLY_CLIPPING:\n        y_train_s = apply_target_clipping(y_train_s, target_constraints)\n        y_val_s = apply_target_clipping(y_val_s, target_constraints)\n",
    )
    run_fold_src = run_fold_src.replace(
        "    ).to(DEVICE)\n\n\n    hist = train_model(model, train_loader, val_loader, max_epochs, patience)\n",
        "    ).to(DEVICE)\n    model.set_prediction_constraints(\n        target_constraints['clip_low'],\n        target_constraints['clip_high'],\n        target_constraints['sigma_cap'],\n    )\n\n    hist = train_model(model, train_loader, val_loader, max_epochs, patience)\n",
    )
    run_fold_src = run_fold_src.replace(".to(DEVICE)\n    rag_limit = min(RAG_MAX_PATTERNS, len(X_train))", ".to(RAG_DEVICE)\n    rag_limit = min(RAG_MAX_PATTERNS, len(X_train))")
    run_fold_src = run_fold_src.replace(
        "    rag_retriever.build_database(\n        sequences=torch.from_numpy(X_train[:rag_limit]).float(),\n        future_paths=torch.from_numpy(y_train_r[:rag_limit]).float(),\n    )\n",
        "    rag_retriever.build_database(\n        sequences=torch.from_numpy(X_train[:rag_limit]).float(),\n        future_paths=torch.from_numpy(y_train_r[:rag_limit]).float(),\n        batch_size=RAG_BUILD_BATCH_SIZE,\n    )\n",
    )
    run_fold_src = run_fold_src.replace("    actual_ohlc_1 = price_vals[test_starts + 1]\n    prev_ohlc = price_vals[test_starts]\n", "    actual_ohlc_1 = price_vals[test_starts]\n    prev_ohlc = price_vals[test_starts - 1]\n")
    run_fold_src = run_fold_src.replace("    context_df = price_fold.iloc[test_starts[last_idx]-window:test_starts[last_idx]+1][OHLC_COLS]\n", "    context_df = price_fold.iloc[test_starts[last_idx]-window:test_starts[last_idx]][OHLC_COLS]\n")
    run_fold_src = run_fold_src.replace(
        "    return {\n",
        "    model = model.to('cpu')\n    cuda_cleanup()\n\n    return {\n",
    )
    base.replace_cell_source(nb, "def run_fold(", run_fold_src, cell_type="code")

    rolling_train_src = base.cell_text(nb["cells"][base.find_cell_index(nb, "def train_v7_model_for_rolling(", cell_type="code")])
    rolling_train_src = rolling_train_src.replace(
        "    X_train, y_train_s, y_train_r = X_all[:split], y_all_s[:split], y_all_r[:split]\n    X_val, y_val_s, y_val_r = X_all[split:], y_all_s[split:], y_all_r[split:]\n",
        "    X_train, y_train_s, y_train_r = X_all[:split], y_all_s[:split], y_all_r[:split]\n    X_val, y_val_s, y_val_r = X_all[split:], y_all_s[split:], y_all_r[split:]\n\n    target_constraints = compute_target_constraints(y_train_r)\n    if APPLY_CLIPPING:\n        y_train_s = apply_target_clipping(y_train_s, target_constraints)\n        y_val_s = apply_target_clipping(y_val_s, target_constraints)\n",
    )
    rolling_train_src = rolling_train_src.replace(
        "    train_loader = DataLoader(MultiStepDataset(X_train, y_train_s, y_train_r), batch_size=BATCH_SIZE, shuffle=True)\n    val_loader = DataLoader(MultiStepDataset(X_val, y_val_s, y_val_r), batch_size=BATCH_SIZE, shuffle=False)\n",
        "    train_loader = DataLoader(MultiStepDataset(X_train, y_train_s, y_train_r), batch_size=ROLLING_TRAIN_BATCH_SIZE, shuffle=True)\n    val_loader = DataLoader(MultiStepDataset(X_val, y_val_s, y_val_r), batch_size=ROLLING_TRAIN_BATCH_SIZE, shuffle=False)\n",
    )
    rolling_train_src = rolling_train_src.replace(
        "    ).to(DEVICE)\n\n    print({\n",
        "    ).to(ROLLING_TRAIN_DEVICE)\n    model.set_prediction_constraints(\n        target_constraints['clip_low'],\n        target_constraints['clip_high'],\n        target_constraints['sigma_cap'],\n    )\n\n    print({\n",
    )
    rolling_train_src = rolling_train_src.replace(
        "        'backtest_date': backtest_date,\n",
        "        'backtest_date': backtest_date,\n        'rolling_train_device': str(ROLLING_TRAIN_DEVICE),\n        'rolling_train_batch_size': ROLLING_TRAIN_BATCH_SIZE,\n",
    )
    rolling_train_src = rolling_train_src.replace(
        "    history_df = train_model(model, train_loader, val_loader, max_epochs=FINAL_MAX_EPOCHS, patience=FINAL_PATIENCE)\n",
        "    history_df = train_model(model, train_loader, val_loader, max_epochs=FINAL_MAX_EPOCHS, patience=FINAL_PATIENCE, device=ROLLING_TRAIN_DEVICE)\n\n    model = model.to('cpu')\n    cuda_cleanup()\n",
    )
    rolling_train_src = rolling_train_src.replace(".to(DEVICE)\n    rag_limit = min(RAG_MAX_PATTERNS, len(X_all))", ".to(RAG_DEVICE)\n    rag_limit = min(ROLLING_RAG_MAX_PATTERNS, len(X_all))")
    rolling_train_src = rolling_train_src.replace(
        "    rag_retriever.build_database(\n        sequences=torch.from_numpy(X_all[:rag_limit]).float(),\n        future_paths=torch.from_numpy(y_all_r[:rag_limit]).float(),\n    )\n",
        "    rag_retriever.build_database(\n        sequences=torch.from_numpy(X_all[:rag_limit]).float(),\n        future_paths=torch.from_numpy(y_all_r[:rag_limit]).float(),\n        batch_size=RAG_BUILD_BATCH_SIZE,\n    )\n",
    )
    base.replace_cell_source(nb, "def train_v7_model_for_rolling(", rolling_train_src, cell_type="code")


def build_v10() -> dict:
    nb = deepcopy(base.BASE_NOTEBOOK)

    combined_features = list(dict.fromkeys(base.TECHNICAL_FEATURES + ["returns", "turbulence_60", "regime_indicator"]))
    data_config = f"""
# Data Configuration
SYMBOL = 'MSFT'
LOOKBACK_DAYS = 120
OHLC_COLS = ['Open', 'High', 'Low', 'Close']
RAW_COLS = OHLC_COLS + ['Volume', 'TradeCount', 'VWAP']
BASE_CORE_FEATURES = {base.format_py_list(base.BASE_CORE_FEATURES)}
TECHNICAL_FEATURE_COLS = {base.format_py_list(base.TECHNICAL_FEATURES)}
REGIME_FEATURE_COLS = {base.format_py_list(['returns', 'turbulence_60', 'regime_indicator'])}
V10_FEATURE_COLS = {base.format_py_list(combined_features)}
BASE_FEATURE_COLS = BASE_CORE_FEATURES + V10_FEATURE_COLS
TARGET_COLS = ['rOpen', 'rHigh', 'rLow', 'rClose']
INPUT_EXTRA_COL = 'imputedFracWindow'

HORIZON = 50
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
LOOKBACK_CANDIDATES = [64, 96, 160, 256]
DEFAULT_LOOKBACK = 96
ENABLE_LOOKBACK_SWEEP = True
SKIP_OPEN_BARS_TARGET = 6
"""

    title_md = base.phase_intro(
        "v10",
        "Experiment: MSFT 1-Minute Integrated Forecast Stack (v10)",
        [
            "Combines the strongest non-conflicting ideas from Phases 1 to 6 in `UPDATE.md`.",
            "Uses technical indicators and regime features as inputs, then feeds them into a hybrid iTransformer + GRU + frequency encoder.",
            "Adds retrieval-augmented inference, regime-aware rolling temperatures, and keeps the RL decision layer as a post-forecast section driven by strictly causal rolling predictions.",
        ],
        "Integrated design:\n- Phase 1: technical indicators\n- Phase 2: regime detection in features and rolling inference\n- Phase 3 + 4: hybrid time-domain, iTransformer, and frequency encoder\n- Phase 5: retrieval-augmented path adjustment\n- Phase 6: PPO-based decision layer over rolling outputs",
    )
    base.apply_common_patches(nb, "v10", title_md, data_config, "Integrated Phase Stack")

    base.insert_after(
        nb,
        "# Model Configuration",
        [
            base.code_cell(
                """
# Integrated architecture configuration
D_MODEL = 128
N_HEADS = 8
N_LAYERS = 2
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

    base.insert_after(
        nb,
        "# Inference Configuration",
        [
            base.code_cell(base.PHASE2_REGIME_CONFIG_SRC),
            base.code_cell(base.PHASE5_RAG_CONFIG_SRC),
            base.code_cell(base.PHASE6_RL_CONFIG_SRC),
        ],
        cell_type="code",
    )

    base.insert_after(
        nb,
        "print('GPU:'",
        [
            base.markdown_cell("## Regime Detection Components"),
            base.code_cell(code_src(V92, "class MarketRegime(Enum):")),
        ],
        cell_type="code",
    )

    base.replace_cell_source(nb, "## Feature Engineering Functions", "## Feature Engineering Functions (Integrated Phase 1 + 2)", cell_type="markdown")
    base.insert_after(
        nb,
        "## Feature Engineering Functions (Integrated Phase 1 + 2)",
        [
            base.markdown_cell("### Technical Indicator Engine"),
            base.code_cell(base.TECHNICAL_INDICATOR_SRC),
        ],
        cell_type="markdown",
    )
    base.replace_cell_source(nb, "def build_feature_frame(df: pd.DataFrame)", V10_FEATURE_ENGINEERING_SRC, cell_type="code")

    base.insert_after(
        nb,
        "slices = build_walkforward_slices(price_df)",
        [
            base.markdown_cell("## Retrieval-Augmented Pattern Memory"),
            base.code_cell(base.PHASE5_RAG_SRC),
        ],
        cell_type="code",
    )

    base.replace_cell_source(nb, "## Model Definition", "## Integrated Model Definition (Phase 3 + 4 + 5)", cell_type="markdown")
    base.replace_cell_source(nb, "class Seq2SeqAttnGRU", V10_MODEL_SRC, cell_type="code")
    base.insert_after(
        nb,
        "class Seq2SeqAttnGRU",
        [
            base.markdown_cell("### V10 Model Validation"),
            base.code_cell(V10_MODEL_TEST_SRC),
        ],
        cell_type="code",
    )

    run_fold_src = base.cell_text(nb["cells"][base.find_cell_index(nb, "def run_fold(", cell_type="code")])
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
    base.replace_cell_source(nb, "def run_fold(", run_fold_src, cell_type="code")

    rolling_train_src = base.cell_text(nb["cells"][base.find_cell_index(nb, "def train_v7_model_for_rolling(", cell_type="code")])
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
    base.replace_cell_source(nb, "def train_v7_model_for_rolling(", rolling_train_src, cell_type="code")

    base.replace_cell_source(nb, "class RollingPredictionLog", ROLLING_ENGINE_SRC, cell_type="code")
    base.replace_cell_source(nb, "def _draw_candles(", FRAME_SRC, cell_type="code")

    base.insert_after(
        nb,
        "def generate_rolling_frames(",
        [
            base.markdown_cell("## Regime Validation"),
            base.code_cell(REGIME_TEST_SRC),
            base.markdown_cell(base.PHASE6_RL_SECTION_MD),
            base.code_cell(base.PHASE6_RL_CODE),
        ],
        cell_type="code",
    )

    apply_v10_stability_patches(nb)
    base.patch_min_predicted_vol(nb)
    base.clear_notebook_outputs(nb)
    return nb


def main() -> None:
    notebook = build_v10()
    out_path = base.OUTPUT_DIR / "v10.ipynb"
    out_path.write_text(base.json.dumps(notebook, indent=1) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
