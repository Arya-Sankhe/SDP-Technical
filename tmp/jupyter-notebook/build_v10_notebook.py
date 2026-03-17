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

test_batch_size = 4
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
).to(DEVICE)

test_input = torch.randn(test_batch_size, test_seq_len, test_input_size).to(DEVICE)
test_target = torch.randn(test_batch_size, test_horizon, len(TARGET_COLS)).to(DEVICE)

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
})
"""


ROLLING_ENGINE_SRC = code_src(V92, "class RollingPredictionLog")
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
