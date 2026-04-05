$path = 'D:\APPS\Github\SDP-Technical\output\jupyter-notebook\v9.4.ipynb'
$nb = Get-Content -Raw $path | ConvertFrom-Json

function Get-CellIndex([string]$needle) {
    for ($i = 0; $i -lt $nb.cells.Count; $i++) {
        $src = [string]::Join('', $nb.cells[$i].source)
        if ($src.Contains($needle)) { return $i }
    }
    throw "Could not find cell containing: $needle"
}

function Get-CellText([int]$idx) {
    return [string]::Join('', $nb.cells[$idx].source)
}

function Set-CellText([int]$idx, [string]$text) {
    $text = $text.TrimStart("`r", "`n")
    $lines = New-Object 'System.Collections.Generic.List[string]'
    $reader = New-Object System.IO.StringReader($text)
    while (($line = $reader.ReadLine()) -ne $null) {
        $lines.Add($line + "`n")
    }
    $nb.cells[$idx].source = $lines
}

$trainingCell = @'
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
'@
Set-CellText (Get-CellIndex '# Training Configuration') $trainingCell

$inferenceCell = @'
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
'@
Set-CellText (Get-CellIndex '# Inference Configuration - v7.5 Ensemble Settings') $inferenceCell

$rollingCell = @'
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
'@
Set-CellText (Get-CellIndex '# V8 rolling configuration (frame generator mode)') $rollingCell

$modelIdx = Get-CellIndex 'class Seq2SeqAttnGRU'
$modelSrc = Get-CellText $modelIdx

$old = @'
        self.hidden_size = hidden_size
'@
$new = @'
        self.hidden_size = hidden_size
        self.return_clip_low: Optional[torch.Tensor] = None
        self.return_clip_high: Optional[torch.Tensor] = None
        self.sigma_cap: Optional[torch.Tensor] = None
'@
$modelSrc = $modelSrc.Replace($old, $new)

$old = @'
    def _bound_log_sigma(self, log_sigma):
        return torch.clamp(log_sigma, min=LOG_SIGMA_MIN, max=LOG_SIGMA_MAX)
'@
$new = @'
    def _bound_log_sigma(self, log_sigma):
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
'@
$modelSrc = $modelSrc.Replace($old, $new)

$old = @'
            mu = self.mu_head(out_features)
            log_sigma = self._bound_log_sigma(self.log_sigma_head(out_features))
'@
$new = @'
            mu = self._clip_returns(self.mu_head(out_features), widen=1.0)
            log_sigma = self._bound_log_sigma(self.log_sigma_head(out_features))
'@
$modelSrc = $modelSrc.Replace($old, $new)

$old = @'
                else:
                    noise = torch.randn_like(mu) * torch.exp(log_sigma).detach()
                    dec_input = mu + noise
            else:
                dec_input = mu
'@
$new = @'
                else:
                    dec_input = mu.detach()
            else:
                dec_input = mu
'@
$modelSrc = $modelSrc.Replace($old, $new)

$old = @'
                mu = self.mu_head(out_features)
                log_sigma = self._bound_log_sigma(self.log_sigma_head(out_features))

                sigma = torch.exp(log_sigma) * temperature

                if historical_vol is not None and t < 5:
                    sigma = torch.ones_like(sigma) * historical_vol

                sigma = torch.maximum(sigma, torch.full_like(sigma, MIN_PREDICTED_VOL))

                noise = torch.randn_like(mu) * sigma
                sample = mu + noise
'@
$new = @'
                mu = self._clip_returns(self.mu_head(out_features), widen=1.10)
                log_sigma = self._bound_log_sigma(self.log_sigma_head(out_features))

                sigma = self._cap_sigma(torch.exp(log_sigma) * max(float(temperature), 0.0), historical_vol)
                if float(temperature) <= 0.0:
                    noise = torch.zeros_like(mu)
                else:
                    noise = torch.randn_like(mu) * sigma
                sample = self._clip_returns(mu + noise, widen=1.15)
'@
$modelSrc = $modelSrc.Replace($old, $new)
Set-CellText $modelIdx $modelSrc

$lossIdx = Get-CellIndex 'def clamp_log_sigma('
$lossSrc = Get-CellText $lossIdx

$old = @'
def nll_loss(mu, log_sigma, target):
    """Negative log-likelihood for Gaussian"""
    bounded_log_sigma = clamp_log_sigma(log_sigma)
    sigma = torch.exp(bounded_log_sigma)
    nll = 0.5 * ((target - mu) / sigma) ** 2 + bounded_log_sigma + 0.5 * np.log(2 * np.pi)
    return nll.mean()
'@
$new = @'
def nll_loss(mu, log_sigma, target):
    """Per-step Gaussian negative log-likelihood."""
    bounded_log_sigma = clamp_log_sigma(log_sigma)
    sigma = torch.exp(bounded_log_sigma)
    return 0.5 * ((target - mu) / sigma) ** 2 + bounded_log_sigma + 0.5 * np.log(2 * np.pi)
'@
$lossSrc = $lossSrc.Replace($old, $new)

$old = @'
def volatility_match_loss(log_sigma, target):
    """Encourage predicted uncertainty to match actual error magnitude"""
    pred_vol = torch.exp(clamp_log_sigma(log_sigma)).mean()
    actual_vol = target.std()
    return (pred_vol - actual_vol) ** 2
'@
$new = @'
def volatility_match_loss(mu, log_sigma, target):
    """Calibrate sigma to realized autoregressive forecast error."""
    pred_vol = torch.exp(clamp_log_sigma(log_sigma))
    realized_abs_error = (target - mu).detach().abs()
    return ((pred_vol - realized_abs_error) ** 2).mean()
'@
$lossSrc = $lossSrc.Replace($old, $new)

$old = @'
def directional_penalty(mu, target):
    pred_close = mu[:, :, 3]
    actual_close = target[:, :, 3]
    sign_match = torch.sign(pred_close) * torch.sign(actual_close)
    penalty = torch.clamp(-sign_match, min=0.0)
    return penalty.mean()
'@
$new = @'
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
'@
$lossSrc = $lossSrc.Replace($old, $new)
$lossSrc = $lossSrc.Replace(
    '            vol = volatility_match_loss(log_sigma, yb_s)' + "`n",
    '            vol = volatility_match_loss(mu, log_sigma, yb_s)' + "`n"
)
Set-CellText $lossIdx $lossSrc

$runFoldIdx = Get-CellIndex 'def run_fold('
$runFoldSrc = Get-CellText $runFoldIdx

$old = @'
    X_train, y_train_s, y_train_r = X_all[tr_m], y_all_s[tr_m], y_all_r[tr_m]
    X_val, y_val_s, y_val_r = X_all[va_m], y_all_s[va_m], y_all_r[va_m]
    X_test, y_test_s, y_test_r = X_all[te_m], y_all_s[te_m], y_all_r[te_m]
    test_starts = starts[te_m]
    test_prev_close = prev_close_starts[te_m]
'@
$new = @'
    X_train, y_train_s, y_train_r = X_all[tr_m], y_all_s[tr_m], y_all_r[tr_m]
    X_val, y_val_s, y_val_r = X_all[va_m], y_all_s[va_m], y_all_r[va_m]
    X_test, y_test_s, y_test_r = X_all[te_m], y_all_s[te_m], y_all_r[te_m]
    test_starts = starts[te_m]
    test_prev_close = prev_close_starts[te_m]

    target_constraints = compute_target_constraints(y_train_r)
    if APPLY_CLIPPING:
        y_train_s = apply_target_clipping(y_train_s, target_constraints)
        y_val_s = apply_target_clipping(y_val_s, target_constraints)
'@
$runFoldSrc = $runFoldSrc.Replace($old, $new)

$old = @'
    model = Seq2SeqAttnGRU(
        input_size=X_train.shape[-1],
        output_size=len(TARGET_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        horizon=HORIZON,
        use_frequency=USE_FREQUENCY,
        freq_out_channels=FREQ_OUT_CHANNELS,
    ).to(DEVICE)
'@
$new = @'
    model = Seq2SeqAttnGRU(
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
'@
$runFoldSrc = $runFoldSrc.Replace($old, $new)
$runFoldSrc = $runFoldSrc.Replace(
    '        actual_ohlc_1 = price_vals[test_starts + 1]' + "`n" + '        prev_ohlc = price_vals[test_starts]' + "`n",
    '        actual_ohlc_1 = price_vals[test_starts]' + "`n" + '        prev_ohlc = price_vals[test_starts - 1]' + "`n"
)
$runFoldSrc = $runFoldSrc.Replace(
    '        context_df = price_fold.iloc[test_starts[last_idx]-window:test_starts[last_idx]+1][OHLC_COLS]' + "`n",
    '        context_df = price_fold.iloc[test_starts[last_idx]-window:test_starts[last_idx]][OHLC_COLS]' + "`n"
)
Set-CellText $runFoldIdx $runFoldSrc

$rollingIdx = Get-CellIndex 'def train_v7_model_for_rolling('
$rollingSrc = Get-CellText $rollingIdx

$old = @'
    X_train, y_train_s, y_train_r = X_all[:split], y_all_s[:split], y_all_r[:split]
    X_val, y_val_s, y_val_r = X_all[split:], y_all_s[split:], y_all_r[split:]
'@
$new = @'
    X_train, y_train_s, y_train_r = X_all[:split], y_all_s[:split], y_all_r[:split]
    X_val, y_val_s, y_val_r = X_all[split:], y_all_s[split:], y_all_r[split:]

    target_constraints = compute_target_constraints(y_train_r)
    if APPLY_CLIPPING:
        y_train_s = apply_target_clipping(y_train_s, target_constraints)
        y_val_s = apply_target_clipping(y_val_s, target_constraints)
'@
$rollingSrc = $rollingSrc.Replace($old, $new)

$old = @'
    model = Seq2SeqAttnGRU(
        input_size=X_train.shape[-1],
        output_size=len(TARGET_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        horizon=horizon,
        use_frequency=USE_FREQUENCY,
        freq_out_channels=FREQ_OUT_CHANNELS,
    ).to(DEVICE)
'@
$new = @'
    model = Seq2SeqAttnGRU(
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
'@
$rollingSrc = $rollingSrc.Replace($old, $new)
Set-CellText $rollingIdx $rollingSrc

foreach ($cell in $nb.cells) {
    if ($cell.cell_type -eq 'code') {
        $cell.execution_count = $null
        $cell.outputs = @()
    }
}

$json = $nb | ConvertTo-Json -Depth 100
Set-Content -Path $path -Value ($json + "`n") -Encoding utf8
Write-Output 'Patched v9.4 notebook and cleared outputs.'
