import json
import re
from pathlib import Path


ROOT = Path(r"D:\APPS\Github\SDP-Technical")
BASE_TRAIN = ROOT / "FinalTrain.ipynb"
BASE_INFER = ROOT / "FINAL.ipynb"
OUT_TRAIN = ROOT / "Final1DTrain.ipynb"
OUT_INFER = ROOT / "FINAL1D.ipynb"


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_notebook(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def get_source(notebook: dict, idx: int) -> str:
    return "".join(notebook["cells"][idx].get("source", []))


def set_source(notebook: dict, idx: int, text: str) -> None:
    lines = text.splitlines(keepends=True)
    if text and not text.endswith("\n"):
        lines.append("\n")
    notebook["cells"][idx]["source"] = lines


def replace_or_raise(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"Expected snippet not found:\n{old[:200]}")
    return text.replace(old, new)


TRAIN_CELL0 = """# Final1DTrain.ipynb

This notebook is the offline build step for the daily stacked system. It trains the three daily forecast experts (`v8.5`, `v9.2`, `v9.5`) on 1-day `MSFT` candles, saves every reusable artifact they need for inference, computes ensemble weights on aligned validation/rolling metrics, and then trains the `v9.6` PPO decision layer on top of the aggregate daily forecast stream.

The output of this notebook is a frozen artifact tree under `output/final_1d_artifacts/`. `FINAL1D.ipynb` should only load those artifacts and run inference.
"""


TRAIN_CELL3 = """SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOW_VRAM_GPU = DEVICE.type == "cuda" and (torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)) <= 8.5
LIGHTWEIGHT_V95_ON_LOW_VRAM = os.name == "nt" and LOW_VRAM_GPU

SYMBOL = "MSFT"
LOOKBACK_DAYS = 3650
HORIZON = 7
DAILY_ROLLING_EVAL_BARS = 60
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

ARTIFACT_ROOT = Path("output/final_1d_artifacts")
SHARED_DIR = ARTIFACT_ROOT / "shared"
MODELS_DIR = ARTIFACT_ROOT / "models"
ENSEMBLE_DIR = ARTIFACT_ROOT / "ensemble"
RL_DIR = ARTIFACT_ROOT / "rl"

REFRESH_SHARED_SNAPSHOT = True
RESUME_COMPLETED_EXPERTS = True
SAVE_PARQUET_COMPRESSION = "snappy"
REQUEST_CHUNK_DAYS = 365
MAX_REQUESTS_PER_MINUTE = 120
MAX_RETRIES = 5
SESSION_TZ = "America/New_York"
SESSION_OPEN_SKIP_BARS = 0
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
    ExpertSpec("v9_2", "v9.2", 256, "regime", "gru", ensemble_size=20, batch_size=256, eval_batch_size=256, use_regime=True),
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

ACTIVE_FORECAST_SPECS = FORECAST_SPECS.copy()

print({
    "symbol": SYMBOL,
    "timeframe": "1D",
    "device": str(DEVICE),
    "low_vram_gpu": LOW_VRAM_GPU,
    "lightweight_v95_on_low_vram": LIGHTWEIGHT_V95_ON_LOW_VRAM,
    "artifact_root": str(ARTIFACT_ROOT),
    "horizon_days": HORIZON,
    "rolling_eval_bars": DAILY_ROLLING_EVAL_BARS,
    "forecast_experts": [spec.version for spec in ACTIVE_FORECAST_SPECS],
    "decision_layer": "v9.6 PPO",
})
"""


TRAIN_CELL5 = """class RequestPacer:
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

    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=lookback_days)
    cursor = start
    frames: List[pd.DataFrame] = []
    api_calls = 0

    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(days=REQUEST_CHUNK_DAYS), end)
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
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
            except Exception:
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


def prepare_daily_bars(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        raise ValueError("raw_df is empty")

    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    df["row_imputed"] = df[["open", "high", "low", "close", "vwap"]].isna().any(axis=1)
    for price_col in ["open", "high", "low", "close", "vwap"]:
        df[price_col] = df[price_col].ffill().bfill()
    for count_col in ["volume", "trade_count"]:
        df[count_col] = df[count_col].fillna(0.0)
    df["timestamp_ny"] = df["timestamp"].dt.tz_convert(SESSION_TZ)
    df["session_date"] = df["timestamp_ny"].dt.strftime("%Y-%m-%d")
    df["bar_in_session"] = 0
    df["row_open_skip"] = False
    df["session_minutes"] = 1
    df["session_progress"] = 1.0
    return df
"""


TRAIN_CELL6 = """raw_snapshot_path = SHARED_DIR / "raw_daily_snapshot.parquet"
session_snapshot_path = SHARED_DIR / "daily_snapshot.parquet"

if REFRESH_SHARED_SNAPSHOT or not raw_snapshot_path.exists() or not session_snapshot_path.exists():
    raw_df_utc, api_calls = fetch_bars_alpaca(SYMBOL, LOOKBACK_DAYS)
    session_df = prepare_daily_bars(raw_df_utc)
    save_dataframe(raw_df_utc, raw_snapshot_path)
    save_dataframe(session_df, session_snapshot_path)
    save_json(
        SHARED_DIR / "calendar_meta.json",
        {
            **build_runtime_metadata(),
            "api_calls": api_calls,
            "rows_raw": int(len(raw_df_utc)),
            "rows_daily": int(len(session_df)),
            "trading_days": int(session_df["session_date"].nunique()),
            "session_tz": SESSION_TZ,
            "bar_timeframe": "1D",
        },
    )
else:
    raw_df_utc = pd.read_parquet(raw_snapshot_path)
    session_df = pd.read_parquet(session_snapshot_path)

display(session_df.head())
print(
    {
        "rows_raw": len(raw_df_utc),
        "rows_daily": len(session_df),
        "trading_days": session_df["session_date"].nunique(),
        "first_timestamp": str(session_df["timestamp"].min()),
        "last_timestamp": str(session_df["timestamp"].max()),
    }
)
"""


TRAIN_CELL11 = """def summarize_slice_results(slice_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_keys = ["step_1_close_mae", "horizon_close_mae", "path_close_mae", "directional_accuracy"]
    summary = {}
    for key in metric_keys:
        summary[key] = float(np.mean([result["metrics"][key] for result in slice_results]))
        summary[f"baseline_{key}"] = float(np.mean([result["baseline_metrics"][key] for result in slice_results]))
    summary["slice_count"] = len(slice_results)
    summary["val_windows_used"] = int(np.sum([result["val_windows_used"] for result in slice_results]))
    return summary


def rolling_metric_steps() -> List[int]:
    return sorted({1, min(3, HORIZON), min(5, HORIZON), HORIZON})


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


def daily_temperature(anchor_ts: pd.Timestamp) -> float:
    return 1.0


def select_backtest_date(session_df: pd.DataFrame, max_lookback: int, horizon: int, rolling_eval_bars: int) -> Tuple[str, int]:
    last_anchor = len(session_df) - horizon
    if last_anchor <= max_lookback:
        raise RuntimeError("Not enough rows for a daily rolling backtest.")
    first_anchor = max(max_lookback, last_anchor - rolling_eval_bars + 1)
    backtest_date = pd.Timestamp(session_df["timestamp"].iloc[first_anchor]).tz_convert(SESSION_TZ).strftime("%Y-%m-%d")
    return backtest_date, int(first_anchor)


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
    if history_slice.empty:
        return "NORMAL", 1.0, 0.0
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
    backtest_start_index: int,
    production_bundle: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    feature_columns = production_bundle["feature_columns"]
    scaler = production_bundle["scaler"]
    model = production_bundle["model"]
    retrieval_artifact = production_bundle["retrieval_artifact"]
    model.eval()

    logs: List[Dict[str, Any]] = []
    first_anchor = max(int(backtest_start_index), spec.lookback)
    last_anchor = len(feature_df) - HORIZON
    if first_anchor > last_anchor:
        raise RuntimeError(f"No valid daily rolling anchors for {spec.version}")

    metric_steps = rolling_metric_steps()

    for anchor in tqdm(range(first_anchor, last_anchor + 1), desc=f"{spec.version} rolling"):
        context = feature_df.iloc[anchor - spec.lookback : anchor]
        feature_block = context.loc[:, feature_columns].to_numpy(dtype=np.float32)
        impute_frac = float(context["row_imputed"].mean())
        feature_block = np.concatenate([feature_block, np.full((spec.lookback, 1), impute_frac, dtype=np.float32)], axis=1)
        feature_block = apply_input_scaler(feature_block[None, ...], scaler)[0]

        anchor_prev_close = float(feature_df["prev_close"].iloc[anchor])
        actual_returns = feature_df.iloc[anchor : anchor + HORIZON].loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float32)
        actual_path = returns_to_prices_seq(anchor_prev_close, actual_returns)
        base_temp = daily_temperature(pd.Timestamp(feature_df["timestamp"].iloc[anchor]))
        regime_name = "NORMAL"
        regime_indicator = 0.0
        regime_multiplier = 1.0
        if spec.use_regime:
            regime_name, regime_multiplier, regime_indicator = detect_regime_multiplier(feature_df.iloc[max(0, anchor - 252) : anchor])
        predicted_path = generate_ensemble_with_trend_selection(
            model=model,
            spec=spec,
            x_scaled_single=feature_block,
            anchor_prev_close=anchor_prev_close,
            historical_closes=context["close"].to_numpy(dtype=np.float32),
            temperature=base_temp * regime_multiplier,
            retrieval_artifact=retrieval_artifact,
        )

        row = {
            "timestamp": pd.Timestamp(feature_df["timestamp"].iloc[anchor]),
            "prev_close": anchor_prev_close,
            "predicted_path": predicted_path,
            "actual_path": actual_path,
            "direction_hit": float(np.sign(predicted_path[0, 3] - anchor_prev_close) == np.sign(actual_path[0, 3] - anchor_prev_close)),
            "regime_name": regime_name,
            "regime_indicator": regime_indicator,
            "temperature": base_temp * regime_multiplier,
        }
        for step in metric_steps:
            row[f"step_{step}_close_mae"] = float(abs(predicted_path[step - 1, 3] - actual_path[step - 1, 3]))
        logs.append(row)

    rolling_df = pd.DataFrame(
        [
            {
                "timestamp": row["timestamp"],
                "prev_close": row["prev_close"],
                **{f"step_{step}_close_mae": row[f"step_{step}_close_mae"] for step in metric_steps},
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
        **{f"step_{step}_close_mae": float(rolling_df[f"step_{step}_close_mae"].mean()) for step in metric_steps},
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
"""


TRAIN_CELL12 = """MAX_LOOKBACK = max(spec.lookback for spec in ACTIVE_FORECAST_SPECS)
BACKTEST_DATE, BACKTEST_CUTOFF_INDEX = select_backtest_date(
    session_df,
    max_lookback=MAX_LOOKBACK,
    horizon=HORIZON,
    rolling_eval_bars=DAILY_ROLLING_EVAL_BARS,
)

save_json(
    SHARED_DIR / "split_spec.json",
    {
        "backtest_date": BACKTEST_DATE,
        "backtest_cutoff_index": BACKTEST_CUTOFF_INDEX,
        "walkforward_slices": build_walkforward_slices(len(session_df)),
        "production_val_fraction": PRODUCTION_VAL_FRACTION,
        "validation_slice_fraction": VALIDATION_SLICE_FRACTION,
        "rolling_eval_bars": DAILY_ROLLING_EVAL_BARS,
        "timeframe": "1D",
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
    rolling_df, rolling_payload = run_rolling_for_spec(feature_df, spec, BACKTEST_DATE, BACKTEST_CUTOFF_INDEX, production_bundle)
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
        "timeframe": "1D",
    },
)

display(validation_summary_df)
display(rolling_summary_df)
display(ensemble_weights_df)
"""


TRAIN_CELL15 = """manifest = {
    "runtime": build_runtime_metadata(),
    "shared": {
        "raw_snapshot": str(raw_snapshot_path),
        "daily_snapshot": str(session_snapshot_path),
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
"""


INFER_CELL0 = """# FINAL1D.ipynb

This notebook is the inference-only side of the daily workflow. It expects `Final1DTrain.ipynb` to have already created `output/final_1d_artifacts/manifest.json` plus all forecast checkpoints, scaler stats, feature manifests, ensemble weights, and PPO artifacts.

The notebook loads those frozen artifacts, fetches the latest daily market data, runs the forecast experts one at a time, shows each expert prediction, builds the weighted aggregate forecast, runs the saved `v9.6` decision layer on top, and includes a strictly causal rolling daily backtest driven by the frozen ensemble.
"""


INFER_CELL3 = """SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOW_VRAM_GPU = DEVICE.type == "cuda" and (torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)) <= 8.5
RUN_LABEL = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")

SYMBOL = "MSFT"
LOOKBACK_DAYS = 3650
HORIZON = 7
SESSION_TZ = "America/New_York"
SESSION_OPEN_SKIP_BARS = 0
REQUEST_CHUNK_DAYS = 365
MAX_REQUESTS_PER_MINUTE = 120
MAX_RETRIES = 5
ACTION_NEUTRAL_BAND = 0.10
SAVE_RUN_OUTPUTS = True
FINAL_ROLLING_STEP = 1
FINAL_ROLLING_BACKTEST_DATE = None
FINAL_ROLLING_WINDOW = 60
FINAL_ROLLING_HISTORY_BARS = 180
FINAL_ROLLING_SAVE_FRAMES = False
FINAL_ROLLING_DISPLAY_FRAME_INDEX = -1
FINAL_ROLLING_USE_CPU_ON_LOW_VRAM = LOW_VRAM_GPU
FINAL_ROLLING_DEVICE = torch.device("cpu") if FINAL_ROLLING_USE_CPU_ON_LOW_VRAM else DEVICE
FINAL_ROLLING_CLEAR_EVERY = 24
FINAL_WEIGHT_PRIOR_BLEND = 0.0
FINAL_EXCLUDED_EXPERTS = set()
FINAL_AGGREGATION_WEIGHT_PRIOR = {
    "v8_5": 0.33,
    "v9_2": 0.33,
    "v9_5": 0.34,
}
FINAL_AGGREGATE_SHRINK = 1.00
FINAL_SOFT_SWING_GUARD_ENABLED = False
FINAL_SOFT_SWING_GUARD_LOOKBACK = 252
FINAL_SOFT_SWING_GUARD_Q95_MULT = 1.80
FINAL_SOFT_SWING_GUARD_ATR_MULT = 1.10
FINAL_SOFT_SWING_GUARD_MIN_MOVE = 1.00
FINAL_SOFT_SWING_GUARD_EXCESS_SCALE = 0.45
FINAL_SOFT_SWING_GUARD_HARD_MULT = 1.35
FINAL_SOFT_SWING_GUARD_EXTREME_SCALE = 0.08
FINAL_EMPIRICAL_ENVELOPE_ENABLED = False
FINAL_EMPIRICAL_ENVELOPE_LOOKBACK = 252
FINAL_EMPIRICAL_ENVELOPE_Q95_MULT = 1.45
FINAL_EMPIRICAL_ENVELOPE_STEP_BASE_MULT = 1.15
FINAL_EMPIRICAL_ENVELOPE_ATR_MULT = 1.00
FINAL_EMPIRICAL_ENVELOPE_MIN_MOVE = 1.00
FINAL_EMPIRICAL_ENVELOPE_EXCESS_SCALE = 0.35
FINAL_EMPIRICAL_ENVELOPE_HARD_MULT = 1.28
FINAL_EMPIRICAL_ENVELOPE_EXTREME_SCALE = 0.08
FINAL_CANDLE_RANGE_GUARD_ENABLED = False
FINAL_CANDLE_RANGE_GUARD_LOOKBACK = 252
FINAL_CANDLE_RANGE_GUARD_Q97_MULT = 1.85
FINAL_CANDLE_RANGE_GUARD_MIN_WICK = 0.90
FINAL_AGGREGATE_GUARD_LOOKBACK = max(
    FINAL_SOFT_SWING_GUARD_LOOKBACK,
    FINAL_EMPIRICAL_ENVELOPE_LOOKBACK,
    FINAL_CANDLE_RANGE_GUARD_LOOKBACK,
)
FINAL_T1_TEMPORAL_GUARD_ENABLED = False
FINAL_T1_TEMPORAL_GUARD_LOOKBACK = 252
FINAL_T1_TEMPORAL_Q95_MULT = 1.30
FINAL_T1_TEMPORAL_ATR_MULT = 0.75
FINAL_T1_TEMPORAL_MIN_MOVE = 0.75
FINAL_T1_TEMPORAL_EXCESS_SCALE = 0.22
FINAL_T1_TEMPORAL_HARD_MULT = 1.18
FINAL_T1_TEMPORAL_EXTREME_SCALE = 0.00
FINAL_T1_TEMPORAL_RECAP_EXCESS_SCALE = 0.15
FINAL_T1_TEMPORAL_RECAP_HARD_MULT = 1.12
FINAL_T1_TEMPORAL_RECAP_EXTREME_SCALE = 0.00
FINAL_T1_DIRECTION_GUARD_ENABLED = False
FINAL_T1_DIRECTION_VOTE_THRESHOLD = 0.35
FINAL_T1_DIRECTION_MOMENTUM_BARS = 3
FINAL_T1_DIRECTION_MOMENTUM_MIN_MOVE = 0.15
FINAL_T1_DIRECTION_MAX_ABS_DELTA_TO_OVERRIDE = 0.80
FINAL_T1_DIRECTION_MIN_OVERRIDE_MOVE = 0.15
FINAL_T1_DIRECTION_MAX_OVERRIDE_MOVE = 0.60
FINAL_T1_SHIFT_FADE_POWER = 2.0

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
EXPECTED_EXPERTS: List[str] = []

ARTIFACT_ROOT = Path("output/final_1d_artifacts")
MANIFEST_PATH = ARTIFACT_ROOT / "manifest.json"
FINAL_RUN_DIR = Path("output/final_1d_runs") / RUN_LABEL

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
        "final_rolling_window": FINAL_ROLLING_WINDOW,
        "final_rolling_save_frames": FINAL_ROLLING_SAVE_FRAMES,
        "final_rolling_device": str(FINAL_ROLLING_DEVICE),
        "final_weight_prior_blend": FINAL_WEIGHT_PRIOR_BLEND,
        "final_excluded_experts": sorted(FINAL_EXCLUDED_EXPERTS),
        "final_aggregate_shrink": FINAL_AGGREGATE_SHRINK,
        "final_soft_swing_guard_enabled": FINAL_SOFT_SWING_GUARD_ENABLED,
        "final_soft_swing_guard_lookback": FINAL_SOFT_SWING_GUARD_LOOKBACK,
        "final_empirical_envelope_lookback": FINAL_EMPIRICAL_ENVELOPE_LOOKBACK,
        "final_candle_range_guard_lookback": FINAL_CANDLE_RANGE_GUARD_LOOKBACK,
        "final_t1_temporal_guard_lookback": FINAL_T1_TEMPORAL_GUARD_LOOKBACK,
    }
)
"""


INFER_CELL5 = """fail_if_missing(MANIFEST_PATH, "Missing final 1D artifact manifest. Run Final1DTrain.ipynb first")
manifest = load_json(MANIFEST_PATH)

for key in ["shared", "forecast_models", "ensemble", "rl"]:
    if key not in manifest:
        raise KeyError(f"Manifest missing required section: {key}")

SYMBOL = manifest["runtime"].get("symbol", SYMBOL)
HORIZON = int(manifest["runtime"].get("horizon", HORIZON))

MODEL_ARTIFACTS = {
    expert_name: artifact
    for expert_name, artifact in manifest["forecast_models"].items()
    if expert_name not in FINAL_EXCLUDED_EXPERTS
}
EXPECTED_EXPERTS = list(MODEL_ARTIFACTS.keys())
if len(EXPECTED_EXPERTS) < 2:
    raise ValueError(f"Need at least 2 active forecast experts after exclusions, found {EXPECTED_EXPERTS}")

weights_path = Path(manifest["ensemble"]["weights_json"])
policy_path = Path(manifest["rl"]["policy_path"])
env_config_path = Path(manifest["rl"]["env_config_path"])
state_schema_path = Path(manifest["rl"]["state_schema_path"])

for path in [weights_path, policy_path, env_config_path, state_schema_path]:
    fail_if_missing(path, "Required inference artifact is missing")

saved_ensemble_weights = load_json(weights_path)
if abs(sum(saved_ensemble_weights.values()) - 1.0) >= 1e-6:
    raise ValueError("Saved ensemble weights do not sum to 1.0")
ensemble_weights = build_final_inference_weights(saved_ensemble_weights, EXPECTED_EXPERTS)

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
            "saved_weight": saved_ensemble_weights.get(expert_name, 0.0),
            "final_weight": ensemble_weights[expert_name],
        }
    )
artifact_summary_df = pd.DataFrame(artifact_rows)
display(artifact_summary_df)
"""


INFER_CELL6 = """class RequestPacer:
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
        raise RuntimeError("Set ALPACA_API_KEY / ALPACA_API_SECRET before running FINAL1D.ipynb.")
    return api_key, api_secret


def _resolve_feed() -> DataFeed:
    preferred = (os.getenv("ALPACA_FEED") or "iex").strip().lower()
    return DataFeed.SIP if preferred == "sip" else DataFeed.IEX


def fetch_bars_alpaca(symbol: str, lookback_days: int) -> Tuple[pd.DataFrame, int]:
    api_key, api_secret = _require_alpaca_credentials()
    client = StockHistoricalDataClient(api_key, api_secret)
    feed = _resolve_feed()
    pacer = RequestPacer(MAX_REQUESTS_PER_MINUTE)

    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=lookback_days)
    cursor = start
    frames: List[pd.DataFrame] = []
    api_calls = 0

    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(days=REQUEST_CHUNK_DAYS), end)
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
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
            except Exception:
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


def prepare_daily_bars(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        raise ValueError("raw_df is empty")

    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    df["row_imputed"] = df[["open", "high", "low", "close", "vwap"]].isna().any(axis=1)
    for price_col in ["open", "high", "low", "close", "vwap"]:
        df[price_col] = df[price_col].ffill().bfill()
    for count_col in ["volume", "trade_count"]:
        df[count_col] = df[count_col].fillna(0.0)
    df["timestamp_ny"] = df["timestamp"].dt.tz_convert(SESSION_TZ)
    df["session_date"] = df["timestamp_ny"].dt.strftime("%Y-%m-%d")
    df["bar_in_session"] = 0
    df["row_open_skip"] = False
    df["session_minutes"] = 1
    df["session_progress"] = 1.0
    return df
"""


INFER_CELL7 = """raw_df_utc, api_calls = fetch_bars_alpaca(SYMBOL, LOOKBACK_DAYS)
session_df = prepare_daily_bars(raw_df_utc)

print(
    {
        "symbol": SYMBOL,
        "rows_raw": len(raw_df_utc),
        "rows_daily": len(session_df),
        "trading_days": int(session_df["session_date"].nunique()),
        "last_visible_timestamp": str(session_df["timestamp"].iloc[-1]),
        "api_calls": api_calls,
    }
)
display(session_df.tail())
"""


INFER_CELL15 = """## Daily Rolling Backtest

This section replays a strictly causal rolling daily window using the frozen expert artifacts. For each anchor day, every expert predicts from its own feature block, the saved ensemble weights form the aggregate OHLC path, and the individual expert paths are shown as close-line overlays.
"""


INFER_CELL16 = """FINAL_ROLLING_FRAME_OUTPUT_DIR = FINAL_RUN_DIR / "rolling_frames"
MAX_ROLLING_LOOKBACK = max(int(MODEL_ARTIFACTS[expert_name]["lookback"]) for expert_name in EXPECTED_EXPERTS)

print(
    {
        "final_rolling_window": FINAL_ROLLING_WINDOW,
        "final_rolling_step": FINAL_ROLLING_STEP,
        "final_rolling_backtest_date": FINAL_ROLLING_BACKTEST_DATE,
        "final_rolling_history_bars": FINAL_ROLLING_HISTORY_BARS,
        "final_rolling_save_frames": FINAL_ROLLING_SAVE_FRAMES,
        "final_rolling_device": str(FINAL_ROLLING_DEVICE),
        "final_rolling_clear_every": FINAL_ROLLING_CLEAR_EVERY,
        "max_rolling_lookback": MAX_ROLLING_LOOKBACK,
    }
)
"""


INFER_CELL17 = """@dataclass
class FinalRollingLog:
    anchor_index: int
    anchor_timestamp: pd.Timestamp
    future_timestamps: List[pd.Timestamp]
    anchor_prev_close: float
    actual_path: np.ndarray
    aggregate_path: np.ndarray
    expert_paths: Dict[str, np.ndarray]
    aggregate_regime_indicator: float
    aggregate_soft_swing_cap: Optional[float]
    aggregate_t1_temporal_cap: Optional[float]
    aggregate_t1_direction_vote: float
    aggregate_t1_momentum_sign: float
    temperatures: Dict[str, float]


def rolling_metric_steps() -> List[int]:
    return sorted({1, min(3, HORIZON), min(5, HORIZON), HORIZON})


def select_final_rolling_backtest_date(session_df: pd.DataFrame, requested: Optional[str], max_lookback: int, horizon: int) -> str:
    last_anchor = len(session_df) - horizon
    if last_anchor <= max_lookback:
        raise RuntimeError("Not enough rows for a daily rolling backtest.")

    if requested is not None:
        requested_positions = np.where(session_df["session_date"].to_numpy() == requested)[0]
        if len(requested_positions) == 0:
            raise ValueError(f"Requested backtest start date not found: {requested}")
        first_anchor = int(requested_positions[0])
        if first_anchor < max_lookback:
            raise ValueError(f"Backtest start date {requested} does not have enough prior bars for lookback={max_lookback}.")
        if first_anchor > last_anchor:
            raise ValueError(f"Backtest start date {requested} leaves no room for horizon={horizon}.")
        return requested

    first_anchor = max(max_lookback, last_anchor - FINAL_ROLLING_WINDOW + 1)
    return pd.Timestamp(session_df["timestamp"].iloc[first_anchor]).tz_convert(SESSION_TZ).strftime("%Y-%m-%d")


def build_final_rolling_anchor_indices(session_df: pd.DataFrame, backtest_date: str, max_lookback: int, horizon: int, step: int) -> np.ndarray:
    positions = np.where(session_df["session_date"].to_numpy() >= backtest_date)[0]
    if len(positions) == 0:
        raise RuntimeError(f"No daily positions found for {backtest_date}")
    first_anchor = max(int(positions[0]), max_lookback)
    last_anchor = len(session_df) - horizon
    if first_anchor > last_anchor:
        raise RuntimeError(f"No valid daily rolling anchors for {backtest_date}")
    return np.arange(first_anchor, last_anchor + 1, step, dtype=np.int32)


def build_actual_paths_for_anchor_indices(feature_df: pd.DataFrame, anchor_indices: np.ndarray) -> Dict[str, Any]:
    timestamps: List[pd.Timestamp] = []
    future_timestamps: List[List[pd.Timestamp]] = []
    prev_close: List[float] = []
    actual_paths: List[np.ndarray] = []

    for anchor_index in anchor_indices:
        timestamps.append(pd.Timestamp(feature_df["timestamp"].iloc[anchor_index]))
        prev_close_value = float(feature_df["prev_close"].iloc[anchor_index])
        prev_close.append(prev_close_value)
        future_slice = feature_df.iloc[anchor_index : anchor_index + HORIZON]
        future_timestamps.append([pd.Timestamp(ts) for ts in future_slice["timestamp"].tolist()])
        actual_returns = future_slice.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float32)
        actual_paths.append(returns_to_prices_seq(prev_close_value, actual_returns))

    return {
        "anchor_timestamps": timestamps,
        "future_timestamps": future_timestamps,
        "prev_close": np.asarray(prev_close, dtype=np.float32),
        "actual_paths": np.stack(actual_paths).astype(np.float32),
    }


def run_final_rolling_for_bundle(expert_name: str, anchor_indices: np.ndarray) -> Dict[str, Any]:
    bundle = load_expert_bundle(expert_name, device_override=FINAL_ROLLING_DEVICE)
    feature_df = FEATURE_FRAMES[bundle.feature_mode]
    predicted_paths: List[np.ndarray] = []
    regime_indicators: List[float] = []
    temperatures: List[float] = []

    for offset, anchor_index in enumerate(tqdm(anchor_indices, desc=f"{bundle.version} final rolling")):
        model_input = build_anchor_input(feature_df, bundle, int(anchor_index))
        regime_indicator = 0.0
        regime_multiplier = 1.0
        if bundle.feature_mode == "regime":
            recent_history = feature_df.iloc[max(0, int(anchor_index) - FINAL_AGGREGATE_GUARD_LOOKBACK) : int(anchor_index)].copy()
            _, regime_multiplier, regime_indicator = detect_regime_multiplier(recent_history)

        base_temperature = float(bundle.inference_config["sampling_temperature"])
        effective_temperature = base_temperature

        predicted_path = generate_ensemble_with_trend_selection(
            bundle=bundle,
            model_input=model_input,
            temperature=effective_temperature,
            regime_multiplier=regime_multiplier,
        )

        predicted_paths.append(predicted_path)
        regime_indicators.append(float(regime_indicator))
        temperatures.append(float(effective_temperature * regime_multiplier))

        if (offset + 1) % FINAL_ROLLING_CLEAR_EVERY == 0:
            clear_torch_memory()

    result = {
        "expert": expert_name,
        "version": bundle.version,
        "lookback": bundle.lookback,
        "feature_mode": bundle.feature_mode,
        "architecture": bundle.architecture,
        "predicted_paths": np.stack(predicted_paths).astype(np.float32),
        "regime_indicators": np.asarray(regime_indicators, dtype=np.float32),
        "temperatures": np.asarray(temperatures, dtype=np.float32),
    }

    bundle.model = bundle.model.to("cpu")
    del bundle.model
    clear_torch_memory()
    return result


def build_final_rolling_logs(
    anchor_indices: np.ndarray,
    actual_payload: Dict[str, Any],
    expert_outputs: Dict[str, Dict[str, Any]],
) -> List[FinalRollingLog]:
    logs: List[FinalRollingLog] = []
    previous_aggregate_next_close: Optional[float] = None
    for offset, anchor_index in enumerate(anchor_indices):
        aggregate_raw = np.zeros((HORIZON, 4), dtype=np.float32)
        aggregate_regime_indicator = 0.0
        expert_paths: Dict[str, np.ndarray] = {}
        temperatures: Dict[str, float] = {}
        for expert_name in EXPECTED_EXPERTS:
            expert_path = expert_outputs[expert_name]["predicted_paths"][offset]
            expert_paths[expert_name] = expert_path
            temperatures[expert_name] = float(expert_outputs[expert_name]["temperatures"][offset])
            aggregate_raw += float(ensemble_weights[expert_name]) * expert_path
            aggregate_regime_indicator += float(ensemble_weights[expert_name]) * float(expert_outputs[expert_name]["regime_indicators"][offset])
        aggregate_guard_history = build_guard_history_slice(FEATURE_FRAMES["technical"], int(anchor_index), FINAL_AGGREGATE_GUARD_LOOKBACK)
        aggregate_path, aggregate_soft_swing_cap = postprocess_aggregate_path(
            aggregate_raw,
            float(actual_payload["prev_close"][offset]),
            aggregate_guard_history,
        )
        aggregate_path, aggregate_t1_temporal_cap, aggregate_t1_direction_vote, aggregate_t1_momentum_sign = apply_t1_temporal_direction_guard(
            aggregate_path,
            float(actual_payload["prev_close"][offset]),
            aggregate_guard_history,
            expert_paths,
            ensemble_weights,
            previous_next_close=previous_aggregate_next_close,
        )
        previous_aggregate_next_close = float(aggregate_path[0, 3])

        logs.append(
            FinalRollingLog(
                anchor_index=int(anchor_index),
                anchor_timestamp=actual_payload["anchor_timestamps"][offset],
                future_timestamps=actual_payload["future_timestamps"][offset],
                anchor_prev_close=float(actual_payload["prev_close"][offset]),
                actual_path=actual_payload["actual_paths"][offset],
                aggregate_path=aggregate_path,
                expert_paths=expert_paths,
                aggregate_regime_indicator=float(aggregate_regime_indicator),
                aggregate_soft_swing_cap=aggregate_soft_swing_cap,
                aggregate_t1_temporal_cap=aggregate_t1_temporal_cap,
                aggregate_t1_direction_vote=float(aggregate_t1_direction_vote),
                aggregate_t1_momentum_sign=float(aggregate_t1_momentum_sign),
                temperatures=temperatures,
            )
        )
    return logs


def summarize_final_rolling_logs(logs: List[FinalRollingLog]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, Any]] = []
    next_close_rows: List[Dict[str, Any]] = []
    metric_steps = rolling_metric_steps()

    for log in logs:
        pred_close = log.aggregate_path[:, 3]
        actual_close = log.actual_path[:, 3]
        direction_hit = bool(np.sign(pred_close[0] - log.anchor_prev_close) == np.sign(actual_close[0] - log.anchor_prev_close))
        row = {
            "anchor_timestamp": log.anchor_timestamp,
            "anchor_prev_close": log.anchor_prev_close,
            "aggregate_regime_indicator": float(log.aggregate_regime_indicator),
            "aggregate_regime_name": regime_name_from_indicator(log.aggregate_regime_indicator),
            "aggregate_soft_swing_cap": float(log.aggregate_soft_swing_cap) if log.aggregate_soft_swing_cap is not None else np.nan,
            "aggregate_t1_temporal_cap": float(log.aggregate_t1_temporal_cap) if log.aggregate_t1_temporal_cap is not None else np.nan,
            "aggregate_t1_direction_vote": float(log.aggregate_t1_direction_vote),
            "aggregate_t1_momentum_sign": float(log.aggregate_t1_momentum_sign),
            "direction_hit": float(direction_hit),
            "path_mae": float(np.mean(np.abs(pred_close - actual_close))),
        }
        for step in metric_steps:
            row[f"step_{step}_close_mae"] = float(abs(pred_close[step - 1] - actual_close[step - 1]))
        summary_rows.append(row)

        next_close_row = {
            "anchor_timestamp": log.anchor_timestamp,
            "actual_next_close": float(actual_close[0]),
            "aggregate_next_close": float(pred_close[0]),
        }
        for expert_name, expert_path in log.expert_paths.items():
            next_close_row[f"{expert_name}_next_close"] = float(expert_path[0, 3])
        next_close_rows.append(next_close_row)

    summary_df = pd.DataFrame(summary_rows)
    next_close_df = pd.DataFrame(next_close_rows)
    metrics_rows = [
        {"metric": "prediction_count", "value": int(len(summary_df))},
        {"metric": "directional_hit_rate_t1", "value": float(summary_df["direction_hit"].mean())},
        {"metric": "aggregate_path_mae", "value": float(summary_df["path_mae"].mean())},
    ]
    for step in metric_steps:
        metrics_rows.append({"metric": f"aggregate_step_{step}_close_mae", "value": float(summary_df[f"step_{step}_close_mae"].mean())})
    metrics_df = pd.DataFrame(metrics_rows)
    return summary_df, metrics_df, next_close_df


def _rolling_candle_frame(ohlc_values: np.ndarray, timestamps: Sequence[pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame(ohlc_values, columns=["Open", "High", "Low", "Close"], index=pd.Index(timestamps))


def _draw_rolling_candles(
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
    zorder: int = 3,
) -> None:
    values = ohlc_df[["Open", "High", "Low", "Close"]].to_numpy(dtype=np.float32)
    for idx, (open_price, high_price, low_price, close_price) in enumerate(values):
        x_value = start_x + idx
        bull = close_price >= open_price
        ax.vlines(x_value, low_price, high_price, color=wick_color, linewidth=lw, alpha=alpha, zorder=zorder - 1)
        body_low = min(open_price, close_price)
        body_height = max(abs(close_price - open_price), 1e-6)
        rect = Rectangle(
            (x_value - width / 2, body_low),
            width,
            body_height,
            facecolor=up_face if bull else down_face,
            edgecolor=up_edge if bull else down_edge,
            linewidth=lw,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(rect)


def render_final_rolling_frame(log: FinalRollingLog, pricedf: pd.DataFrame, history_bars: int = FINAL_ROLLING_HISTORY_BARS) -> plt.Figure:
    history_start = max(0, log.anchor_index - history_bars)
    history_df = pricedf.iloc[history_start : log.anchor_index][["timestamp", "open", "high", "low", "close"]].copy()
    history_ohlc = history_df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}).set_index("timestamp")
    actual_ohlc = _rolling_candle_frame(log.actual_path, log.future_timestamps)
    aggregate_ohlc = _rolling_candle_frame(log.aggregate_path, log.future_timestamps)

    fig, ax = plt.subplots(figsize=(18, 8), facecolor="black")
    FigureCanvasAgg(fig)
    ax.set_facecolor("black")

    _draw_rolling_candles(
        ax,
        history_ohlc,
        0,
        up_edge="#00FF00",
        up_face="#00FF00",
        down_edge="#FF0000",
        down_face="#FF0000",
        wick_color="#D0D0D0",
        width=0.60,
        lw=1.0,
        alpha=0.95,
        zorder=3,
    )

    future_start_x = len(history_ohlc)
    _draw_rolling_candles(
        ax,
        actual_ohlc,
        future_start_x,
        up_edge="#1D6F42",
        up_face="#1D6F42",
        down_edge="#8E2F25",
        down_face="#8E2F25",
        wick_color="#8E8E8E",
        width=0.58,
        lw=0.9,
        alpha=0.40,
        zorder=2,
    )
    _draw_rolling_candles(
        ax,
        aggregate_ohlc,
        future_start_x,
        up_edge="#FFFFFF",
        up_face="#FFFFFF",
        down_edge="#FFFFFF",
        down_face="#000000",
        wick_color="#F3F3F3",
        width=0.50,
        lw=1.2,
        alpha=1.0,
        zorder=4,
    )

    color_map = plt.cm.tab10(np.linspace(0, 1, max(len(log.expert_paths), 1)))
    line_handles = []
    for idx, expert_name in enumerate(EXPECTED_EXPERTS):
        expert_path = log.expert_paths[expert_name]
        close_values = expert_path[:, 3]
        x_values = np.arange(len(close_values)) + future_start_x
        handle, = ax.plot(
            x_values,
            close_values,
            color=color_map[idx % len(color_map)],
            linewidth=1.15,
            alpha=0.85,
            zorder=5,
            label=expert_name,
        )
        line_handles.append(handle)

    now_x = len(history_ohlc) - 0.5
    ax.axvline(now_x, color="white", linestyle="--", linewidth=1.0, alpha=0.85, zorder=6)

    full_index = history_ohlc.index.append(actual_ohlc.index)
    tick_step = max(1, len(full_index) // 10)
    ticks = list(range(0, len(full_index), tick_step))
    if ticks[-1] != len(full_index) - 1:
        ticks.append(len(full_index) - 1)
    labels = [pd.Timestamp(full_index[i]).tz_convert(SESSION_TZ).strftime("%Y-%m-%d") for i in ticks]

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=25, ha="right", color="white", fontsize=9)
    ax.tick_params(axis="y", colors="white")
    for spine in ax.spines.values():
        spine.set_color("#666666")
    ax.grid(color="#242424", linewidth=0.6, alpha=0.35)

    header = (
        f"{SYMBOL} 1D Final Rolling | Anchor: {log.anchor_timestamp.tz_convert(SESSION_TZ).strftime('%Y-%m-%d')} | "
        f"Regime: {regime_name_from_indicator(log.aggregate_regime_indicator)}"
    )
    ax.set_title(header, color="white", pad=12)
    ax.set_ylabel("Price", color="white")

    legend_handles = [
        Patch(facecolor="#00FF00", edgecolor="#00FF00", label="History (bull)"),
        Patch(facecolor="#FF0000", edgecolor="#FF0000", label="History (bear)"),
        Patch(facecolor="#1D6F42", edgecolor="#1D6F42", label="Actual Future (dim bull)"),
        Patch(facecolor="#8E2F25", edgecolor="#8E2F25", label="Actual Future (dim bear)"),
        Patch(facecolor="#FFFFFF", edgecolor="#FFFFFF", label="Aggregate (bull)"),
        Patch(facecolor="#000000", edgecolor="#FFFFFF", label="Aggregate (bear)"),
    ] + line_handles
    legend = ax.legend(handles=legend_handles, facecolor="black", edgecolor="#666666", loc="upper left", ncol=2)
    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout()
    return fig


def generate_final_rolling_frames(logs: List[FinalRollingLog], pricedf: pd.DataFrame, output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    for idx, log in enumerate(tqdm(logs, desc="Saving final rolling frames")):
        fig = render_final_rolling_frame(log, pricedf, history_bars=FINAL_ROLLING_HISTORY_BARS)
        output_path = output_dir / f"frame_{idx:04d}.png"
        fig.savefig(output_path, dpi=180, facecolor="black", bbox_inches="tight")
        saved_paths.append(output_path)
        plt.close(fig)
    return saved_paths
"""


INFER_CELL18 = """FINAL_ROLLING_BACKTEST_DATE = select_final_rolling_backtest_date(
    session_df=session_df,
    requested=FINAL_ROLLING_BACKTEST_DATE,
    max_lookback=MAX_ROLLING_LOOKBACK,
    horizon=HORIZON,
)
FINAL_ROLLING_ANCHOR_INDICES = build_final_rolling_anchor_indices(
    session_df=session_df,
    backtest_date=FINAL_ROLLING_BACKTEST_DATE,
    max_lookback=MAX_ROLLING_LOOKBACK,
    horizon=HORIZON,
    step=FINAL_ROLLING_STEP,
)

FINAL_ROLLING_ACTUAL = build_actual_paths_for_anchor_indices(FEATURE_FRAMES["core"], FINAL_ROLLING_ANCHOR_INDICES)

FINAL_ROLLING_EXPERT_OUTPUTS: Dict[str, Dict[str, Any]] = {}
for expert_name in EXPECTED_EXPERTS:
    FINAL_ROLLING_EXPERT_OUTPUTS[expert_name] = run_final_rolling_for_bundle(expert_name, FINAL_ROLLING_ANCHOR_INDICES)

FINAL_ROLLING_LOGS = build_final_rolling_logs(
    anchor_indices=FINAL_ROLLING_ANCHOR_INDICES,
    actual_payload=FINAL_ROLLING_ACTUAL,
    expert_outputs=FINAL_ROLLING_EXPERT_OUTPUTS,
)

FINAL_ROLLING_SUMMARY_DF, FINAL_ROLLING_METRICS_DF, FINAL_ROLLING_NEXT_CLOSE_DF = summarize_final_rolling_logs(FINAL_ROLLING_LOGS)

print(
    {
        "rolling_backtest_start_date": FINAL_ROLLING_BACKTEST_DATE,
        "rolling_prediction_count": len(FINAL_ROLLING_LOGS),
        "first_anchor": str(FINAL_ROLLING_LOGS[0].anchor_timestamp),
        "last_anchor": str(FINAL_ROLLING_LOGS[-1].anchor_timestamp),
    }
)
display(FINAL_ROLLING_METRICS_DF)
display(FINAL_ROLLING_SUMMARY_DF.head())

plt.figure(figsize=(14, 5))
plt.plot(FINAL_ROLLING_NEXT_CLOSE_DF["anchor_timestamp"], FINAL_ROLLING_NEXT_CLOSE_DF["actual_next_close"], color="black", linewidth=2.6, label="actual_next_close")
plt.plot(FINAL_ROLLING_NEXT_CLOSE_DF["anchor_timestamp"], FINAL_ROLLING_NEXT_CLOSE_DF["aggregate_next_close"], color="#1f77b4", linewidth=2.0, label="aggregate_next_close")
for expert_name in EXPECTED_EXPERTS:
    plt.plot(
        FINAL_ROLLING_NEXT_CLOSE_DF["anchor_timestamp"],
        FINAL_ROLLING_NEXT_CLOSE_DF[f"{expert_name}_next_close"],
        linewidth=1.0,
        alpha=0.65,
        label=expert_name,
    )
plt.title(f"{SYMBOL} Daily Rolling Backtest (t+1 Close)")
plt.xlabel("Anchor Timestamp")
plt.ylabel("Close Price")
plt.grid(alpha=0.2)
plt.legend(ncol=3)
plt.tight_layout()
plt.show()

preview_index = FINAL_ROLLING_DISPLAY_FRAME_INDEX if FINAL_ROLLING_DISPLAY_FRAME_INDEX >= 0 else len(FINAL_ROLLING_LOGS) - 1
preview_index = max(0, min(preview_index, len(FINAL_ROLLING_LOGS) - 1))
FINAL_ROLLING_PREVIEW_FIG = render_final_rolling_frame(
    FINAL_ROLLING_LOGS[preview_index],
    pricedf=session_df,
    history_bars=FINAL_ROLLING_HISTORY_BARS,
)
plt.show()

FINAL_ROLLING_SAVED_FRAMES: List[Path] = []
if FINAL_ROLLING_SAVE_FRAMES:
    FINAL_ROLLING_SAVED_FRAMES = generate_final_rolling_frames(
        logs=FINAL_ROLLING_LOGS,
        pricedf=session_df,
        output_dir=FINAL_ROLLING_FRAME_OUTPUT_DIR,
    )
    print(
        {
            "rolling_frames_dir": str(FINAL_ROLLING_FRAME_OUTPUT_DIR),
            "rolling_frames_saved": len(FINAL_ROLLING_SAVED_FRAMES),
        }
    )
"""


def build_final1d_train() -> dict:
    nb = load_notebook(BASE_TRAIN)
    set_source(nb, 0, TRAIN_CELL0)
    set_source(nb, 3, TRAIN_CELL3)
    set_source(nb, 5, TRAIN_CELL5)
    set_source(nb, 6, TRAIN_CELL6)
    set_source(nb, 11, TRAIN_CELL11)
    set_source(nb, 12, TRAIN_CELL12)
    set_source(nb, 15, TRAIN_CELL15)

    cell4 = get_source(nb, 4)
    cell4 = replace_or_raise(
        cell4,
        '        "horizon": HORIZON,\n    }\n',
        '        "horizon": HORIZON,\n        "timeframe": "1D",\n        "rolling_eval_bars": DAILY_ROLLING_EVAL_BARS,\n    }\n',
    )
    set_source(nb, 4, cell4)

    cell7 = get_source(nb, 7)
    cell7 = replace_or_raise(
        cell7,
        '    df["prev_close"] = df["close"].shift(1)\n    session_change = df["session_date"] != df["session_date"].shift(1)\n    df.loc[session_change, "prev_close"] = df.loc[session_change, "open"]\n',
        '    df["prev_close"] = df["close"].shift(1)\n    if len(df) > 0:\n        df.loc[df.index[0], "prev_close"] = df.loc[df.index[0], "open"]\n',
    )
    set_source(nb, 7, cell7)

    nb["metadata"]["title"] = "Final1DTrain.ipynb"
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return nb


def build_final1d_infer() -> dict:
    nb = load_notebook(BASE_INFER)
    set_source(nb, 0, INFER_CELL0)
    set_source(nb, 3, INFER_CELL3)
    set_source(nb, 5, INFER_CELL5)
    set_source(nb, 6, INFER_CELL6)
    set_source(nb, 7, INFER_CELL7)
    set_source(nb, 15, INFER_CELL15)
    set_source(nb, 16, INFER_CELL16)
    set_source(nb, 17, INFER_CELL17)
    set_source(nb, 18, INFER_CELL18)

    cell8 = get_source(nb, 8)
    cell8 = replace_or_raise(
        cell8,
        '    df["prev_close"] = df["close"].shift(1)\n    session_change = df["session_date"] != df["session_date"].shift(1)\n    df.loc[session_change, "prev_close"] = df.loc[session_change, "open"]\n',
        '    df["prev_close"] = df["close"].shift(1)\n    if len(df) > 0:\n        df.loc[df.index[0], "prev_close"] = df.loc[df.index[0], "open"]\n',
    )
    set_source(nb, 8, cell8)

    cell9 = get_source(nb, 9)
    cell9 = re.sub(
        r"def intraday_temperature\(anchor_ts: pd\.Timestamp\) -> float:\n(?:    .*\n)+?\n\n",
        "def intraday_temperature(anchor_ts: pd.Timestamp) -> float:\n    return 1.0\n\n\n",
        cell9,
        flags=re.MULTILINE,
    )
    set_source(nb, 9, cell9)

    cell11 = get_source(nb, 11)
    cell11 = cell11.replace("recent_history = feature_df.iloc[max(0, len(feature_df) - 390) :].copy()", "recent_history = feature_df.iloc[max(0, len(feature_df) - FINAL_AGGREGATE_GUARD_LOOKBACK) :].copy()")
    cell11 = cell11.replace("previous_history = feature_df.iloc[max(0, len(feature_df) - 1 - 390) : len(feature_df) - 1].copy()", "previous_history = feature_df.iloc[max(0, len(feature_df) - 1 - FINAL_AGGREGATE_GUARD_LOOKBACK) : len(feature_df) - 1].copy()")
    cell11 = cell11.replace("time_temperature = intraday_temperature(model_input[\"anchor_timestamp\"])\n    effective_temperature = base_temperature * (time_temperature / 1.5)", "effective_temperature = base_temperature")
    cell11 = cell11.replace("previous_time_temperature = intraday_temperature(previous_model_input[\"anchor_timestamp\"])\n        previous_effective_temperature = base_temperature * (previous_time_temperature / 1.5)", "previous_effective_temperature = base_temperature")
    set_source(nb, 11, cell11)

    cell14 = get_source(nb, 14)
    cell14 = cell14.replace('plt.title(f"{SYMBOL} Forecast Close Paths")', 'plt.title(f"{SYMBOL} 1D Forecast Close Paths")')
    cell14 = cell14.replace('plot_aggregate_candles(aggregate_path, f"{SYMBOL} Aggregate OHLC Forecast")', 'plot_aggregate_candles(aggregate_path, f"{SYMBOL} Aggregate 1D OHLC Forecast")')
    set_source(nb, 14, cell14)

    cell19 = get_source(nb, 19)
    cell19 = cell19.replace('"rolling_backtest_date": FINAL_ROLLING_BACKTEST_DATE,', '"rolling_backtest_start_date": FINAL_ROLLING_BACKTEST_DATE,')
    set_source(nb, 19, cell19)

    nb["metadata"]["title"] = "FINAL1D.ipynb"
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return nb


def validate_notebook(path: Path) -> None:
    nb = load_notebook(path)
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"{path.name}::cell_{idx}", "exec")


def main() -> None:
    train_nb = build_final1d_train()
    infer_nb = build_final1d_infer()
    save_notebook(OUT_TRAIN, train_nb)
    save_notebook(OUT_INFER, infer_nb)
    validate_notebook(OUT_TRAIN)
    validate_notebook(OUT_INFER)
    print(json.dumps({"created": [str(OUT_TRAIN), str(OUT_INFER)]}, indent=2))


if __name__ == "__main__":
    main()
