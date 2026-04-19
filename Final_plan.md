# FINAL.ipynb Plan

No `FINAL.ipynb` exists in the workspace right now, so this plan assumes we are building a new notebook from scratch.

## Goal

`FINAL.ipynb` should be an **inference-only** notebook that:

- loads already-trained artifacts from `FinalTrain.ipynb`
- runs each expert model separately
- shows each model's prediction clearly
- builds one aggregated final forecast
- runs the `v9.6` decision layer on top of the aggregated forecast
- does **zero training** inside the notebook

## Intended workflow

The intended production workflow is:

1. run `FinalTrain.ipynb` once
2. save all trained weights, scaler stats, configs, feature manifests, ensemble weights, and PPO artifacts to disk
3. open `FINAL.ipynb`
4. load those saved artifacts
5. run inference for each forecast expert
6. aggregate the expert forecasts
7. run the saved `v9.6` decision layer on the aggregate

So yes, the final notebook is supposed to do:

- loading
- feature building
- inference
- aggregation
- display / export

and **not**:

- training
- retraining
- tuning
- sweeping
- saving updated model weights

If trained artifacts are missing, `FINAL.ipynb` should fail early instead of silently retraining anything.

## Final stack design

The cleanest design is a **two-layer stack**:

### Layer 1: Forecast experts

These are the models that should each produce their own OHLC path prediction:

- `v8.5`
- `v9.1`
- `v9.2`
- `v9.3`
- `v9.5`

### Layer 2: Decision expert

- `v9.6`

Important design note:

`v9.6` is not really a sixth OHLC forecaster. It is a PPO trading-policy layer. So it should **not** be forced into the same price-path average as the other five models. Instead:

- the five forecast experts produce OHLC predictions
- those are aggregated into one final price forecast
- `v9.6` consumes the aggregated forecast plus market/portfolio state and outputs action/confidence

That keeps the design faithful to what each notebook actually does.

## Fixed expert configurations

No lookback sweep should run in the final notebook. Use the best-selected settings from the learning docs.

| Expert | Role | Fixed lookback | Main feature set | Main architecture | Key notes |
|---|---|---:|---|---|---|
| `v8.5` | base path expert | `64` | `10` core features | attention GRU | simplest baseline expert |
| `v9.1` | technical-indicator expert | `160` | `42` engineered features | attention GRU | broad technical-input view |
| `v9.2` | regime-aware expert | `256` | core + regime features | attention GRU | use dynamic regime wrapper, not the constant-label bug |
| `v9.3` | hybrid encoder expert | `96` | `10` core features | iTransformer + GRU | keep low-VRAM settings |
| `v9.5` | retrieval expert | `256` | `10` core features | attention GRU + pattern memory | use corrected retrieval artifact from training |
| `v9.6` | decision expert | N/A | aggregate forecast + market + portfolio state | PPO actor-critic | outputs action/confidence, not OHLC path |

Shared forecasting settings that should remain fixed unless we later decide to revise all experts together:

- `LOOKBACK_DAYS = 120`
- `HORIZON = 50`
- `HIDDEN_SIZE = 256`
- `NUM_LAYERS = 2`
- `DROPOUT = 0.20`
- `SAMPLING_TEMPERATURE = 1.5`
- `TREND_LOOKBACK_BARS = 20`
- `STRONG_TREND_THRESHOLD = 0.002`
- `MIN_PREDICTED_VOL = 0.0001`

`v9.3` should keep its own special runtime settings:

- `batch_size = 32`
- AMP-enabled inference path if supported
- low-VRAM-safe execution order

## Aggregation design

The final price forecast should be a **weighted ensemble of the five forecast experts**.

### What each forecast expert returns

Each expert should produce:

- one selected forecast path of shape `HORIZON x 4`
- path in absolute OHLC prices
- metadata:
  - model name
  - selected lookback
  - feature set used
  - ensemble size used
  - forecast timestamp
  - saved reliability weight

### Aggregation rule

Use a **static reliability-weighted average** loaded from training artifacts:

`final_path[t, c] = sum_i weight_i * expert_i_path[t, c]`

Where:

- `i` is one of the five forecast experts
- `t` is horizon step
- `c` is one of `Open, High, Low, Close`

After averaging:

- enforce candle validity on the aggregate path
- store both raw aggregate and candle-corrected aggregate

Why weighted average instead of a new learned meta-model:

- it is simpler
- it avoids adding another training dependency inside `FINAL.ipynb`
- it lets us see each model contribution directly
- it fits the user's requirement that final notebook should only load and infer

## How `v9.6` should be used

`v9.6` should run **after** the price aggregation step.

Recommended input to the PPO policy:

- the aggregated `50 x 4` forecast path
- current market features
- current portfolio state
- regime indicator

Recommended outputs to display:

- action in `[-1, 1]`
- implied directional stance: short / neutral / long
- confidence score
- regime state

So the final notebook should produce two final outputs:

1. `final_aggregated_price_forecast`
2. `final_trade_signal_from_v9.6`

## What the notebook should show

At minimum, `FINAL.ipynb` should display:

- one table with every expert and its saved configuration
- one table with every expert's next-step and horizon-end forecast
- one plot overlaying:
  - each expert close path
  - final aggregated close path
- one candle chart for the final aggregated OHLC path
- one small panel for `v9.6`:
  - action
  - confidence
  - regime
  - recommended position size

The user should be able to answer both questions immediately:

- "What did each model predict?"
- "What is the final combined view?"

## Notebook section plan

Recommended notebook flow:

1. **Title / run intent**
   Explain that this notebook is inference-only and loads frozen artifacts.

2. **Imports and device setup**
   Include CUDA checks, but do not allocate models yet.

3. **Artifact paths and manifest loading**
   Load the global artifact manifest, model configs, scalers, feature manifests, and ensemble weights.

4. **Fetch latest data**
   Pull the latest market data using the shared production-safe data pipeline.

5. **Sessionize and build per-expert features**
   Build only the feature blocks needed by each expert.

6. **Run forecast experts one at a time**
   Load one expert, infer, move prediction to CPU, unload, clear cache, repeat.

7. **Assemble comparison tables**
   Build a per-model summary frame and a per-step forecast cache.

8. **Aggregate the five forecast experts**
   Apply saved ensemble weights, enforce candle validity, save final path.

9. **Run `v9.6` policy on the aggregated path**
   Output trade/action metadata.

10. **Visualization**
    Show individual experts, aggregated result, and policy output.

11. **Export block**
    Optional save of the current run output, but no model-saving or retraining.

## Artifact contract

`FINAL.ipynb` should assume `FinalTrain.ipynb` has already produced:

- one checkpoint bundle per expert
- scaler stats per expert
- feature manifest per expert
- exact selected hyperparameters per expert
- inference config per expert
- validation metrics per expert
- ensemble weights
- PPO policy weights and env config
- one top-level manifest file pointing to all artifacts

If any required artifact is missing, the notebook should fail early with a readable message.

## VRAM / RAM rules

The final notebook must fit inside:

- `8 GB` VRAM
- `32 GB` system RAM

Recommended runtime rules:

- keep raw data, engineered features, and prediction tables in CPU RAM
- load only **one forecast model at a time** onto GPU
- after each expert finishes:
  - move outputs to CPU
  - delete model object
  - call GPU cache cleanup
- keep the `v9.5` retrieval database on CPU memory, not GPU memory
- run the PPO policy only after all forecasting models are finished
- do not keep all expert models resident on GPU simultaneously

This execution order should be:

1. `v8.5`
2. unload
3. `v9.1`
4. unload
5. `v9.2`
6. unload
7. `v9.3`
8. unload
9. `v9.5`
10. unload
11. aggregate on CPU
12. run `v9.6`

## Design decisions carried forward from the learnings

- Use each model's best-selected lookback, not rolling-default mismatches from earlier notebooks.
- Keep `v9.3` as the only hybrid encoder model.
- Keep `v9.5` only if the training notebook fixes its untrained retrieval issue.
- Use `v9.6` as a decision layer on top of the aggregate, not as a fake sixth OHLC generator.
- Do not run lookback sweeps inside either final inference or final training production flow.

## Non-goals for FINAL.ipynb

- no hyperparameter sweeps
- no model retraining
- no artifact mutation
- no walk-forward re-evaluation
- no backtest generation beyond optional inference diagnostics

## End state

When this notebook is done, opening `FINAL.ipynb` should let us:

- load all frozen artifacts
- produce five expert forecasts
- inspect each expert individually
- produce one aggregated final price forecast
- produce one `v9.6` trade/action output

That gives us a clean production split:

- `FinalTrain.ipynb` does all heavy work once
- `FINAL.ipynb` only loads and infers
