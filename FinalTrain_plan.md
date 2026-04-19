# FinalTrain.ipynb Plan

No `FinalTrain.ipynb` exists in the workspace right now, so this plan assumes we are building a new notebook from scratch.

## Goal

`FinalTrain.ipynb` should be the **only** notebook that performs training. Its job is to:

- train every forecast expert once using fixed best-known settings
- save everything needed for later inference
- compute ensemble weights
- train the `v9.6` decision layer after forecast artifacts exist
- produce a single artifact manifest that `FINAL.ipynb` can load

The result should be that `FINAL.ipynb` never trains anything.

## Intended workflow

The intended workflow is:

1. run `FinalTrain.ipynb` one time to build the production artifact set
2. save everything needed for later reuse
3. stop using the training notebook during normal prediction runs
4. use `FINAL.ipynb` only for loading those saved artifacts and running inference

In other words:

- `FinalTrain.ipynb` is the offline build step
- `FINAL.ipynb` is the online inference step

`FinalTrain.ipynb` should only be rerun when we intentionally want to retrain, refresh data, or rebuild artifacts.

## Core design principle

Use a **single frozen data snapshot and a single artifact schema** for all experts.

That is important because:

- all experts need comparable evaluation
- ensemble weights only make sense if models were trained and validated on aligned data
- the RL layer should consume forecasts generated from the exact saved experts

## Fixed expert training set

No lookback sweep should run in `FinalTrain.ipynb`. Use the best-selected settings from the learning docs.

| Expert | Fixed lookback | Feature block | Architecture | Training notes |
|---|---:|---|---|---|
| `v8.5` | `64` | `10` core features | attention GRU | direct baseline expert |
| `v9.1` | `160` | `42` technical-indicator features | attention GRU | full technical feature expert |
| `v9.2` | `256` | core + regime features | attention GRU | keep regime wrapper; do not preserve constant regime-label bug |
| `v9.3` | `96` | `10` core features | iTransformer + GRU | must use low-VRAM settings |
| `v9.5` | `256` | `10` core features | attention GRU + retrieval memory | retrieval artifact must be fixed during training |
| `v9.6` | N/A | aggregate forecast state | PPO actor-critic | trained after forecast ensemble exists |

Shared forecasting hyperparameters that should remain fixed across compatible models:

- `LOOKBACK_DAYS = 120`
- `HORIZON = 50`
- `HIDDEN_SIZE = 256`
- `NUM_LAYERS = 2`
- `DROPOUT = 0.20`
- `LEARNING_RATE = 5e-4`
- `WEIGHT_DECAY = 1e-5`
- `TF_DECAY_RATE = 0.95`
- `FINAL_MAX_EPOCHS = 60`
- `FINAL_PATIENCE = 12`

Shared inference defaults that should be saved with artifacts:

- `SAMPLING_TEMPERATURE = 1.5`
- `TREND_LOOKBACK_BARS = 20`
- `STRONG_TREND_THRESHOLD = 0.002`
- `MIN_PREDICTED_VOL = 0.0001`

## Important implementation decisions

### 1. `v9.5` retrieval must be corrected

The learning doc shows that the notebook version of `v9.5` attaches an untrained retrieval encoder after the forecaster is trained. That should **not** be copied forward as-is.

Recommended final training decision:

- train the `v9.5` forecaster normally
- build retrieval embeddings from the **trained forecaster encoder output** rather than a random new LSTM encoder
- store the retrieval database on CPU/disk as part of the artifact bundle

This keeps the `v9.5` idea but removes the biggest flaw without adding a second heavy training stage.

### 2. `v9.2` regime handling should be cleaned up

The learning doc shows `regime_indicator` was effectively constant in the original notebook. In the final training notebook:

- keep ATR and turbulence features
- compute a real dynamic regime stream offline
- either train with that dynamic regime feature or drop the bad constant feature entirely

The important part to preserve is:

- regime-aware inference wrapper
- ATR / turbulence information

not the constant-label bug.

### 3. `v9.6` should be trained on the aggregate, not a single legacy forecaster

The best final design is:

- first train all forecast experts
- then compute the ensemble forecast artifact
- then train the PPO policy on the **aggregated forecast path**

That makes `v9.6` a decision layer for the final system, not just for one older expert.

## Recommended artifact layout

Use one stable directory, for example:

`output/final_artifacts/`

Recommended structure:

```text
output/final_artifacts/
  manifest.json
  shared/
    raw_snapshot.parquet
    sessionized_snapshot.parquet
    split_spec.json
    backtest_spec.json
    calendar_meta.json
  models/
    v8_5/
      model.pt
      scaler.npz
      feature_manifest.json
      train_config.json
      inference_config.json
      metrics.json
      history.csv
    v9_1/
      ...
    v9_2/
      ...
    v9_3/
      ...
    v9_5/
      model.pt
      scaler.npz
      feature_manifest.json
      train_config.json
      inference_config.json
      metrics.json
      history.csv
      rag_database.npz
      rag_config.json
  ensemble/
    weights.json
    validation_summary.csv
    rolling_summary.csv
    aggregate_config.json
  rl/
    policy.pt
    env_config.json
    state_schema.json
    training_metrics.json
```

## What each expert artifact must contain

Every forecast expert bundle should include:

- model weights
- exact feature column list and order
- input-scaler stats
- training config
- inference config
- selected lookback
- exact selected hyperparameters
- walk-forward metrics
- rolling metrics
- training history

That way `FINAL.ipynb` never has to guess how to rebuild a model.

## Training notebook structure

Recommended notebook order:

1. **Title / purpose**
   Explain that this notebook creates all reusable artifacts for final inference.

2. **Imports, seed, device, artifact paths**
   Set deterministic seeds and define output folders.

3. **Fetch one shared market snapshot**
   Pull the latest approved training window once and save it.

4. **Sessionize and save shared datasets**
   Save raw and sessionized forms for reproducibility.

5. **Build one shared split spec**
   Save walk-forward slices and the chosen rolling backtest date.

6. **Train `v8.5`**
   Save full artifact bundle.

7. **Train `v9.1`**
   Save full artifact bundle.

8. **Train `v9.2`**
   Save full artifact bundle with cleaned regime handling.

9. **Train `v9.3`**
   Save full artifact bundle with low-VRAM settings.

10. **Train `v9.5`**
    Save full artifact bundle plus retrieval database.

11. **Run common validation and rolling evaluation for all forecast experts**
    Save aligned summary tables across models.

12. **Compute ensemble weights**
    Save static reliability weights for the five forecast experts.

13. **Generate aggregate rolling predictions**
    Use the saved forecast experts and weights to create the final forecast stream for RL.

14. **Train `v9.6` PPO decision layer**
    Train on the aggregate predictions and save the policy.

15. **Manifest + smoke-load check**
    Build one top-level manifest and verify every saved artifact reloads cleanly.

## Ensemble-weight generation

Do not hand-pick final weights inside `FINAL.ipynb`.

Instead, `FinalTrain.ipynb` should compute and save them from validation metrics.

Recommended approach:

- use the same aligned validation window for all forecast experts
- compute a reliability score per expert from:
  - one-step MAE
  - short-horizon path MAE
  - directional accuracy
  - rolling robustness
- normalize scores into weights that sum to `1.0`

This matters because the learning docs already show the experts are not equally reliable:

- `v9.1` had very unstable rolling-path errors
- `v9.3` needed low-VRAM handling and showed some training instability
- `v9.5` needs retrieval cleanup

So final weights should come from saved metrics, not from equal averaging.

## RL training design

The `v9.6` training phase should use:

- aggregated forecast path as the prediction input
- current market features:
  - RSI
  - MACD histogram
  - Bollinger position
  - ATR percentage
  - momentum features
  - VWAP deviation
  - OBV slope
  - direction
  - relative volume
- portfolio state
- regime indicator

Recommended PPO config to preserve from learnings:

- `ppo_lr = 3e-4`
- `ppo_gamma = 0.99`
- `ppo_gae_lambda = 0.95`
- `ppo_clip_epsilon = 0.2`
- `ppo_value_coef = 0.5`
- `ppo_entropy_coef = 0.01`
- `ppo_max_grad_norm = 0.5`
- `ppo_rollout_length = 512`
- `ppo_update_epochs = 4`
- `ppo_batch_size = 64`
- `rl_training_steps = 5000`

Recommended environment defaults:

- `initial_balance = 100000`
- `transaction_cost = 0.001`
- `max_position = 1.0`
- `dsr_eta = 0.1`
- regime-aware position scaling enabled

## VRAM / RAM plan

Target hardware:

- `8 GB` VRAM
- `32 GB` system RAM

### GPU rules

- train **one forecast expert at a time**
- after each expert:
  - save checkpoint
  - delete model
  - empty CUDA cache
- never keep multiple expert models on GPU together
- do not keep retrieval memory on GPU between phases

### Model-specific runtime choices

- `v8.5`, `v9.1`, `v9.2`, `v9.5`:
  - start with `batch_size = 256`
  - if memory pressure appears, drop to `128`
- `v9.3`:
  - `batch_size = 32`
  - AMP enabled
  - evaluation in batches
  - explicit memory cleanup after train/eval
- `v9.6` PPO:
  - run on CPU by default
  - dataset is tiny compared with the forecasting models, so it does not need GPU

### RAM rules

- keep dataframes and cached predictions in system RAM
- prefer writing large reusable objects to disk after each phase
- if window caches get large, use memmap / on-disk arrays instead of duplicating them in RAM

## Must-save outputs for FINAL.ipynb

At the end of `FinalTrain.ipynb`, these must exist:

- five forecast expert bundles
- one ensemble weight file
- one PPO bundle
- one top-level manifest
- one validation summary table across all experts

Those saved outputs are the source of truth for `FINAL.ipynb`. The final notebook should load them directly rather than recomputing training state.

If those exist, `FINAL.ipynb` can be pure loading + inference.

## Validation checklist for TrainFinal.ipynb

Before calling the notebook done:

- every expert artifact reloads successfully
- feature order is stored and reloadable
- scaler stats are saved and reloadable
- ensemble weights sum to `1.0`
- aggregated forecast can be regenerated from saved artifacts
- PPO state dimension matches saved env schema
- manifest paths are valid

## End state

When this notebook is done, we should have:

- one reproducible training notebook
- frozen artifacts for all forecast experts
- frozen ensemble weights
- frozen `v9.6` decision policy
- a clean path for `FINAL.ipynb` to do inference only

That is the split we want:

- `FinalTrain.ipynb` = all heavy training and saving
- `FINAL.ipynb` = load, predict, aggregate, display
