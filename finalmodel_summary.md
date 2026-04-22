# Final Model Summary

`FINAL.ipynb` is the inference-only notebook for the deployed final system. It does not train anything. Instead, it loads frozen artifacts from `output/final_artifacts/`, fetches the latest `MSFT` 1-minute bars from Alpaca, rebuilds the feature frames, runs the saved forecast experts one at a time, aggregates their forecasts into one final OHLC path, and then runs the saved `v9.6` PPO policy on top to produce a trading stance and position size.

## Active Forecast Stack

The current active experts are:

- `v8.5`: core-feature attention GRU, lookback `64`
- `v9.1`: technical-feature attention GRU, lookback `160`
- `v9.5`: core-feature attention GRU with RAG-style retrieval, lookback `256`

`v9.2` still exists in the artifact manifest but is explicitly excluded in the final notebook. `v9.3` is not part of the current deployed stack. `v9.6` is used as a decision layer, not as a direct forecaster.

## How The Final Forecast Is Built

1. Fetch the latest intraday data and sessionize it.
2. Build three feature views:
   - `core`
   - `technical`
   - `regime`
3. For each active expert:
   - load its saved checkpoint, scaler, feature manifest, and inference config
   - build its own lookback window
   - scale inputs with the saved training statistics
   - sample multiple future return paths with the probabilistic seq2seq model
   - for `v9.5`, retrieve similar historical futures from the saved RAG database (`k=5`, blend weight `0.25`)
   - select the best sampled path by trend alignment with recent history
4. Aggregate the expert paths using fixed final weights:
   - `v8.5 = 0.10`
   - `v9.1 = 0.10`
   - `v9.5 = 0.80`
5. Post-process the aggregate forecast:
   - global shrink toward the anchor close: `0.70`
   - soft swing guard:
     - enabled
     - lookback `120`
     - cap based on recent 1-minute move `q95 * 6.0`, `ATR14 * 1.75`, and a minimum move floor of `1.20`
     - only the excess beyond the cap is compressed (`excess_scale = 0.55`)

That last step is important: normal predictions are left alone, while only unrealistic 1-minute jumps are scaled down.

## Decision Layer

After the forecast is built, the notebook loads the saved `v9.6` PPO actor-critic policy and forms a `215`-dimensional RL state:

- flattened aggregate OHLC path: `200`
- market features: `10`
- portfolio features: `4`
- regime feature: `1`

The PPO policy does not change the forecast candles. It outputs a recommended action, stance, and position size on top of the forecast.

## Validation In The Notebook

The notebook also runs a strictly causal one-day rolling backtest. For each anchor minute in the selected session, every expert predicts using only prior data, the aggregate path is rebuilt, and the notebook reports rolling metrics plus charts.

Latest saved run:

- run folder: `output/final_runs/20260421_081700`
- rolling date: `2026-04-20`
- anchors: `334`
- aggregate step-1 close MAE: `0.7268`
- aggregate path MAE: `2.4172`
- aggregate step-10 close MAE: `2.1363`
- directional hit rate at `t+1`: `0.4701`

## Current Drawbacks / Problems

- The system is now much better on path error than earlier runs, but direction accuracy is still weak. The latest `t+1` hit rate is below `50%`.
- The final ensemble is effectively dominated by `v9.5`. That is helping accuracy, but it means the ensemble is not very diverse right now.
- Because `v9.2` is excluded, the aggregate regime signal is effectively inactive in the current notebook. In practice, the final run is usually treated as `NORMAL`, so the PPO regime input is not adding much information.
- The soft swing guard is a heuristic overlay, not a learned correction. It helps remove impossible 1-minute spikes, but it can also damp genuine fast moves.
- The rolling validation in the notebook is only a one-day backtest by default. That is useful for debugging, but not enough by itself to prove full robustness.
- The final system still depends on live Alpaca data quality, sessionization, and feed consistency.

In short: the current final notebook is a frozen multi-expert inference pipeline with a `v9.5`-heavy forecast ensemble, heuristic swing suppression, and a PPO trading overlay. It is materially better than earlier versions on forecast error, but still needs work on direction quality, true regime-awareness, and robustness across more backtest days.
