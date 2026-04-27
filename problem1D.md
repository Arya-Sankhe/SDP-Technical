# problem1D.md — Why daily (1D) forecasts can show the “same shape every time”

This document lists **specific, technical reasons** the MSFT 1D pipeline (`Final1DTrain.ipynb`, `FINAL1D.ipynb`, artifacts under `output/final_1d_artifacts/`) can produce **rolling frames that look like a repeated template** (flat or low‑volatility multi‑day paths, expert lines stacked on top of each other, aggregate candles barely moving vs. dimmed “actual future”).

It is **diagnostic**, not a fix list. Items are grouped by layer (inference → training → data → ensemble → guards). Several items can be true at once.

---

## 1. Inference: mean-only collapse and ignored uncertainty (historical + residual)

### 1.1 Deterministic Gaussian mean (`mu` only)

- The heads are Gaussian: each forward pass returns **`mu`** and **`log_sigma`** for all horizon steps and OHLC channels.
- For a long time, **`generate_point_forecast` used only `mu`**, treating the problem as a pure point forecast.
- **Effect:** If `mu` is small or weakly dependent on distant context (see §3), every anchor maps to a **similar vector of normalized returns**. After denormalization and price chaining, that becomes a **similar price path shape** in the plot window.

### 1.2 `sampling_temperature == 0` in saved `inference_config.json`

- With temperature 0 and **no** sampling, two anchors with **similar** `mu` produce **nearly identical** decoded paths.
- Even when anchors differ, if `mu` varies less than plotting resolution, charts look “the same.”

### 1.3 `log_sigma` unused at decode (until fixed)

- **`log_sigma` encodes predicted uncertainty.** Ignoring it throws away the only part of the Gaussian head that encourages **state-dependent width** of the distribution at inference.
- **Effect:** All randomness and many “volatility aware” training signals are **invisible** at run time; only the conditional mean is shown.

### 1.4 Regime / daily temperature passed but not applied (historical)

- Rolling and regime logic compute an **effective temperature** multiplier in some call paths; if the forecast function **does not consume** `temperature`, regime scaling **cannot** change the path.
- **Effect:** Regime‑aware inference was partly **cosmetic** relative to the actual tensor that got decoded.

### 1.5 RAG blend weight zero and retrieval unused (historical)

- For `gru_rag`, a **retrieval index** (`rag_database.npz`) exists, but **`generate_point_forecast` did not blend** retrieved neighbor futures when `blend_weight` was 0 or when blend was never called.
- **Effect:** The “RAG” expert **degenerated** to the same backbone behavior as a non‑RAG model at inference, losing a source of **input‑dependent** path diversity.

### 1.6 Rolling temperature floor vs. live forecast asymmetry

- A **minimum sampling temperature** used only in rolling (e.g. `FINAL_ROLLING_MIN_SAMPLING_TEMPERATURE`) makes rolling paths **more stochastic** than a strict `temperature=0` live aggregate run.
- **Effect:** You can still see **template‑like means** in some views while rolling looks “slightly better”; they are not the same code path.

---

## 2. Decoding: `returns_to_prices_seq` and `enforce_candle_validity`

### 2.1 Chained decoding vs. how targets were built

- Targets are **per‑day** log returns vs. **that day’s** true `prev_close` in the dataset.
- Decoding uses **`returns_to_prices_seq`**: each step multiplies `exp(r_t)` by a **running** `prev_close` taken from **previous predicted close** (after step 0).
- **Training mismatch:** The model is trained to predict each day’s returns **conditionally on history**, not on recursively feeding its own previous‑day price errors the way an autoregressive sequence model would be trained.
- **Effect:** Small errors in early `mu` **compound**; the optimizer can favor **tiny returns** everywhere (safe, flat path in price space) to avoid blowing up the chain.

### 2.2 `enforce_candle_validity`

- After exponentiation, OHLC rows are **clamped** so High/Low bracket Open/Close sensibly.
- **Effect:** If raw `mu` implies inconsistent OHLC in return space, **validity repair shrinks bodies and ranges**, which visually **compresses** candles toward a thin bar.

---

## 3. Architecture: single vector for the full horizon

### 3.1 Direct multi‑horizon head (no latent autoregression across days in the head)

- `Seq2SeqAttnGRU` / `HybridSeq2SeqForecaster` collapse the encoder output to a **fixed‑size feature vector**, then apply **linear heads** to emit **`horizon × 4`** values at once.
- **Effect:** Far days (t+5, t+6) are not generated with **explicit hidden state evolution per future day**; they are **affine functions of the same summary**. That favors **global templates** (similar slopes across anchors) when the summary is weakly discriminative.

### 3.2 Attention + pooling averages context

- The model uses **attention over encoder time** and also **mean‑pools** encoder memory in the head path.
- **Effect:** Strong **smoothing** of temporal detail; abrupt pre‑anchor moves can be **washed out** before they influence far‑horizon logits.

### 3.3 `teacher_forcing_ratio` and `y` ignored in `forward`

- `forward(..., y=..., teacher_forcing_ratio=...)` accepts `y` but **does not use it** in the implemented architectures.
- **Effect:** “Teacher forcing” in the training loop **does not change** the forward pass; the naming is misleading. The model is always a **direct map** `context → full horizon`, not a trained autoregressive decoder over its own outputs.

---

## 4. Training objective: losses that permit “flat horizon” solutions

### 4.1 Huber / NLL dominate toward conservative `mu`

- **Huber (SmoothL1)** on all OHLC channels penalizes outliers less than MSE; combined with **Gaussian NLL**, the model can reduce loss by predicting **small |mu|** in normalized space when signal‑to‑noise is low.
- **Effect:** A **low‑amplitude template** across the horizon is a **strong local optimum** for noisy financial targets.

### 4.2 Per‑horizon weighting is mild

- Huber uses `torch.linspace(1.0, 1.5, horizon)` over steps; NLL uses `1.0 → 2.0`. That is only a **50–100%** relative emphasis from first to last day.
- **Effect:** The objective does not **strongly** force late steps to track **independent** realized variation; they can stay near a **shared baseline**.

### 4.3 `candle_range_loss` operates in normalized return space

- Range loss compares **predicted vs. target range in scaled return space**, not dollar range after decoding.
- **Effect:** It helps shape **normalized** spreads but does not guarantee **price‑chart realism** after `target_scale` and `exp` chaining.

### 4.4 Historical zero weights on `volatility` / `directional`

- When those weights were **0.0**, nothing explicitly penalized **“same demeaned path every window”** or **wrong t+1 sign** beyond what Huber/NLL already do.
- **Effect:** **Template collapse** was **unpenalized** along those axes (partially addressed in later notebook versions).

---

## 5. Target scaling: `target_scale` floor and ceiling

### 5.1 `estimate_target_scale`

- Each window uses a **scalar** `target_scale` from recent ATR% / realized vol (clamped by `TARGET_SCALE_FLOOR` / `TARGET_SCALE_CEILING`).
- **Effect:** Targets in training are **normalized** by that scale; the model learns **shape in normalized units**. If scales are **similar** across many anchors, **normalized `mu` patterns** can look alike even when dollar vol differs.

### 5.2 Floor clipping

- When volatility estimates are weak, **`TARGET_SCALE_FLOOR`** kicks in.
- **Effect:** Very different quiet periods can get **similar normalization**, which **reduces the model’s pressure** to output very different normalized profiles.

---

## 6. Data and windows: fewer effective samples than calendar days

### 6.1 Row filters shrink diversity

- `make_multistep_windows` **drops** any horizon where `row_imputed` or `row_open_skip` is true.
- **Effect:** Training sees **only “clean” segments**; the model may under‑learn **messy** real‑world stretches that still appear at inference.

### 6.2 Fixed cutoff and nonstationarity

- Production training is on `feature_df.iloc[:BACKTEST_CUTOFF_INDEX]` with a **chronological** val tail.
- **Effect:** Adding older data does not always help **current** regimes; the optimum can still be a **generic** path if the mapping is hard.

### 6.3 Walkforward validation subsampling

- Walkforward metrics cap validation windows (`MAX_VAL_WINDOWS_PER_SLICE`, stricter for low‑VRAM v9.5).
- **Effect:** Early stopping / reporting may be **noisier or biased** toward a subset of time; not the direct cause of identical shapes, but it can **under‑penalize** bad generalization.

---

## 7. Ensemble and aggregate: three experts → one shape

### 7.1 Similar architectures and overlapping inputs

- Experts share the **same symbol**, overlapping **core** features, and **Gaussian + direct horizon** heads (with variations: core vs regime vs RAG).
- **Effect:** If each expert’s `mu` is **near template**, the **weighted average** is still a **template** (often tighter than any single expert).

### 7.2 Weights concentrated

- If ensemble weights are **roughly uniform** or one expert dominates, **diversity** across versions in the chart is limited.

### 7.3 RAG expert without working blend (historical)

- v9.5 could not contribute **retrieval‑driven diversity** at decode when blend was unused or weight was 0.
- **Effect:** Three lines **collapse visually** to one bundle.

---

## 8. Post‑processing guards (when enabled): mechanical flattening

> In many runs, `FINAL_SOFT_SWING_GUARD_ENABLED`, `FINAL_EMPIRICAL_ENVELOPE_ENABLED`, `FINAL_CANDLE_RANGE_GUARD_ENABLED`, and T1 guards are **false**, so these are **not** always active—but if turned on, they are **explicit** shape compressors.

### 8.1 `shrink_path_to_anchor`

- Pulls the entire path toward `anchor_prev_close` by factor `FINAL_AGGREGATE_SHRINK`.
- **Effect:** When `< 1`, **reduces all excursions** toward the anchor level.

### 8.2 `soft_cap_step_swings`

- Caps **per‑step close move** relative to ATR / quantiles of recent moves.
- **Effect:** **Clips** large trends in the aggregate path; repeated caps across frames yield **similar micro‑moves**.

### 8.3 `empirical_anchor_envelope_cap_path`

- Caps each step’s close **relative to the initial anchor**, not an expanding budget.
- **Effect:** Multi‑day drift is **strongly limited**; paths can look like **wiggles around a fixed level**.

### 8.4 `apply_t1_temporal_direction_guard` + `shift_path_from_first_close`

- Adjusts first close then **fades** shifts along the horizon with a power law.
- **Effect:** Later days are **pulled toward** the pre‑shift template; can **homogenize** tail shape across anchors when combined with other caps.

### 8.5 `cap_candle_ranges`

- Limits wick extension using historical high‑low quantiles.
- **Effect:** **Narrower predicted candles** vs. history.

---

## 9. Visualization and perception (not bugs, but explain “sameness”)

### 9.1 Same horizon length (7 days)

- Every frame shows **seven** future steps; the human eye compares **slot patterns** (day 1 bump, day 2 dip, …). Models that output **weakly varying vectors** look like “one memorized week.”

### 9.2 Price level tracks anchor, shape looks repeated

- Even when **level** shifts with `anchor_prev_close`, a **similar normalized profile** after decode looks like “the same squiggle pasted at a new height.”

### 9.3 Expert lines on the same axes

- v8_5 / v9_2 / v9_5 close paths plotted together: if RMS difference is a few dollars on a 400–550 y‑axis, lines **overlap visually**.

---

## 10. Root interaction (summary diagram in words)

1. **Encoder** smooths context → **weakly varying** summary.  
2. **Direct head** maps summary → **full‑horizon `mu`**.  
3. **Losses** allow **small, smooth `mu`** as a cheap solution.  
4. **Decode** (`exp`, chain, `enforce_candle_validity`) **compresses** OHLC.  
5. **Ensemble** averages similar experts → **one bundle**.  
6. **Optional guards** (if on) **clip** remaining structure.  
7. **Plot** emphasizes **seven similar candles** next to volatile truth.

That loop is why **“more years of data” alone** often moves the needle less than changing **objective, head, decode training match, or post‑processing**.

---

## 11. Checklist: what to verify for each suspected cause

| Symptom / hypothesis | What to check |
|----------------------|----------------|
| Mean collapse | Distribution of `mu` across many anchors (norm space): variance near 0? |
| Ignored sigma / temp | Inference code path: is `log_sigma` used when `temperature > 0`? Saved JSON values? |
| RAG inert | `rag_config.json` `blend_weight`; does decode call `blend_retrieved_future`? |
| Chain vs training | Compare decode with **oracle** prev closes vs **predicted** prev closes for the same `mu` |
| Ensemble overlap | Pairwise correlation of expert **raw `mu`** vectors across anchors |
| Guards | `FINAL_*_ENABLED` flags and cap magnitudes vs ATR in crisis weeks |
| Scaling | Histogram of `target_scale` at anchors where paths look identical |

---

## 12. Scope

- Applies to the **1D final** stack described in `Final1DTrain.ipynb` / `FINAL1D.ipynb` and artifacts they consume/produce.  
- Some bullets describe **historical** behavior before inference fixes; they remain listed because **old checkpoints**, **zero temperature**, or **zero RAG blend** configs can **reproduce** the same failure mode until retrained and reconfigured.

---

*File generated for internal engineering review. Update as the pipeline changes.*
