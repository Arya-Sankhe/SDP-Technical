# PRD: GRU-Based 1-Minute Stock Price Forecasting (Design Phase)

## 1. Product Goal
Design a deep learning forecasting system (PyTorch, GRU) that predicts short-term stock prices from raw 1-minute candle prices.

Initial scope is **stocks only**. Crypto support is a later extension.

## 2. Scope and Phase
### Current Phase (Now)
- Design and specification only.
- No implementation in this phase.

### Next Phase
- Build and train the model based on this PRD.
- Run experiments and evaluate performance.

## 3. Core Forecasting Requirement
- Data granularity: **1-minute candles**.
- Historical range used: **last 60 calendar days**.
- Model input window: **last 500 candles**.
- Model output: **next candle prices**.
- Inference method: **recursive multi-step forecasting**.
- Forecast horizon: **15 steps (15 future 1-minute candles)**.

Recursive logic:
1. Use candles `t-499 ... t` to predict candle `t+1`.
2. Append predicted candle `t+1` into the rolling sequence.
3. Use updated last 500-candle window to predict `t+2`.
4. Repeat until `t+15`.

## 4. Data and Features
### Included (for now)
- Candle prices only: **Open, High, Low, Close (OHLC)**.

### Explicitly Excluded (for now)
- Technical indicators (RSI, MACD, EMA, Bollinger Bands, etc.).
- Fundamental, news, sentiment, or macro features.
- Order-book/microstructure features.

## 5. Model Design
### Architecture
- Framework: **PyTorch**.
- Backbone: **GRU** sequence model.
- Input shape (per sample): `[500, 4]` (OHLC over 500 minutes).
- Output shape (per prediction): `[4]` (next candle OHLC).

### Baseline Configuration (initial)
- GRU layers: 2
- Hidden size: 128
- Dropout: 0.2 (between GRU layers)
- Head: fully connected layer from hidden state to 4 outputs

These hyperparameters are a starting baseline and can be tuned in implementation.

## 6. Training Design
### Dataset Construction
- Build rolling windows from each symbol's 60-day 1-minute data.
- For each window:
  - Input `X`: candles `[t-499 ... t]`
  - Target `y`: candle `[t+1]`

### Split Strategy
- Time-based split (no random shuffle across time):
  - Train: oldest segment
  - Validation: middle segment
  - Test: newest segment

### Normalization
- Use train-only fit normalization.
- Candidate baseline: per-feature standardization (OHLC independently).

### Loss and Optimization
- Primary loss: MSE on OHLC outputs.
- Optimizer: Adam.
- Learning-rate scheduler: optional in first implementation.

## 7. Inference Design
Given the most recent 500 real candles:
1. Predict next candle (step 1).
2. Feed predicted candle back into the input sequence.
3. Repeat recursively to 15 predicted candles.

Output:
- Sequence of 15 predicted future candles.
- Derived predicted close-price path for charting and evaluation.

## 8. Evaluation Design
### Core Metrics
- MAE and RMSE on Close price (primary).
- MAE and RMSE on OHLC (secondary).
- Directional accuracy for Close movement (up/down).

### Evaluation Modes
- One-step test performance (`t+1`).
- 15-step recursive performance (`t+1 ... t+15`) to measure error accumulation.

## 9. System and Environment Requirements
- OS target: **Windows**.
- GPU target: **NVIDIA RTX 3070**.
- Deep learning stack: **PyTorch with CUDA**.

Implementation should support:
- GPU training/inference by default when CUDA is available.
- CPU fallback for debugging.

## 10. Non-Goals (Current Iteration)
- Live trading execution.
- Portfolio management or risk engine.
- Multi-asset joint modeling.
- Indicator-enhanced feature engineering.

## 11. Deliverables for Design Phase
- Finalized model/data/training/inference specification (this PRD).
- Implementation task breakdown (next step).
- Experiment plan template (baseline + tuning runs).

## 12. Implementation Readiness Checklist
- [ ] Confirm stock data provider and symbol universe.
- [ ] Confirm market hours handling and missing-minute policy.
- [ ] Confirm train/val/test time boundaries.
- [ ] Confirm PyTorch + CUDA setup on Windows RTX 3070.
- [ ] Start implementation phase.
