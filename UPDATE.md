# Stock Price Prediction Model - Comprehensive Update Guide

## Executive Summary

This document provides a detailed roadmap for upgrading your current Seq2SeqAttnGRU model based on insights from 8 cutting-edge research papers. Your current model achieves ~48.7% directional accuracy (worse than a simple persistence baseline of 73.8%), indicating fundamental issues with trend injection and feature representation.

**Priority Implementation Order:**
1. **Phase 1 (Quick Wins)**: Technical Indicators + Regime Detection
2. **Phase 2 (Architecture)**: iTransformer Encoder + Frequency Features  
3. **Phase 3 (Advanced)**: RAG Retrieval + RL Decision Layer

---

## Table of Contents

1. [Current Model Analysis](#1-current-model-analysis)
2. [Phase 1: Technical Indicators](#2-phase-1-technical-indicators)
3. [Phase 2: Market Regime Detection](#3-phase-2-market-regime-detection)
4. [Phase 3: iTransformer Architecture](#4-phase-3-itransformer-architecture)
5. [Phase 4: Time-Frequency Features](#5-phase-4-time-frequency-features)
6. [Phase 5: RAG Retrieval System](#6-phase-5-rag-retrieval-system)
7. [Phase 6: RL Decision Layer](#7-phase-6-rl-decision-layer)
8. [Testing Framework](#8-testing-framework)
9. [Implementation Timeline](#9-implementation-timeline)

---

## 1. Current Model Analysis

### 1.1 What's Working
- Probabilistic forecasting with NLL loss
- Attention mechanism for sequence modeling
- Volatility matching for realistic predictions
- Candle validity enforcement (OHLC structure)

### 1.2 Critical Issues

| Issue | Evidence | Root Cause |
|-------|----------|------------|
| Directional Accuracy 48.7% | Below random chance | Trend injection creates momentum-following random walk |
| Close MAE 0.30 | Worse than persistence (0.21) | No memory of similar historical patterns |
| No regime awareness | Same behavior in all markets | Missing turbulence/volatility regime detection |
| Limited features | Only OHLCV + derived | Missing standard technical indicators |

### 1.3 Why Trend Injection Fails

Your current ensemble trend injection selects paths based on slope matching:
```python
# Current approach - problematic
selected_path = min(generated_paths, key=lambda p: abs(slope(p) - slope(recent_window)))
```

This creates a **self-fulfilling prophecy**: you're selecting paths that look like recent history, not paths that predict future movement. The model essentially says "tomorrow will look like a smoothed version of today" - which is why you underperform persistence.

---

## 2. Phase 1: Technical Indicators

### 2.1 Why This Helps

**Source Paper**: VTA (Verbal Technical Analysis) - Paper 2

Technical indicators encode decades of trader wisdom into mathematical formulas. They provide:
- **Trend information**: SMA, EMA crossovers indicate momentum
- **Momentum signals**: RSI, MACD show overbought/oversold conditions
- **Volatility context**: Bollinger Bands, ATR show market regime
- **Volume confirmation**: OBV, VWAP validate price movements

**Expected Impact**: +5-10% directional accuracy, better volatility prediction

### 2.2 Implementation

#### Step 1: Add Technical Indicator Calculator

Create new file: `technical_indicators.py`

```python
"""
Technical Indicators Module
Implements standard technical analysis indicators for feature engineering.
Based on: VTA Paper (Verbal Technical Analysis for Stock Forecasting)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class TechnicalIndicatorCalculator:
    """
    Calculates technical indicators for financial time series.
    
    Indicators implemented:
    - Trend: SMA, EMA, MACD
    - Momentum: RSI, Stochastic
    - Volatility: Bollinger Bands, ATR
    - Volume: OBV, VWAP
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with price data.
        
        Args:
            df: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        self.df = df.copy()
        self._validate_columns()
    
    def _validate_columns(self):
        """Ensure required columns exist."""
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
    
    # ==================== TREND INDICATORS ====================
    
    def add_sma(self, periods: List[int] = [5, 10, 20, 50]) -> 'TechnicalIndicatorCalculator':
        """
        Simple Moving Average - smoothed price over N periods.
        
        Why: Reduces noise, shows underlying trend direction.
        Golden Cross (SMA50 > SMA200) = bullish, Death Cross = bearish.
        """
        for period in periods:
            self.df[f'sma_{period}'] = self.df['Close'].rolling(window=period).mean()
        return self
    
    def add_ema(self, periods: List[int] = [12, 26]) -> 'TechnicalIndicatorCalculator':
        """
        Exponential Moving Average - weighted average giving more importance to recent prices.
        
        Why: Reacts faster to price changes than SMA. EMA12/EMA26 used in MACD.
        """
        for period in periods:
            self.df[f'ema_{period}'] = self.df['Close'].ewm(span=period, adjust=False).mean()
        return self
    
    def add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> 'TechnicalIndicatorCalculator':
        """
        MACD (Moving Average Convergence Divergence) - trend-following momentum indicator.
        
        Formula:
        - MACD Line = EMA(fast) - EMA(slow)
        - Signal Line = EMA(MACD Line, signal_period)
        - Histogram = MACD Line - Signal Line
        
        Why: Shows relationship between two EMAs. Crossovers signal momentum shifts.
        """
        ema_fast = self.df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['Close'].ewm(span=slow, adjust=False).mean()
        
        self.df['macd_line'] = ema_fast - ema_slow
        self.df['macd_signal'] = self.df['macd_line'].ewm(span=signal, adjust=False).mean()
        self.df['macd_histogram'] = self.df['macd_line'] - self.df['macd_signal']
        
        # MACD momentum (rate of change)
        self.df['macd_momentum'] = self.df['macd_histogram'].diff()
        
        return self
    
    # ==================== MOMENTUM INDICATORS ====================
    
    def add_rsi(self, period: int = 14) -> 'TechnicalIndicatorCalculator':
        """
        Relative Strength Index - momentum oscillator (0-100).
        
        Formula: RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss over N periods
        
        Interpretation:
        - RSI > 70: Overbought (potential sell)
        - RSI < 30: Oversold (potential buy)
        - RSI > 50: Bullish momentum
        - RSI < 50: Bearish momentum
        
        Why: Identifies overbought/oversold conditions, divergence signals reversals.
        """
        delta = self.df['Close'].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))
        
        # Use Wilder's smoothing (exponential moving average)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        self.df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # RSI trend (momentum of momentum)
        self.df[f'rsi_{period}_slope'] = self.df[f'rsi_{period}'].diff(5)
        
        return self
    
    def add_stochastic(self, k_period: int = 14, d_period: int = 3) -> 'TechnicalIndicatorCalculator':
        """
        Stochastic Oscillator - compares closing price to price range over period.
        
        Formula:
        - %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        - %D = SMA(%K, d_period)
        
        Why: Shows momentum by comparing close to recent range. Overbought > 80, Oversold < 20.
        """
        low_min = self.df['Low'].rolling(window=k_period).min()
        high_max = self.df['High'].rolling(window=k_period).max()
        
        self.df['stoch_k'] = 100 * (self.df['Close'] - low_min) / (high_max - low_min)
        self.df['stoch_d'] = self.df['stoch_k'].rolling(window=d_period).mean()
        
        return self
    
    # ==================== VOLATILITY INDICATORS ====================
    
    def add_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> 'TechnicalIndicatorCalculator':
        """
        Bollinger Bands - volatility bands placed above/below moving average.
        
        Formula:
        - Middle Band = SMA(period)
        - Upper Band = Middle + (std_dev * std)
        - Lower Band = Middle - (std_dev * std)
        
        Interpretation:
        - Price near upper band: potentially overbought
        - Price near lower band: potentially oversold
        - Band width indicates volatility
        
        Why: Dynamic support/resistance levels based on volatility.
        """
        sma = self.df['Close'].rolling(window=period).mean()
        std = self.df['Close'].rolling(window=period).std()
        
        self.df['bb_middle'] = sma
        self.df['bb_upper'] = sma + (std_dev * std)
        self.df['bb_lower'] = sma - (std_dev * std)
        self.df['bb_width'] = (self.df['bb_upper'] - self.df['bb_lower']) / sma
        self.df['bb_position'] = (self.df['Close'] - self.df['bb_lower']) / (self.df['bb_upper'] - self.df['bb_lower'])
        
        return self
    
    def add_atr(self, period: int = 14) -> 'TechnicalIndicatorCalculator':
        """
        Average True Range - volatility indicator.
        
        True Range = max(
            High - Low,
            abs(High - Previous Close),
            abs(Low - Previous Close)
        )
        ATR = EMA(True Range, period)
        
        Why: Measures volatility independent of direction. Used for position sizing.
        """
        high_low = self.df['High'] - self.df['Low']
        high_close = abs(self.df['High'] - self.df['Close'].shift())
        low_close = abs(self.df['Low'] - self.df['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df[f'atr_{period}'] = true_range.ewm(alpha=1/period, adjust=False).mean()
        
        # ATR as percentage of price
        self.df[f'atr_{period}_pct'] = self.df[f'atr_{period}'] / self.df['Close']
        
        return self
    
    # ==================== VOLUME INDICATORS ====================
    
    def add_obv(self) -> 'TechnicalIndicatorCalculator':
        """
        On-Balance Volume - cumulative volume flow indicator.
        
        Formula:
        - If Close > Previous Close: OBV += Volume
        - If Close < Previous Close: OBV -= Volume
        - If Close = Previous Close: OBV unchanged
        
        Why: Volume precedes price. OBV divergence from price signals trend weakness.
        """
        obv = [0]
        for i in range(1, len(self.df)):
            if self.df['Close'].iloc[i] > self.df['Close'].iloc[i-1]:
                obv.append(obv[-1] + self.df['Volume'].iloc[i])
            elif self.df['Close'].iloc[i] < self.df['Close'].iloc[i-1]:
                obv.append(obv[-1] - self.df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        self.df['obv'] = obv
        self.df['obv_slope'] = self.df['obv'].diff(5)
        
        return self
    
    def add_vwap(self, period: int = 20) -> 'TechnicalIndicatorCalculator':
        """
        Volume-Weighted Average Price - average price weighted by volume.
        
        Formula: VWAP = sum(Price * Volume) / sum(Volume)
        where Price = (High + Low + Close) / 3
        
        Why: Institutions use VWAP as benchmark. Price above VWAP = bullish.
        """
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        
        # Rolling VWAP
        tp_vol = typical_price * self.df['Volume']
        self.df[f'vwap_{period}'] = tp_vol.rolling(window=period).sum() / self.df['Volume'].rolling(window=period).sum()
        
        # VWAP deviation
        self.df[f'vwap_{period}_dev'] = (self.df['Close'] - self.df[f'vwap_{period}']) / self.df[f'vwap_{period}']
        
        return self
    
    # ==================== CUSTOM FEATURES ====================
    
    def add_price_momentum(self, periods: List[int] = [5, 10, 20]) -> 'TechnicalIndicatorCalculator':
        """
        Price momentum - rate of change over N periods.
        
        Why: Simple but effective trend strength indicator.
        """
        for period in periods:
            self.df[f'price_momentum_{period}'] = (
                self.df['Close'] / self.df['Close'].shift(period) - 1
            )
        return self
    
    def add_candle_features(self) -> 'TechnicalIndicatorCalculator':
        """
        Candlestick pattern features.
        
        Why: Encodes price action patterns used by technical traders.
        """
        # Body size
        self.df['body_size'] = abs(self.df['Close'] - self.df['Open'])
        self.df['body_pct'] = self.df['body_size'] / (self.df['High'] - self.df['Low'] + 1e-8)
        
        # Upper/Lower shadows
        self.df['upper_shadow'] = self.df['High'] - self.df[['Open', 'Close']].max(axis=1)
        self.df['lower_shadow'] = self.df[['Open', 'Close']].min(axis=1) - self.df['Low']
        
        # Direction
        self.df['direction'] = np.where(self.df['Close'] > self.df['Open'], 1, 
                                       np.where(self.df['Close'] < self.df['Open'], -1, 0))
        
        return self
    
    def get_all_indicators(self) -> pd.DataFrame:
        """Calculate all indicators and return enhanced DataFrame."""
        return (self
            .add_sma()
            .add_ema()
            .add_macd()
            .add_rsi()
            .add_stochastic()
            .add_bollinger_bands()
            .add_atr()
            .add_obv()
            .add_vwap()
            .add_price_momentum()
            .add_candle_features()
            .df)


def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to add all technical indicators to a DataFrame.
    
    Args:
        df: DataFrame with OHLCV columns
        
    Returns:
        DataFrame with added technical indicator columns
    """
    calc = TechnicalIndicatorCalculator(df)
    return calc.get_all_indicators()
```

#### Step 2: Update Feature Configuration

Modify your `BASE_FEATURE_COLS` in the main script:

```python
# OLD: Limited feature set
BASE_FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'returns', 'log_returns', 'volatility_20',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
]

# NEW: Comprehensive technical indicators
BASE_FEATURE_COLS = [
    # Original OHLCV
    'Open', 'High', 'Low', 'Close', 'Volume',
    
    # Trend Indicators
    'sma_5', 'sma_10', 'sma_20', 'sma_50',
    'ema_12', 'ema_26',
    'macd_line', 'macd_signal', 'macd_histogram', 'macd_momentum',
    
    # Momentum Indicators  
    'rsi_14', 'rsi_14_slope',
    'stoch_k', 'stoch_d',
    
    # Volatility Indicators
    'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
    'atr_14', 'atr_14_pct',
    
    # Volume Indicators
    'obv', 'obv_slope',
    'vwap_20', 'vwap_20_dev',
    
    # Price Action
    'price_momentum_5', 'price_momentum_10', 'price_momentum_20',
    'body_size', 'body_pct', 'upper_shadow', 'lower_shadow', 'direction',
    
    # Temporal (keep existing)
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
]

# Total: ~35 features (vs current ~12)
FEATURE_DIM = len(BASE_FEATURE_COLS)
```

#### Step 3: Integrate into Data Loading

```python
# In your data loading function
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    
    # Add technical indicators
    df = calculate_technical_features(df)
    
    # Handle NaN values from indicator calculations
    df = df.fillna(method='ffill').fillna(0)
    
    return df
```

### 2.3 Testing Technical Indicators

```python
# test_indicators.py
import unittest
import pandas as pd
import numpy as np
from technical_indicators import TechnicalIndicatorCalculator, calculate_technical_features


class TestTechnicalIndicators(unittest.TestCase):
    
    def setUp(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100
        
        # Generate realistic price data with trend
        trend = np.linspace(100, 150, n)
        noise = np.random.randn(n) * 2
        close = trend + noise
        
        self.df = pd.DataFrame({
            'Open': close - np.random.rand(n) * 2,
            'High': close + np.random.rand(n) * 3,
            'Low': close - np.random.rand(n) * 3,
            'Close': close,
            'Volume': np.random.randint(1000, 10000, n)
        })
    
    def test_sma_calculation(self):
        """Test SMA values are correct."""
        calc = TechnicalIndicatorCalculator(self.df)
        calc.add_sma(periods=[20])
        
        # Manual calculation
        expected_sma = self.df['Close'].rolling(20).mean()
        pd.testing.assert_series_equal(
            calc.df['sma_20'], 
            expected_sma,
            check_names=False
        )
    
    def test_rsi_range(self):
        """Test RSI is within 0-100 range."""
        calc = TechnicalIndicatorCalculator(self.df)
        calc.add_rsi()
        
        rsi = calc.df['rsi_14'].dropna()
        self.assertTrue((rsi >= 0).all() and (rsi <= 100).all())
    
    def test_bollinger_band_width(self):
        """Test Bollinger Bands have correct width relationship."""
        calc = TechnicalIndicatorCalculator(self.df)
        calc.add_bollinger_bands()
        
        # Upper > Middle > Lower
        self.assertTrue((calc.df['bb_upper'] >= calc.df['bb_middle']).all())
        self.assertTrue((calc.df['bb_middle'] >= calc.df['bb_lower']).all())
    
    def test_macd_signal_lag(self):
        """Test MACD signal is smoother (less volatile) than MACD line."""
        calc = TechnicalIndicatorCalculator(self.df)
        calc.add_macd()
        
        macd_std = calc.df['macd_line'].std()
        signal_std = calc.df['macd_signal'].std()
        
        # Signal should be less volatile
        self.assertLess(signal_std, macd_std)
    
    def test_all_indicators(self):
        """Test that all indicators are added correctly."""
        result = calculate_technical_features(self.df)
        
        expected_cols = [
            'sma_20', 'ema_12', 'macd_line', 'rsi_14',
            'bb_upper', 'atr_14', 'obv', 'vwap_20'
        ]
        
        for col in expected_cols:
            self.assertIn(col, result.columns)


if __name__ == '__main__':
    unittest.main()
```

**Run Tests:**
```bash
python -m pytest test_indicators.py -v
```

---

## 3. Phase 2: Market Regime Detection

### 3.1 Why This Helps

**Source Papers**: Ensemble DRL (Paper 3) + Pipeline.docx (Paper 6)

Markets behave differently in different regimes:
- **Normal regime**: Trend-following works, moderate volatility
- **Elevated volatility**: Mean-reversion dominates, wider stops needed
- **Crisis regime**: Correlation → 1, liquidity dries up, reduce exposure

Your current model uses the same temperature and strategy regardless of market conditions. This causes:
- Overtrading in high-volatility periods
- Missing opportunities in trending markets
- Large drawdowns during crashes

**Expected Impact**: -30% max drawdown, +15% risk-adjusted returns

### 3.2 Implementation

#### Step 1: Create Regime Detector

Create new file: `regime_detector.py`

```python
"""
Market Regime Detection Module
Detects market regimes (normal, elevated, crisis) using turbulence index.
Based on: Ensemble DRL Paper + Pipeline Architecture Paper
"""

import numpy as np
import pandas as pd
from typing import Literal, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    """Market regime classification."""
    NORMAL = "normal"
    ELEVATED = "elevated"  
    CRISIS = "crisis"


@dataclass
class RegimeConfig:
    """Configuration for regime thresholds and behavior."""
    # Turbulence thresholds (percentiles of historical distribution)
    normal_threshold: float = 0.75      # 75th percentile
    elevated_threshold: float = 0.90    # 90th percentile
    
    # Temperature adjustments
    normal_temp_mult: float = 1.0
    elevated_temp_mult: float = 1.3
    crisis_temp_mult: float = 1.8
    
    # Position sizing adjustments (for trading)
    normal_position_mult: float = 1.0
    elevated_position_mult: float = 0.7
    crisis_position_mult: float = 0.3
    
    # Lookback for regime calculation
    lookback: int = 60


class TurbulenceIndexCalculator:
    """
    Calculates Kritzman-Li turbulence index.
    
    Turbulence measures the statistical unusualness of returns relative to
    recent history. High turbulence = market stress.
    
    Formula: d_t = (r_t - mu)^T * Sigma^{-1} * (r_t - mu)
    where r_t = return vector, mu = mean, Sigma = covariance
    """
    
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self._history: list = []
        self._percentiles: Dict[float, float] = {}
    
    def update(self, returns: np.ndarray) -> float:
        """
        Calculate turbulence for current return vector.
        
        Args:
            returns: Vector of returns (can be single asset or multi-asset)
            
        Returns:
            Turbulence score (Mahalanobis distance)
        """
        self._history.append(returns)
        
        # Need enough history
        if len(self._history) < self.lookback:
            return 0.0
        
        # Keep only recent history
        self._history = self._history[-self.lookback:]
        
        # Calculate mean and covariance
        history_array = np.array(self._history)
        mean = np.mean(history_array, axis=0)
        
        # Regularized covariance (add small diagonal for stability)
        cov = np.cov(history_array.T)
        cov += np.eye(cov.shape[0]) * 1e-6
        
        # Mahalanobis distance
        diff = returns - mean
        try:
            inv_cov = np.linalg.inv(cov)
            turbulence = np.sqrt(diff @ inv_cov @ diff)
        except np.linalg.LinAlgError:
            # Fallback to Euclidean distance if singular
            turbulence = np.linalg.norm(diff)
        
        return float(turbulence)
    
    def calibrate_percentiles(self, historical_turbulence: np.ndarray):
        """
        Calibrate percentile thresholds from historical data.
        
        Args:
            historical_turbulence: Array of historical turbulence values
        """
        self._percentiles = {
            0.50: np.percentile(historical_turbulence, 50),
            0.75: np.percentile(historical_turbulence, 75),
            0.90: np.percentile(historical_turbulence, 90),
            0.95: np.percentile(historical_turbulence, 95),
        }


class MarketRegimeDetector:
    """
    Detects market regime based on turbulence and volatility indicators.
    
    Integrates multiple signals:
    1. Turbulence index (statistical unusualness)
    2. ATR percentile (volatility level)
    3. VIX proxy (if available)
    """
    
    def __init__(self, config: RegimeConfig = None):
        self.config = config or RegimeConfig()
        self.turbulence_calc = TurbulenceIndexCalculator(self.config.lookback)
        self._atr_history: list = []
        self._current_regime: MarketRegime = MarketRegime.NORMAL
    
    def detect_regime(self, returns: np.ndarray, atr: float) -> MarketRegime:
        """
        Detect current market regime.
        
        Args:
            returns: Current return vector
            atr: Current ATR value
            
        Returns:
            MarketRegime classification
        """
        # Update turbulence
        turbulence = self.turbulence_calc.update(returns)
        
        # Update ATR history
        self._atr_history.append(atr)
        self._atr_history = self._atr_history[-self.config.lookback:]
        
        # Need enough history
        if len(self._atr_history) < self.config.lookback:
            return MarketRegime.NORMAL
        
        # Calculate percentiles
        atr_percentile = np.mean(np.array(self._atr_history) <= atr)
        
        # Get turbulence percentile (if calibrated)
        turb_percentile = 0.5  # default
        if self.turbulence_calc._percentiles:
            turb_percentile = np.mean([
                turbulence <= v for v in self.turbulence_calc._percentiles.values()
            ])
        
        # Combined regime detection
        # Crisis: both high turbulence AND high volatility
        # Elevated: either high turbulence OR high volatility
        # Normal: neither
        
        if turb_percentile > self.config.elevated_threshold and atr_percentile > self.config.elevated_threshold:
            regime = MarketRegime.CRISIS
        elif turb_percentile > self.config.normal_threshold or atr_percentile > self.config.normal_threshold:
            regime = MarketRegime.ELEVATED
        else:
            regime = MarketRegime.NORMAL
        
        self._current_regime = regime
        return regime
    
    def get_temperature_multiplier(self) -> float:
        """Get temperature adjustment for current regime."""
        multipliers = {
            MarketRegime.NORMAL: self.config.normal_temp_mult,
            MarketRegime.ELEVATED: self.config.elevated_temp_mult,
            MarketRegime.CRISIS: self.config.crisis_temp_mult
        }
        return multipliers.get(self._current_regime, 1.0)
    
    def get_position_multiplier(self) -> float:
        """Get position sizing adjustment for current regime."""
        multipliers = {
            MarketRegime.NORMAL: self.config.normal_position_mult,
            MarketRegime.ELEVATED: self.config.elevated_position_mult,
            MarketRegime.CRISIS: self.config.crisis_position_mult
        }
        return multipliers.get(self._current_regime, 1.0)
    
    def should_halt_trading(self) -> bool:
        """Check if trading should be halted (crisis regime)."""
        return self._current_regime == MarketRegime.CRISIS


def calculate_historical_turbulence(df: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """
    Calculate historical turbulence for entire DataFrame.
    
    Args:
        df: DataFrame with returns column
        lookback: Lookback window for turbulence calculation
        
    Returns:
        Series of turbulence values
    """
    calculator = TurbulenceIndexCalculator(lookback)
    turbulence = []
    
    for i in range(len(df)):
        if i < lookback:
            turbulence.append(0.0)
        else:
            ret = df['returns'].iloc[i]
            turb = calculator.update(np.array([ret]))
            turbulence.append(turb)
    
    return pd.Series(turbulence, index=df.index)
```

#### Step 2: Integrate into Prediction Pipeline

```python
# In your prediction function
def predict_with_regime_awareness(model, frame, regime_detector):
    """
    Make predictions with regime-aware temperature adjustment.
    """
    # Detect regime
    recent_returns = frame['returns'].values[-regime_detector.config.lookback:]
    current_atr = frame['atr_14'].iloc[-1]
    
    regime = regime_detector.detect_regime(
        returns=np.array([recent_returns[-1]]), 
        atr=current_atr
    )
    
    # Adjust temperature
    base_temp = 0.8
    temp_mult = regime_detector.get_temperature_multiplier()
    adjusted_temp = base_temp * temp_mult
    
    # Generate predictions with adjusted temperature
    predictions = generate_predictions(model, frame, temperature=adjusted_temp)
    
    return predictions, regime
```

### 3.3 Testing Regime Detection

```python
# test_regime_detector.py
import unittest
import numpy as np
import pandas as pd
from regime_detector import (
    MarketRegimeDetector, 
    TurbulenceIndexCalculator,
    MarketRegime,
    RegimeConfig
)


class TestRegimeDetection(unittest.TestCase):
    
    def setUp(self):
        """Create sample data with different regimes."""
        np.random.seed(42)
        
        # Normal regime: low volatility
        self.normal_returns = np.random.randn(100) * 0.01
        
        # Elevated regime: higher volatility
        self.elevated_returns = np.random.randn(100) * 0.03
        
        # Crisis regime: extreme moves
        self.crisis_returns = np.random.randn(100) * 0.08
        self.crisis_returns[50:55] = -0.15  # Crash event
    
    def test_turbulence_increases_with_volatility(self):
        """Test that turbulence increases with return volatility."""
        calc = TurbulenceIndexCalculator(lookback=20)
        
        # Calculate turbulence for different regimes
        normal_turb = [calc.update(np.array([r])) for r in self.normal_returns]
        elevated_turb = [calc.update(np.array([r])) for r in self.elevated_returns]
        
        # Elevated should have higher average turbulence
        self.assertGreater(np.mean(elevated_turb[20:]), np.mean(normal_turb[20:]))
    
    def test_regime_detection_normal(self):
        """Test detection of normal regime."""
        detector = MarketRegimeDetector()
        
        # Feed normal returns
        for r in self.normal_returns[:70]:
            regime = detector.detect_regime(np.array([r]), atr=0.01)
        
        # Should be normal (or at least not crisis)
        self.assertNotEqual(regime, MarketRegime.CRISIS)
    
    def test_regime_detection_crisis(self):
        """Test detection of crisis regime."""
        config = RegimeConfig(
            normal_threshold=0.6,  # Lower thresholds for testing
            elevated_threshold=0.8
        )
        detector = MarketRegimeDetector(config)
        
        # First establish baseline with normal returns
        for r in self.normal_returns[:70]:
            detector.detect_regime(np.array([r]), atr=0.01)
        
        # Then feed crisis returns
        for r in self.crisis_returns[:30]:
            regime = detector.detect_regime(np.array([r]), atr=0.08)
        
        # Should detect elevated or crisis
        self.assertIn(regime, [MarketRegime.ELEVATED, MarketRegime.CRISIS])
    
    def test_temperature_adjustment(self):
        """Test temperature multipliers are applied correctly."""
        detector = MarketRegimeDetector()
        
        # Manually set regime
        detector._current_regime = MarketRegime.NORMAL
        self.assertEqual(detector.get_temperature_multiplier(), 1.0)
        
        detector._current_regime = MarketRegime.ELEVATED
        self.assertEqual(detector.get_temperature_multiplier(), 1.3)
        
        detector._current_regime = MarketRegime.CRISIS
        self.assertEqual(detector.get_temperature_multiplier(), 1.8)


if __name__ == '__main__':
    unittest.main()
```

---

## 4. Phase 3: iTransformer Architecture

### 4.1 Why This Helps

**Source Paper**: FT-iTransformer (Paper 7)

Your current GRU struggles with:
1. **Long-range dependencies**: GRU forgets information from >50 steps ago
2. **Cross-variable relationships**: Each feature processed independently
3. **Parallelization**: GRU is sequential, slow to train

iTransformer solves these by:
1. **Attention across variables**: Models relationships between features (e.g., how RSI affects price)
2. **FFN across time**: Captures temporal patterns in parallel
3. **Invert dimensions**: [batch, time, features] → [batch, features, time]

**Expected Impact**: -9% RMSE, -10% MAE, better long-horizon predictions

### 4.2 Implementation

#### Step 1: Create iTransformer Encoder

Create new file: `itransformer.py`

```python
"""
iTransformer Encoder Module
Inverted Transformer for time series forecasting.
Based on: FT-iTransformer Paper (Time-Frequency Domain Collaborative Analysis)

Key Innovation: Attention across variables (not time), FFN across time.
This captures cross-variable relationships more effectively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class iTransformerEncoderLayer(nn.Module):
    """
    Single iTransformer encoder layer.
    
    Architecture:
    1. Variable Attention: Multi-head attention across feature dimension
    2. Time FFN: Feed-forward network across time dimension
    
    Why this works:
    - Financial indicators are interrelated (RSI affects MACD interpretation)
    - Attention captures these relationships
    - FFN captures temporal patterns efficiently
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        
        d_ff = d_ff or 4 * d_model
        
        # Layer 1: Attention across variables
        self.var_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Layer 2: FFN across time
        self.time_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: [batch, features, time, d_model] - transposed input
            
        Returns:
            [batch, features, time, d_model]
        """
        # Attention across variables
        # Reshape for attention: [batch*time, features, d_model]
        batch, features, time, d_model = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch * time, features, d_model)
        
        attn_out, _ = self.var_attention(x_reshaped, x_reshaped, x_reshaped)
        attn_out = attn_out.reshape(batch, time, features, d_model).permute(0, 2, 1, 3)
        
        x = self.norm1(x + self.dropout1(attn_out))
        
        # FFN across time (applied per feature)
        ffn_out = self.time_ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class iTransformerEncoder(nn.Module):
    """
    Complete iTransformer encoder stack.
    
    Replaces GRU/LSTM with attention-based architecture.
    """
    
    def __init__(
        self,
        input_size: int,      # Number of input features
        d_model: int = 128,   # Model dimension
        n_layers: int = 2,
        n_heads: int = 8,
        d_ff: int = None,
        dropout: float = 0.1,
        lookback: int = 60    # Input sequence length
    ):
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.lookback = lookback
        
        # Project input to d_model
        # Input: [batch, time, features] → [batch, features, time]
        self.input_projection = nn.Linear(lookback, d_model)
        
        # Positional encoding for time dimension
        self.pos_encoding = self._create_positional_encoding(lookback, d_model)
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            iTransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, max_len: int, d_model: int):
        """Create sinusoidal positional encodings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0).unsqueeze(0), requires_grad=False)  # [1, 1, time, d_model]
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: [batch, time, features] - input time series
            
        Returns:
            [batch, d_model] - encoded representation
        """
        batch, time, features = x.shape
        
        # Transpose: [batch, features, time]
        x = x.transpose(1, 2)
        
        # Project to d_model: [batch, features, d_model]
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x.unsqueeze(2)  # [batch, features, 1, d_model]
        x = x + self.pos_encoding[:, :, :1, :]
        
        # Expand to [batch, features, time, d_model] for processing
        x = x.expand(-1, -1, time, -1)
        
        # Apply encoder layers
        for layer in self.layers:
            x = layer(x)
        
        # Global average pooling over time
        x = x.mean(dim=2)  # [batch, features, d_model]
        
        # Final pooling over features
        x = x.mean(dim=1)  # [batch, d_model]
        
        return x


class HybridEncoder(nn.Module):
    """
    Hybrid encoder combining iTransformer with GRU.
    
    Why hybrid: iTransformer for cross-variable relationships,
    GRU for sequential temporal dynamics.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 8,
        lookback: int = 60
    ):
        super().__init__()
        
        # iTransformer for variable relationships
        self.itransformer = iTransformerEncoder(
            input_size=input_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            lookback=lookback
        )
        
        # GRU for temporal dynamics
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model + hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch, time, features]
            
        Returns:
            [batch, hidden_size] - fused representation
        """
        # iTransformer path
        itrans_out = self.itransformer(x)  # [batch, d_model]
        
        # GRU path
        gru_out, hidden = self.gru(x)  # [batch, time, hidden]
        gru_out = hidden[-1]  # [batch, hidden]
        
        # Fuse
        combined = torch.cat([itrans_out, gru_out], dim=-1)
        fused = self.fusion(combined)
        
        return fused
```

#### Step 2: Replace Encoder in Your Model

```python
# OLD: Pure GRU encoder
class Seq2SeqAttnGRU(nn.Module):
    def __init__(self, ...):
        self.encoder = nn.GRU(
            input_size=FEATURE_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

# NEW: Hybrid iTransformer + GRU
class Seq2SeqAttnGRU(nn.Module):
    def __init__(self, ...):
        self.encoder = HybridEncoder(
            input_size=FEATURE_DIM,
            hidden_size=HIDDEN_SIZE,
            d_model=128,
            n_layers=2,
            n_heads=8,
            lookback=LOOKBACK_WINDOW
        )
```

### 4.3 Testing iTransformer

```python
# test_itransformer.py
import unittest
import torch
from itransformer import iTransformerEncoder, HybridEncoder


class TestiTransformer(unittest.TestCase):
    
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 60
        self.n_features = 35
        self.d_model = 128
    
    def test_itransformer_output_shape(self):
        """Test iTransformer produces correct output shape."""
        encoder = iTransformerEncoder(
            input_size=self.n_features,
            d_model=self.d_model,
            n_layers=2,
            n_heads=8,
            lookback=self.seq_len
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.n_features)
        out = encoder(x)
        
        self.assertEqual(out.shape, (self.batch_size, self.d_model))
    
    def test_hybrid_encoder_output_shape(self):
        """Test hybrid encoder produces correct output shape."""
        encoder = HybridEncoder(
            input_size=self.n_features,
            hidden_size=128,
            d_model=128,
            n_layers=2,
            n_heads=8,
            lookback=self.seq_len
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.n_features)
        out = encoder(x)
        
        self.assertEqual(out.shape, (self.batch_size, 128))
    
    def test_itransformer_gradient_flow(self):
        """Test gradients flow through iTransformer."""
        encoder = iTransformerEncoder(
            input_size=self.n_features,
            d_model=self.d_model,
            n_layers=2,
            n_heads=8,
            lookback=self.seq_len
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.n_features, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertTrue((x.grad != 0).any())
    
    def test_attention_across_variables(self):
        """Test that attention captures variable relationships."""
        encoder = iTransformerEncoder(
            input_size=4,  # Small for testing
            d_model=32,
            n_layers=1,
            n_heads=2,
            lookback=10
        )
        
        # Create input where feature 0 and 1 are correlated
        x = torch.zeros(1, 10, 4)
        x[:, :, 0] = torch.randn(1, 10)
        x[:, :, 1] = x[:, :, 0] * 0.9  # Highly correlated
        x[:, :, 2] = torch.randn(1, 10)  # Uncorrelated
        x[:, :, 3] = torch.randn(1, 10)  # Uncorrelated
        
        out = encoder(x)
        
        # Output should be non-zero
        self.assertTrue((out != 0).any())


if __name__ == '__main__':
    unittest.main()
```

---

## 5. Phase 4: Time-Frequency Features

### 5.1 Why This Helps

**Source Paper**: FT-iTransformer (Paper 7)

Financial time series have patterns at multiple frequencies:
- **High frequency**: Intraday noise, microstructure effects
- **Medium frequency**: Daily/weekly trends
- **Low frequency**: Monthly/quarterly cycles

Your current model only sees time-domain patterns. Adding frequency domain:
1. Captures periodic patterns (e.g., day-of-week effects)
2. Identifies dominant frequencies in price movements
3. Filters noise via frequency selection

**Expected Impact**: +3-5% directional accuracy, better cycle detection

### 5.2 Implementation

#### Step 1: Create Frequency Feature Extractor

Add to `itransformer.py` or create `frequency_features.py`:

```python
"""
Frequency Domain Feature Extraction
Uses Short-Time Fourier Transform (STFT) to capture frequency patterns.
Based on: FT-iTransformer Paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyFeatureExtractor(nn.Module):
    """
    Extracts frequency domain features using STFT + CNN.
    
    Why STFT: Captures how frequency content changes over time.
    Why CNN: Learns to identify important frequency patterns.
    """
    
    def __init__(
        self,
        n_fft: int = 16,
        hop_length: int = 4,
        n_freq_bins: int = None,
        out_channels: int = 32
    ):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_freq_bins = n_freq_bins or (n_fft // 2 + 1)
        
        # 1D-CNN to process frequency features
        self.freq_cnn = nn.Sequential(
            nn.Conv1d(self.n_freq_bins, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)  # Global pooling
        )
        
    def forward(self, x):
        """
        Extract frequency features.
        
        Args:
            x: [batch, time, features]
            
        Returns:
            [batch, out_channels] - frequency features
        """
        batch, time, features = x.shape
        
        freq_features = []
        
        for f in range(features):
            signal = x[:, :, f]  # [batch, time]
            
            # Apply STFT
            stft = torch.stft(
                signal,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True,
                center=True
            )
            
            # Magnitude spectrum
            magnitude = torch.abs(stft)  # [batch, freq, time_frames]
            freq_features.append(magnitude)
        
        # Stack: [batch, features, freq, time_frames]
        freq_features = torch.stack(freq_features, dim=1)
        
        # Reshape for CNN: [batch, freq, features * time_frames]
        batch, features, freq, time_frames = freq_features.shape
        freq_features = freq_features.permute(0, 2, 1, 3).reshape(batch, freq, -1)
        
        # Apply CNN
        out = self.freq_cnn(freq_features)  # [batch, out_channels, 1]
        
        return out.squeeze(-1)  # [batch, out_channels]


class MultiScaleFrequencyExtractor(nn.Module):
    """
    Extracts frequency features at multiple scales.
    
    Different n_fft values capture different frequency resolutions:
    - Small n_fft: High time resolution, low frequency resolution
    - Large n_fft: Low time resolution, high frequency resolution
    """
    
    def __init__(
        self,
        n_ffts: list = [8, 16, 32],
        out_channels: int = 32
    ):
        super().__init__()
        
        self.extractors = nn.ModuleList([
            FrequencyFeatureExtractor(
                n_fft=n_fft,
                out_channels=out_channels // len(n_ffts)
            )
            for n_fft in n_ffts
        ])
        
        self.fusion = nn.Linear(out_channels, out_channels)
        
    def forward(self, x):
        """Extract and fuse multi-scale frequency features."""
        features = [extractor(x) for extractor in self.extractors]
        combined = torch.cat(features, dim=-1)
        return self.fusion(combined)
```

#### Step 2: Integrate with iTransformer

```python
class FTiTransformerEncoder(nn.Module):
    """
    FT-iTransformer: Time-Frequency domain collaborative analysis.
    Combines time-domain and frequency-domain features.
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 8,
        lookback: int = 60,
        use_frequency: bool = True,
        freq_out_channels: int = 32
    ):
        super().__init__()
        
        # Time-domain encoder (iTransformer)
        self.time_encoder = iTransformerEncoder(
            input_size=input_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            lookback=lookback
        )
        
        # Frequency-domain encoder
        self.use_frequency = use_frequency
        if use_frequency:
            self.freq_encoder = MultiScaleFrequencyExtractor(
                n_ffts=[8, 16, 32],
                out_channels=freq_out_channels
            )
            
            # Fusion layer
            self.fusion = nn.Sequential(
                nn.Linear(d_model + freq_out_channels, d_model),
                nn.LayerNorm(d_model),
                nn.GELU()
            )
        
    def forward(self, x):
        """
        Args:
            x: [batch, time, features]
            
        Returns:
            [batch, d_model] - fused representation
        """
        # Time features
        time_features = self.time_encoder(x)
        
        if self.use_frequency:
            # Frequency features
            freq_features = self.freq_encoder(x)
            
            # Fuse
            combined = torch.cat([time_features, freq_features], dim=-1)
            output = self.fusion(combined)
        else:
            output = time_features
        
        return output
```

### 5.3 Testing Frequency Features

```python
# test_frequency_features.py
import unittest
import torch
from frequency_features import FrequencyFeatureExtractor, MultiScaleFrequencyExtractor


class TestFrequencyFeatures(unittest.TestCase):
    
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 60
        self.n_features = 10
    
    def test_stft_output_shape(self):
        """Test STFT produces correct output shape."""
        extractor = FrequencyFeatureExtractor(n_fft=16, out_channels=32)
        
        x = torch.randn(self.batch_size, self.seq_len, self.n_features)
        out = extractor(x)
        
        self.assertEqual(out.shape, (self.batch_size, 32))
    
    def test_frequency_captures_periodicity(self):
        """Test that frequency features capture periodic signals."""
        extractor = FrequencyFeatureExtractor(n_fft=16, out_channels=16)
        
        # Create periodic signal
        t = torch.linspace(0, 4 * 3.14159, self.seq_len)
        periodic = torch.sin(t).unsqueeze(0).unsqueeze(-1)  # [1, time, 1]
        periodic = periodic.repeat(self.batch_size, 1, self.n_features)
        
        # Create random signal
        random = torch.randn(self.batch_size, self.seq_len, self.n_features)
        
        out_periodic = extractor(periodic)
        out_random = extractor(random)
        
        # Periodic should have different frequency signature
        self.assertFalse(torch.allclose(out_periodic, out_random))
    
    def test_multi_scale_extractor(self):
        """Test multi-scale frequency extraction."""
        extractor = MultiScaleFrequencyExtractor(
            n_ffts=[8, 16, 32],
            out_channels=32
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.n_features)
        out = extractor(x)
        
        self.assertEqual(out.shape, (self.batch_size, 32))


if __name__ == '__main__':
    unittest.main()
```

---

## 6. Phase 5: RAG Retrieval System

### 6.1 Why This Helps

**Source Paper**: FinSrag (Paper 1)

Your current model has **no memory**. It sees only the last 60 timesteps. Humans trading stocks remember:
- "This pattern looks like the 2020 crash"
- "RSI divergence preceded the last 3 rallies"
- "Similar MACD crossovers led to breakouts"

RAG adds this memory by:
1. Encoding historical windows into embeddings
2. Retrieving similar past patterns
3. Conditioning predictions on retrieved context

**Expected Impact**: +10-15% directional accuracy, better handling of rare events

### 6.2 Implementation

#### Step 1: Create Pattern Retriever

Create new file: `pattern_retriever.py`

```python
"""
Pattern Retrieval Module (RAG for Time Series)
Retrieves similar historical patterns to augment predictions.
Based on: FinSrag Paper (Retrieval-Augmented LLMs for Financial Forecasting)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RetrievedPattern:
    """A retrieved historical pattern."""
    embedding: torch.Tensor
    sequence: torch.Tensor
    timestamp: int
    similarity: float
    future_return: float  # What happened after this pattern


class PatternEncoder(nn.Module):
    """
    Encodes time series windows into embedding vectors.
    
    Uses a lightweight LSTM to compress sequence into fixed-size embedding.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        embedding_dim: int = 64,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Tanh()  # Normalize to [-1, 1] for cosine similarity
        )
        
    def forward(self, x):
        """
        Encode sequence to embedding.
        
        Args:
            x: [batch, time, features]
            
        Returns:
            [batch, embedding_dim]
        """
        _, (hidden, _) = self.lstm(x)
        
        # Use last layer hidden state
        last_hidden = hidden[-1]  # [batch, hidden_size]
        
        embedding = self.projection(last_hidden)
        
        return embedding


class PatternDatabase:
    """
    Stores and indexes historical patterns for fast retrieval.
    
    Uses FAISS for efficient similarity search.
    """
    
    def __init__(self, embedding_dim: int = 64, use_gpu: bool = False):
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        
        # FAISS index for fast similarity search
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product = cosine for normalized vectors
        
        if use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        
        # Store metadata
        self.sequences: List[torch.Tensor] = []
        self.timestamps: List[int] = []
        self.future_returns: List[float] = []
        
    def add_patterns(
        self,
        embeddings: torch.Tensor,
        sequences: torch.Tensor,
        timestamps: List[int],
        future_returns: List[float]
    ):
        """
        Add patterns to the database.
        
        Args:
            embeddings: [N, embedding_dim] - pattern embeddings
            sequences: [N, time, features] - actual sequences
            timestamps: List of timestamps
            future_returns: List of future returns for each pattern
        """
        # Normalize embeddings for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Add to FAISS index
        self.index.add(embeddings.cpu().numpy())
        
        # Store metadata
        self.sequences.extend([s for s in sequences])
        self.timestamps.extend(timestamps)
        self.future_returns.extend(future_returns)
        
    def search(
        self,
        query_embedding: torch.Tensor,
        k: int = 5
    ) -> List[RetrievedPattern]:
        """
        Search for k most similar patterns.
        
        Args:
            query_embedding: [embedding_dim] or [batch, embedding_dim]
            k: Number of patterns to retrieve
            
        Returns:
            List of RetrievedPattern objects
        """
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        
        # Normalize query
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
        
        # Search
        similarities, indices = self.index.search(
            query_embedding.cpu().numpy(), 
            k
        )
        
        # Build results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0 or idx >= len(self.sequences):
                continue
                
            results.append(RetrievedPattern(
                embedding=None,  # Not stored for efficiency
                sequence=self.sequences[idx],
                timestamp=self.timestamps[idx],
                similarity=float(sim),
                future_return=self.future_returns[idx]
            ))
        
        return results
    
    def size(self) -> int:
        """Return number of patterns in database."""
        return self.index.ntotal


class RAGPatternRetriever(nn.Module):
    """
    Complete RAG retrieval system for time series.
    
    Integrates encoder, database, and retrieval logic.
    """
    
    def __init__(
        self,
        input_size: int,
        embedding_dim: int = 64,
        k_retrieve: int = 5,
        use_frequency_weighting: bool = True
    ):
        super().__init__()
        
        self.k_retrieve = k_retrieve
        self.use_frequency_weighting = use_frequency_weighting
        
        # Pattern encoder
        self.encoder = PatternEncoder(
            input_size=input_size,
            embedding_dim=embedding_dim
        )
        
        # Pattern database (initialized empty)
        self.database: Optional[PatternDatabase] = None
        
        # Fusion layer for retrieved patterns
        self.retrieval_fusion = nn.Sequential(
            nn.Linear(embedding_dim * k_retrieve, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, embedding_dim)
        )
        
    def build_database(
        self,
        historical_data: torch.Tensor,
        window_size: int = 60,
        future_horizon: int = 5
    ):
        """
        Build pattern database from historical data.
        
        Args:
            historical_data: [total_time, features] - full historical series
            window_size: Size of pattern windows
            future_horizon: How far ahead to calculate future returns
        """
        self.database = PatternDatabase(
            embedding_dim=self.encoder.projection[0].out_features
        )
        
        embeddings = []
        sequences = []
        timestamps = []
        future_returns = []
        
        # Slide window over historical data
        for i in range(window_size, len(historical_data) - future_horizon):
            window = historical_data[i-window_size:i]
            future = historical_data[i:i+future_horizon, 3]  # Close prices
            
            # Calculate future return
            ret = (future[-1] / historical_data[i-1, 3] - 1).item()
            
            # Encode
            with torch.no_grad():
                emb = self.encoder(window.unsqueeze(0)).squeeze(0)
            
            embeddings.append(emb)
            sequences.append(window)
            timestamps.append(i)
            future_returns.append(ret)
        
        # Add to database
        self.database.add_patterns(
            embeddings=torch.stack(embeddings),
            sequences=torch.stack(sequences),
            timestamps=timestamps,
            future_returns=future_returns
        )
        
        print(f"Built database with {self.database.size()} patterns")
        
    def retrieve_and_fuse(
        self,
        query_sequence: torch.Tensor
    ) -> Tuple[torch.Tensor, List[RetrievedPattern]]:
        """
        Retrieve similar patterns and fuse them into a context vector.
        
        Args:
            query_sequence: [batch, time, features] or [time, features]
            
        Returns:
            (fused_context, retrieved_patterns)
        """
        if self.database is None or self.database.size() == 0:
            # No database, return zero context
            embedding_dim = self.encoder.projection[0].out_features
            return torch.zeros(query_sequence.size(0) if query_sequence.dim() == 3 else 1, embedding_dim), []
        
        if query_sequence.dim() == 2:
            query_sequence = query_sequence.unsqueeze(0)
        
        # Encode query
        query_embedding = self.encoder(query_sequence)
        
        # Retrieve for each in batch
        all_contexts = []
        all_patterns = []
        
        for q_emb in query_embedding:
            patterns = self.database.search(q_emb, k=self.k_retrieve)
            all_patterns.append(patterns)
            
            # Encode retrieved patterns
            if patterns:
                # Use encoder to get embeddings of retrieved sequences
                retrieved_seqs = torch.stack([p.sequence for p in patterns])
                with torch.no_grad():
                    retrieved_embs = self.encoder(retrieved_seqs)
                
                # Weight by similarity
                weights = torch.tensor([p.similarity for p in patterns], 
                                      device=retrieved_embs.device)
                weights = F.softmax(weights, dim=0)
                
                # Weighted fusion
                weighted_embs = retrieved_embs * weights.unsqueeze(1)
                fused = weighted_embs.sum(dim=0)
            else:
                fused = torch.zeros_like(q_emb)
            
            all_contexts.append(fused)
        
        return torch.stack(all_contexts), all_patterns


class RAGAugmentedModel(nn.Module):
    """
    Your existing model augmented with RAG retrieval.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        input_size: int,
        embedding_dim: int = 64,
        k_retrieve: int = 5
    ):
        super().__init__()
        
        self.base_model = base_model
        self.retriever = RAGPatternRetriever(
            input_size=input_size,
            embedding_dim=embedding_dim,
            k_retrieve=k_retrieve
        )
        
        # Fusion layer to combine base model output with retrieval context
        hidden_size = base_model.decoder_hidden_size  # Adjust based on your model
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_size + embedding_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
    def forward(self, x, use_retrieval: bool = True):
        """
        Forward pass with optional retrieval augmentation.
        
        Args:
            x: [batch, time, features]
            use_retrieval: Whether to use RAG
            
        Returns:
            Model predictions
        """
        # Get base model encoding
        base_encoding = self.base_model.encode(x)  # Adjust based on your model
        
        if use_retrieval and self.retriever.database is not None:
            # Retrieve similar patterns
            retrieval_context, patterns = self.retriever.retrieve_and_fuse(x)
            
            # Fuse
            combined = torch.cat([base_encoding, retrieval_context], dim=-1)
            fused_encoding = self.context_fusion(combined)
        else:
            fused_encoding = base_encoding
        
        # Continue with decoder
        predictions = self.base_model.decode(fused_encoding)
        
        return predictions
```

### 6.3 Testing RAG Retrieval

```python
# test_pattern_retriever.py
import unittest
import torch
import numpy as np
from pattern_retriever import (
    PatternEncoder,
    PatternDatabase,
    RAGPatternRetriever
)


class TestPatternRetriever(unittest.TestCase):
    
    def setUp(self):
        self.seq_len = 60
        self.n_features = 10
        self.embedding_dim = 64
    
    def test_encoder_output_shape(self):
        """Test encoder produces correct embedding shape."""
        encoder = PatternEncoder(
            input_size=self.n_features,
            embedding_dim=self.embedding_dim
        )
        
        x = torch.randn(4, self.seq_len, self.n_features)
        emb = encoder(x)
        
        self.assertEqual(emb.shape, (4, self.embedding_dim))
    
    def test_encoder_normalization(self):
        """Test encoder outputs are normalized."""
        encoder = PatternEncoder(
            input_size=self.n_features,
            embedding_dim=self.embedding_dim
        )
        
        x = torch.randn(4, self.seq_len, self.n_features)
        emb = encoder(x)
        
        # Should be in [-1, 1] due to Tanh
        self.assertTrue((emb >= -1).all() and (emb <= 1).all())
    
    def test_database_search(self):
        """Test database retrieves similar patterns."""
        db = PatternDatabase(embedding_dim=self.embedding_dim)
        
        # Add some patterns
        n_patterns = 100
        embeddings = torch.randn(n_patterns, self.embedding_dim)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        sequences = [torch.randn(self.seq_len, self.n_features) for _ in range(n_patterns)]
        timestamps = list(range(n_patterns))
        future_returns = [np.random.randn() for _ in range(n_patterns)]
        
        db.add_patterns(embeddings, sequences, timestamps, future_returns)
        
        # Search
        query = torch.randn(1, self.embedding_dim)
        query = torch.nn.functional.normalize(query, p=2, dim=1)
        
        results = db.search(query.squeeze(), k=5)
        
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r.similarity > 0 for r in results))
    
    def test_similar_patterns_retrieved(self):
        """Test that similar patterns have higher similarity scores."""
        db = PatternDatabase(embedding_dim=self.embedding_dim)
        
        # Create two groups of patterns
        # Group 1: Similar to each other
        base_pattern = torch.randn(self.embedding_dim)
        group1 = [base_pattern + torch.randn(self.embedding_dim) * 0.1 
                  for _ in range(10)]
        
        # Group 2: Different
        group2 = [torch.randn(self.embedding_dim) for _ in range(10)]
        
        all_embeddings = torch.stack(group1 + group2)
        all_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)
        
        sequences = [torch.randn(self.seq_len, self.n_features) for _ in range(20)]
        timestamps = list(range(20))
        future_returns = [0.0] * 20
        
        db.add_patterns(all_embeddings, sequences, timestamps, future_returns)
        
        # Query with pattern similar to group 1
        query = base_pattern + torch.randn(self.embedding_dim) * 0.05
        query = torch.nn.functional.normalize(query.unsqueeze(0), p=2, dim=1)
        
        results = db.search(query.squeeze(), k=5)
        
        # Top results should mostly be from group 1 (indices 0-9)
        top_indices = [r.timestamp for r in results]
        group1_matches = sum(1 for idx in top_indices if idx < 10)
        
        self.assertGreater(group1_matches, 2)  # Most should be from group 1


if __name__ == '__main__':
    unittest.main()
```

---

## 7. Phase 6: RL Decision Layer

### 7.1 Why This Helps

**Source Papers**: Ensemble DRL (Paper 3) + Pipeline.docx (Paper 6)

Your current model predicts prices but doesn't decide:
- How much to trade
- When to enter/exit
- How to manage risk

An RL decision layer:
1. Learns optimal trading policy from predictions
2. Maximizes risk-adjusted returns (Sharpe ratio)
3. Handles transaction costs and slippage
4. Adapts to market conditions

**Expected Impact**: +50% risk-adjusted returns, -30% drawdown

### 7.2 Implementation

#### Step 1: Create RL Trading Environment

Create new file: `trading_env.py`

```python
"""
Reinforcement Learning Trading Environment
Implements MDP for stock trading with predictions as state.
Based on: Ensemble DRL Paper + Pipeline Architecture Paper
"""

import numpy as np
import torch
import gym
from gym import spaces
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: int
    action: str  # 'buy', 'sell', 'hold'
    shares: float
    price: float
    cost: float


class StockTradingEnv(gym.Env):
    """
    Stock trading environment for RL training.
    
    State: Model predictions + market state + portfolio state
    Action: Continuous position sizing [-1, 1] (short to long)
    Reward: Differential Sharpe Ratio (step-by-step Sharpe)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,  # Model predictions [time, horizon, 4]
        initial_balance: float = 100000,
        transaction_cost: float = 0.001,  # 0.1%
        max_position: float = 1.0,  # Max 100% of portfolio
        lookback: int = 60,
        use_turbulence: bool = True
    ):
        super().__init__()
        
        self.df = df
        self.predictions = predictions
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.lookback = lookback
        self.use_turbulence = use_turbulence
        
        # State dimension
        self.state_dim = self._calculate_state_dim()
        
        # Action space: continuous position sizing
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        
        # Portfolio state
        self.balance = initial_balance
        self.shares = 0
        self.portfolio_value = initial_balance
        self.trades: list = []
        
        # For DSR calculation
        self.returns_history: list = []
        self.A = 0  # EMA of returns
        self.B = 0  # EMA of squared returns
        self.dsr_eta = 0.1  # DSR decay rate
        
        # Turbulence detector
        if use_turbulence:
            from regime_detector import MarketRegimeDetector
            self.regime_detector = MarketRegimeDetector()
        
        self.current_step = 0
        
    def _calculate_state_dim(self) -> int:
        """Calculate state vector dimension."""
        # Model predictions: horizon * 4 (OHLC)
        pred_dim = self.predictions.shape[1] * 4
        
        # Market state: technical indicators
        market_dim = 10  # RSI, MACD, BB position, ATR, etc.
        
        # Portfolio state
        portfolio_dim = 4  # balance, shares, portfolio_value, position_pct
        
        # Regime
        regime_dim = 1 if self.use_turbulence else 0
        
        return pred_dim + market_dim + portfolio_dim + regime_dim
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.balance = self.initial_balance
        self.shares = 0
        self.portfolio_value = self.initial_balance
        self.trades = []
        self.returns_history = []
        self.A = 0
        self.B = 0
        self.current_step = self.lookback
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Construct state vector."""
        # Model predictions
        pred = self.predictions[self.current_step].flatten()
        
        # Market state
        market = self._get_market_state()
        
        # Portfolio state
        portfolio = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.shares / 1000,  # Normalized shares
            self.portfolio_value / self.initial_balance,  # Normalized value
            (self.shares * self.df['Close'].iloc[self.current_step]) / self.portfolio_value
        ])
        
        # Regime
        if self.use_turbulence:
            regime = np.array([self._get_regime_indicator()])
            return np.concatenate([pred, market, portfolio, regime])
        
        return np.concatenate([pred, market, portfolio])
    
    def _get_market_state(self) -> np.ndarray:
        """Extract current market state features."""
        idx = self.current_step
        return np.array([
            self.df['rsi_14'].iloc[idx] / 100,  # Normalized RSI
            self.df['macd_histogram'].iloc[idx],  # MACD momentum
            self.df['bb_position'].iloc[idx],  # Bollinger position
            self.df['atr_14_pct'].iloc[idx],  # ATR as %
            self.df['price_momentum_5'].iloc[idx],  # 5-day momentum
            self.df['price_momentum_20'].iloc[idx],  # 20-day momentum
            self.df['vwap_20_dev'].iloc[idx],  # VWAP deviation
            self.df['obv_slope'].iloc[idx] / 1e6,  # OBV slope
            self.df['direction'].iloc[idx],  # Last candle direction
            self.df['volume'].iloc[idx] / self.df['volume'].rolling(20).mean().iloc[idx]  # Volume ratio
        ])
    
    def _get_regime_indicator(self) -> float:
        """Get current regime as numeric indicator."""
        if not hasattr(self, 'regime_detector'):
            return 0.0
        
        # Returns 0 for normal, 0.5 for elevated, 1.0 for crisis
        regime_map = {'normal': 0.0, 'elevated': 0.5, 'crisis': 1.0}
        return regime_map.get(self.regime_detector._current_regime.value, 0.0)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one trading step.
        
        Args:
            action: Position target [-1, 1] where -1 = max short, 1 = max long
            
        Returns:
            (observation, reward, done, info)
        """
        # Current price
        current_price = self.df['Close'].iloc[self.current_step]
        
        # Target position value
        target_position_value = action[0] * self.max_position * self.portfolio_value
        
        # Current position value
        current_position_value = self.shares * current_price
        
        # Trade needed
        trade_value = target_position_value - current_position_value
        
        # Execute trade
        if abs(trade_value) > 10:  # Min trade size
            shares_to_trade = trade_value / current_price
            trade_cost = abs(trade_value) * self.transaction_cost
            
            if shares_to_trade > 0:  # Buy
                cost = shares_to_trade * current_price + trade_cost
                if cost <= self.balance:
                    self.shares += shares_to_trade
                    self.balance -= cost
                    self.trades.append(Trade(
                        timestamp=self.current_step,
                        action='buy',
                        shares=shares_to_trade,
                        price=current_price,
                        cost=trade_cost
                    ))
            else:  # Sell
                shares_to_sell = min(abs(shares_to_trade), self.shares)
                if shares_to_sell > 0:
                    proceeds = shares_to_sell * current_price - trade_cost
                    self.shares -= shares_to_sell
                    self.balance += proceeds
                    self.trades.append(Trade(
                        timestamp=self.current_step,
                        action='sell',
                        shares=shares_to_sell,
                        price=current_price,
                        cost=trade_cost
                    ))
        
        # Update portfolio value
        self.portfolio_value = self.balance + self.shares * current_price
        
        # Calculate return
        if len(self.returns_history) > 0:
            ret = (self.portfolio_value / (self.balance + self.shares * self.df['Close'].iloc[self.current_step - 1])) - 1
        else:
            ret = 0
        
        self.returns_history.append(ret)
        
        # Calculate DSR reward
        reward = self._calculate_dsr_reward(ret)
        
        # Update regime detector
        if self.use_turbulence:
            self.regime_detector.detect_regime(
                np.array([self.df['returns'].iloc[self.current_step]]),
                self.df['atr_14'].iloc[self.current_step]
            )
        
        # Advance
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        info = {
            'portfolio_value': self.portfolio_value,
            'return': ret,
            'shares': self.shares,
            'balance': self.balance
        }
        
        return self._get_observation(), reward, done, info
    
    def _calculate_dsr_reward(self, ret: float) -> float:
        """
        Calculate Differential Sharpe Ratio (DSR) as reward.
        
        DSR gives immediate feedback on how each action affects the Sharpe ratio,
        rather than waiting until the end of episode.
        
        Formula: DSR_t = (B_{t-1} * r_t - 0.5 * A_{t-1} * r_t^2) / (B_{t-1} - A_{t-1}^2)^{3/2}
        where A = EMA(returns), B = EMA(returns^2)
        """
        if len(self.returns_history) < 2:
            return 0.0
        
        # Update EMAs
        A_prev = self.A
        B_prev = self.B
        
        self.A = A_prev + self.dsr_eta * (ret - A_prev)
        self.B = B_prev + self.dsr_eta * (ret**2 - B_prev)
        
        # Calculate DSR
        denominator = (B_prev - A_prev**2)**1.5
        if denominator < 1e-8:
            return 0.0
        
        dsr = (B_prev * ret - 0.5 * A_prev * ret**2) / denominator
        
        return dsr


class ActionMaskingWrapper(gym.ActionWrapper):
    """
    Masks actions during high turbulence (crisis regime).
    
    Instead of hard halts, mask logits to force 100% cash.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.mask_threshold = 0.8  # Crisis regime indicator
    
    def action(self, action):
        """Mask action if in crisis regime."""
        regime = self.env._get_regime_indicator()
        
        if regime >= self.mask_threshold:
            # Force neutral position (all cash)
            return np.array([0.0])
        
        return action
```

#### Step 2: Create PPO Agent

Create new file: `ppo_agent.py`:

```python
"""
PPO Agent for Trading
Implements Proximal Policy Optimization for trading decisions.
Based on: Ensemble DRL Paper + PPO Trading Paper
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Tuple, List


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    
    Actor: Outputs mean and std of action distribution
    Critic: Estimates state value
    """
    
    def __init__(self, state_dim: int, action_dim: int = 1, hidden_dim: int = 256):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        # Actor head
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        """Forward pass returning action distribution and value."""
        features = self.shared(state)
        
        # Actor
        mean = torch.tanh(self.actor_mean(features))  # Bound to [-1, 1]
        std = torch.exp(self.actor_log_std).expand_as(mean)
        
        # Critic
        value = self.critic(features)
        
        return mean, std, value
    
    def get_action(self, state, deterministic=False):
        """Sample action from policy."""
        mean, std, value = self.forward(state)
        
        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()
        
        log_prob = Normal(mean, std).log_prob(action).sum(dim=-1)
        
        return action, log_prob, value


class PPO:
    """
    Proximal Policy Optimization agent.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float
    ) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation.
        
        GAE balances bias (n-step returns) and variance (advantage).
        """
        advantages = []
        gae = 0
        
        values = values + [next_value]
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        
        return advantages, returns
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        epochs: int = 10,
        batch_size: int = 64
    ) -> dict:
        """
        Update policy using PPO clipped objective.
        """
        dataset_size = len(states)
        
        metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'approx_kl': []
        }
        
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(dataset_size)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                mean, std, values = self.policy(batch_states)
                dist = Normal(mean, std)
                
                # New log probs
                log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                
                # PPO clipped objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * ((values.squeeze() - batch_returns) ** 2).mean()
                
                # Entropy bonus
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - ratio.log()).mean()
                    
                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy'].append(entropy.item())
                metrics['approx_kl'].append(approx_kl.item())
        
        return {k: np.mean(v) for k, v in metrics.items()}


def train_ppo(
    env,
    ppo: PPO,
    total_timesteps: int = 100000,
    rollout_length: int = 2048,
    update_epochs: int = 10
):
    """
    Train PPO agent on trading environment.
    """
    state = env.reset()
    
    states_buf = []
    actions_buf = []
    rewards_buf = []
    values_buf = []
    log_probs_buf = []
    dones_buf = []
    
    episode_rewards = []
    
    for timestep in range(total_timesteps):
        # Get action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(ppo.device)
        with torch.no_grad():
            action, log_prob, value = ppo.policy.get_action(state_tensor)
        
        # Step environment
        next_state, reward, done, info = env.step(action.cpu().numpy()[0])
        
        # Store transition
        states_buf.append(state)
        actions_buf.append(action.cpu().numpy()[0])
        rewards_buf.append(reward)
        values_buf.append(value.item())
        log_probs_buf.append(log_prob.item())
        dones_buf.append(done)
        
        state = next_state
        
        if done:
            episode_rewards.append(info['portfolio_value'])
            state = env.reset()
        
        # Update if enough data
        if len(states_buf) >= rollout_length:
            # Get next value for GAE
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(ppo.device)
                _, _, next_value = ppo.policy(state_tensor)
                next_value = next_value.item()
            
            # Compute GAE
            advantages, returns = ppo.compute_gae(
                rewards_buf, values_buf, dones_buf, next_value
            )
            
            # Convert to tensors
            states_tensor = torch.FloatTensor(np.array(states_buf)).to(ppo.device)
            actions_tensor = torch.FloatTensor(np.array(actions_buf)).to(ppo.device)
            old_log_probs_tensor = torch.FloatTensor(log_probs_buf).to(ppo.device)
            advantages_tensor = torch.FloatTensor(advantages).to(ppo.device)
            returns_tensor = torch.FloatTensor(returns).to(ppo.device)
            
            # Normalize advantages
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
            
            # Update policy
            metrics = ppo.update(
                states_tensor,
                actions_tensor,
                old_log_probs_tensor,
                advantages_tensor,
                returns_tensor,
                epochs=update_epochs
            )
            
            print(f"Timestep {timestep}: {metrics}")
            
            # Clear buffers
            states_buf = []
            actions_buf = []
            rewards_buf = []
            values_buf = []
            log_probs_buf = []
            dones_buf = []
    
    return episode_rewards
```

### 7.3 Testing RL Components

```python
# test_trading_env.py
import unittest
import numpy as np
import pandas as pd
import torch
from trading_env import StockTradingEnv
from ppo_agent import ActorCritic, PPO


class TestTradingEnv(unittest.TestCase):
    
    def setUp(self):
        """Create sample data."""
        np.random.seed(42)
        n = 200
        
        self.df = pd.DataFrame({
            'Open': 100 + np.random.randn(n).cumsum(),
            'High': 102 + np.random.randn(n).cumsum(),
            'Low': 98 + np.random.randn(n).cumsum(),
            'Close': 100 + np.random.randn(n).cumsum(),
            'Volume': np.random.randint(1000, 10000, n),
            'returns': np.random.randn(n) * 0.01,
            'rsi_14': np.random.rand(n) * 100,
            'macd_histogram': np.random.randn(n),
            'bb_position': np.random.rand(n),
            'atr_14_pct': np.random.rand(n) * 0.05,
            'price_momentum_5': np.random.randn(n) * 0.01,
            'price_momentum_20': np.random.randn(n) * 0.02,
            'vwap_20_dev': np.random.randn(n) * 0.01,
            'obv_slope': np.random.randn(n) * 1e6,
            'direction': np.random.choice([-1, 0, 1], n)
        })
        
        # Dummy predictions [time, horizon, 4]
        self.predictions = np.random.randn(n, 5, 4) * 0.01 + 100
    
    def test_env_initialization(self):
        """Test environment initializes correctly."""
        env = StockTradingEnv(self.df, self.predictions)
        
        self.assertEqual(env.balance, 100000)
        self.assertEqual(env.shares, 0)
    
    def test_buy_action(self):
        """Test buying increases shares."""
        env = StockTradingEnv(self.df, self.predictions)
        
        initial_balance = env.balance
        
        # Buy action
        obs, reward, done, info = env.step(np.array([0.5]))  # 50% position
        
        self.assertGreater(env.shares, 0)
        self.assertLess(env.balance, initial_balance)
    
    def test_sell_action(self):
        """Test selling decreases shares."""
        env = StockTradingEnv(self.df, self.predictions)
        
        # First buy
        env.step(np.array([0.5]))
        shares_after_buy = env.shares
        
        # Then sell
        obs, reward, done, info = env.step(np.array([-0.5]))
        
        self.assertLess(env.shares, shares_after_buy)
    
    def test_transaction_costs(self):
        """Test transaction costs are applied."""
        env = StockTradingEnv(self.df, self.predictions, transaction_cost=0.01)
        
        initial_value = env.portfolio_value
        
        # Execute trades
        for _ in range(10):
            env.step(np.array([np.random.uniform(-1, 1)]))
        
        # Should have some trades with costs
        total_cost = sum(t.cost for t in env.trades)
        self.assertGreater(total_cost, 0)
    
    def test_dsr_calculation(self):
        """Test DSR reward is calculated."""
        env = StockTradingEnv(self.df, self.predictions)
        
        rewards = []
        for _ in range(20):
            obs, reward, done, info = env.step(np.array([0.1]))
            rewards.append(reward)
        
        # Rewards should be non-zero after warmup
        self.assertTrue(any(r != 0 for r in rewards[5:]))


class TestPPOAgent(unittest.TestCase):
    
    def test_actor_critic_output(self):
        """Test ActorCritic produces correct outputs."""
        model = ActorCritic(state_dim=50, action_dim=1)
        
        state = torch.randn(4, 50)
        mean, std, value = model(state)
        
        self.assertEqual(mean.shape, (4, 1))
        self.assertEqual(std.shape, (4, 1))
        self.assertEqual(value.shape, (4, 1))
        
        # Mean should be bounded
        self.assertTrue((mean >= -1).all() and (mean <= 1).all())
    
    def test_ppo_update(self):
        """Test PPO update runs without errors."""
        ppo = PPO(state_dim=50, action_dim=1)
        
        # Create dummy batch
        states = torch.randn(64, 50)
        actions = torch.randn(64, 1)
        old_log_probs = torch.randn(64)
        advantages = torch.randn(64)
        returns = torch.randn(64)
        
        metrics = ppo.update(
            states, actions, old_log_probs, advantages, returns,
            epochs=2, batch_size=32
        )
        
        self.assertIn('policy_loss', metrics)
        self.assertIn('value_loss', metrics)


if __name__ == '__main__':
    unittest.main()
```

---

## 8. Testing Framework

### 8.1 Integration Testing

Create `test_integration.py`:

```python
"""
Integration tests for the complete pipeline.
"""

import unittest
import torch
import pandas as pd
import numpy as np

from technical_indicators import calculate_technical_features
from regime_detector import MarketRegimeDetector
from itransformer import FTiTransformerEncoder
from pattern_retriever import RAGPatternRetriever
from trading_env import StockTradingEnv


class TestFullPipeline(unittest.TestCase):
    """Test complete prediction pipeline."""
    
    def setUp(self):
        """Create sample data."""
        np.random.seed(42)
        n = 500
        
        # Generate realistic price data
        returns = np.random.randn(n) * 0.02
        prices = 100 * np.exp(np.cumsum(returns))
        
        self.df = pd.DataFrame({
            'Open': prices * (1 - np.abs(np.random.randn(n) * 0.005)),
            'High': prices * (1 + np.abs(np.random.randn(n) * 0.005)),
            'Low': prices * (1 - np.abs(np.random.randn(n) * 0.008)),
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, n)
        })
        
        # Add technical indicators
        self.df = calculate_technical_features(self.df)
        self.df = self.df.fillna(0)
    
    def test_end_to_end_prediction(self):
        """Test full prediction pipeline."""
        # 1. Prepare features
        feature_cols = [c for c in self.df.columns if c not in ['Open', 'High', 'Low', 'Close']]
        features = self.df[feature_cols].values
        
        # 2. Create model
        model = FTiTransformerEncoder(
            input_size=len(feature_cols),
            d_model=64,
            n_layers=2,
            n_heads=4,
            lookback=60,
            use_frequency=True
        )
        
        # 3. Forward pass
        batch = torch.FloatTensor(features[:60]).unsqueeze(0)
        output = model(batch)
        
        self.assertEqual(output.shape, (1, 64))
    
    def test_regime_aware_prediction(self):
        """Test prediction with regime awareness."""
        detector = MarketRegimeDetector()
        
        # Simulate regime detection
        for i in range(60):
            detector.detect_regime(
                np.array([self.df['returns'].iloc[i]]),
                self.df['atr_14'].iloc[i]
            )
        
        # Get temperature adjustment
        temp_mult = detector.get_temperature_multiplier()
        self.assertGreaterEqual(temp_mult, 1.0)
    
    def test_rag_augmented_prediction(self):
        """Test prediction with RAG augmentation."""
        from pattern_retriever import RAGPatternRetriever
        
        feature_cols = [c for c in self.df.columns if c not in ['Open', 'High', 'Low', 'Close']]
        features = self.df[feature_cols].values
        
        retriever = RAGPatternRetriever(
            input_size=len(feature_cols),
            embedding_dim=32,
            k_retrieve=3
        )
        
        # Build database
        retriever.build_database(
            historical_data=torch.FloatTensor(features),
            window_size=60,
            future_horizon=5
        )
        
        # Retrieve
        query = torch.FloatTensor(features[60:120]).unsqueeze(0)
        context, patterns = retriever.retrieve_and_fuse(query)
        
        self.assertEqual(context.shape, (1, 32))
        self.assertEqual(len(patterns), 1)
        self.assertEqual(len(patterns[0]), 3)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics calculation."""
    
    def test_directional_accuracy(self):
        """Test directional accuracy calculation."""
        actual = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 2])
        predicted = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 2])
        
        # Calculate directional accuracy
        actual_direction = np.sign(np.diff(actual))
        pred_direction = np.sign(np.diff(predicted))
        accuracy = np.mean(actual_direction == pred_direction)
        
        self.assertEqual(accuracy, 1.0)  # Perfect prediction
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = np.random.randn(100) * 0.01
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        self.assertIsInstance(sharpe, float)
    
    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        prices = np.array([100, 110, 105, 120, 100, 90, 95, 100])
        
        # Calculate drawdown
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak
        max_dd = np.min(drawdown)
        
        self.assertLess(max_dd, 0)


if __name__ == '__main__':
    unittest.main()
```

### 8.2 Running All Tests

```bash
# Run all tests
python -m pytest test_*.py -v

# Run specific test file
python -m pytest test_indicators.py -v

# Run with coverage
python -m pytest test_*.py --cov=. --cov-report=html
```

---

## 9. Implementation Timeline

### Week 1-2: Foundation
| Day | Task | Files to Modify |
|-----|------|-----------------|
| 1-2 | Implement technical indicators | `technical_indicators.py` |
| 3-4 | Add indicators to data pipeline | Main training script |
| 5-7 | Test and validate indicators | `test_indicators.py` |
| 8-10 | Implement regime detector | `regime_detector.py` |
| 11-14 | Integrate regime detection | Main prediction script |

### Week 3-4: Architecture Upgrade
| Day | Task | Files to Modify |
|-----|------|-----------------|
| 15-17 | Implement iTransformer encoder | `itransformer.py` |
| 18-21 | Replace GRU with iTransformer | Model definition |
| 22-24 | Add frequency features | `frequency_features.py` |
| 25-28 | Test architecture changes | `test_itransformer.py` |

### Week 5-6: Advanced Features
| Day | Task | Files to Modify |
|-----|------|-----------------|
| 29-32 | Implement pattern retriever | `pattern_retriever.py` |
| 33-35 | Build pattern database | Data preprocessing |
| 36-38 | Integrate RAG into model | Model forward pass |
| 39-42 | Test RAG system | `test_pattern_retriever.py` |

### Week 7-8: RL Layer
| Day | Task | Files to Modify |
|-----|------|-----------------|
| 43-46 | Implement trading environment | `trading_env.py` |
| 47-49 | Implement PPO agent | `ppo_agent.py` |
| 50-52 | Train RL agent | Training script |
| 53-56 | Test complete system | `test_integration.py` |

---

## 10. Expected Improvements

| Metric | Current | After Phase 1 | After Phase 2 | After Phase 3 | Final |
|--------|---------|---------------|---------------|---------------|-------|
| Directional Accuracy | 48.7% | 53% | 57% | 62% | 65% |
| Close MAE | 0.30 | 0.26 | 0.23 | 0.20 | 0.18 |
| Sharpe Ratio (trading) | N/A | N/A | N/A | 0.8 | 1.2 |
| Max Drawdown | N/A | N/A | N/A | -15% | -10% |

---

## 11. Key Implementation Notes

### Memory Management
- Use gradient checkpointing for iTransformer if OOM
- Pre-compute pattern embeddings to save inference time
- Use FAISS GPU index for faster retrieval

### Training Tips
- Train technical indicator model first (Phase 1)
- Freeze encoder when adding RAG (Phase 3)
- Use curriculum learning for RL (start with easier markets)

### Debugging
- Log regime transitions during inference
- Visualize retrieved patterns to verify quality
- Monitor DSR values during RL training

---

## References

1. **FinSrag**: Retrieval-Augmented LLMs for Financial Time Series Forecasting (2502.05878v3)
2. **VTA**: Verbal Technical Analysis for Stock Forecasting (2511.08616v1)
3. **Ensemble DRL**: Deep Reinforcement Learning for Automated Stock Trading (2511.12120v1)
4. **LLM Sentiment**: Impact of LLM News Sentiment Analysis on Stock Price Prediction (2602.00086v2)
5. **PPO Trading**: PPO-Driven RL Model for Automatic Stock Trading (10103.pdf)
6. **Pipeline**: AI Stock Market Prediction Architecture Pipeline (Pipeline.docx)
7. **FT-iTransformer**: Time-Frequency Domain Collaborative Analysis (technologies-14-00061)
8. **RAG Thesis**: RAG for LLMs: Enhancing Economic Reasoning and Forecasting (quintero-sebq-meng-eecs-2025-thesis)
