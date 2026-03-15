# %% [markdown] cell 0
# Experiment: MSFT 1-Minute GRU Forecast with RL Decision Layer (Phase 6 v9.6)

Key changes for v9.6 - Reinforcement Learning Decision Layer:
1. **Trade dataclass**: Records all trading activity with timestamp, action, shares, price, cost
2. **StockTradingEnv**: Gym environment for RL training with:
   - State: predictions + market state + portfolio state + regime
   - Action: Continuous position sizing [-1, 1] (short to long)
   - Reward: Differential Sharpe Ratio (DSR)
   - Transaction costs and max position limits
3. **DSR calculation**: EMA-based differential Sharpe ratio for step-by-step feedback
4. **ActorCritic network**: Shared features with actor (mean/log_std) and critic (value) heads
5. **PPO Agent**: Proximal Policy Optimization with GAE and clipped objective
6. **Integration with rolling backtest**: RL agent makes position decisions based on model predictions

Base architecture preserved:
- Probabilistic outputs (mu + log_sigma) with sampling
- Autoregressive recursive generation (feed predictions back)
- Temperature-controlled noise injection for volatility
- Volatility matching to recent historical realized vol
- Strict candle validity enforcement at each step

# %% [markdown] cell 1
## Package Installation & Imports

# %% [code] cell 2
import importlib.util
import subprocess
import sys

required = {
    'alpaca': 'alpaca-py',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'pandas_market_calendars': 'pandas-market-calendars',
}
missing = [pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None]
if missing:
    print('Installing missing packages:', missing)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
else:
    print('All required third-party packages are already installed.')

# %% [code] cell 3
from __future__ import annotations
import copy
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg
from torch.utils.data import DataLoader, Dataset

print('Imports complete')

# %% [markdown] cell 4
## Random Seed & Device Setup

# %% [code] cell 5
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))

# %% [markdown] cell 6
## Configuration

# %% [code] cell 7
# Data Configuration
SYMBOL = 'MSFT'
LOOKBACK_DAYS = 120
OHLC_COLS = ['Open', 'High', 'Low', 'Close']
RAW_COLS = OHLC_COLS + ['Volume', 'TradeCount', 'VWAP']
BASE_FEATURE_COLS = [
    'rOpen', 'rHigh', 'rLow', 'rClose',
    'logVolChange', 'logTradeCountChange',
    'vwapDelta', 'rangeFrac', 'orderFlowProxy', 'tickPressure',
]
TARGET_COLS = ['rOpen', 'rHigh', 'rLow', 'rClose']
INPUT_EXTRA_COL = 'imputedFracWindow'

HORIZON = 50
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
LOOKBACK_CANDIDATES = [64, 96, 160, 256]
DEFAULT_LOOKBACK = 96
ENABLE_LOOKBACK_SWEEP = True
SKIP_OPEN_BARS_TARGET = 6

# %% [code] cell 8
# Model Configuration
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.20
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 256

# %% [code] cell 9
# Training Configuration
SWEEP_MAX_EPOCHS = 15
SWEEP_PATIENCE = 5
FINAL_MAX_EPOCHS = 60
FINAL_PATIENCE = 12
TF_START = 1.0
TF_END = 0.0
TF_DECAY_RATE = 0.95

# %% [code] cell 10
# Loss Configuration
RANGE_LOSS_WEIGHT = 0.3
VOLATILITY_WEIGHT = 0.5
DIR_PENALTY_WEIGHT = 0.1
STEP_LOSS_POWER = 1.5

# %% [code] cell 11
# Inference Configuration
SAMPLING_TEMPERATURE = 1.5
ENSEMBLE_SIZE = 20
TREND_LOOKBACK_BARS = 20
STRONG_TREND_THRESHOLD = 0.002
VOLATILITY_SCALING = True
MIN_PREDICTED_VOL = 0.0001

# %% [code] cell 12
# RL Configuration (Phase 6)
RL_CONFIG = {
    'initial_balance': 100000.0,
    'transaction_cost': 0.001,  # 0.1%
    'max_position': 1.0,  # Max 100% of portfolio
    'dsr_eta': 0.1,  # DSR decay rate
    'ppo_lr': 3e-4,
    'ppo_gamma': 0.99,
    'ppo_gae_lambda': 0.95,
    'ppo_clip_epsilon': 0.2,
    'ppo_value_coef': 0.5,
    'ppo_entropy_coef': 0.01,
    'ppo_max_grad_norm': 0.5,
    'ppo_rollout_length': 2048,
    'ppo_update_epochs': 10,
    'ppo_batch_size': 64,
    'rl_training_steps': 50000,
}
print('RL Configuration:', RL_CONFIG)

# %% [code] cell 13
# Data Processing Configuration
STANDARDIZE_TARGETS = False
APPLY_CLIPPING = True
CLIP_QUANTILES = (0.001, 0.999)
DIRECTION_EPS = 0.0001
STD_RATIO_TARGET_MIN = 0.3

# %% [code] cell 14
# Alpaca API Configuration
ALPACA_FEED = os.getenv('ALPACA_FEED', 'iex').strip().lower()
SESSION_TZ = 'America/New_York'
REQUEST_CHUNK_DAYS = 5
MAX_REQUESTS_PER_MINUTE = 120
MAX_RETRIES = 5
MAX_SESSION_FILL_RATIO = 0.15

# %% [markdown] cell 15
## Phase 6: RL Decision Layer Components

# %% [markdown] cell 16
### 6.1 Trade Dataclass

Records all trading activity with timestamp, action, shares, price, and cost.

# %% [code] cell 17
@dataclass
class Trade:
    """
    Represents a single trade execution.
    
    Attributes:
        timestamp: Time when trade was executed
        action: Trade direction ('buy', 'sell', 'hold')
        shares: Number of shares traded
        price: Execution price
        cost: Transaction cost paid
    """
    timestamp: Union[int, pd.Timestamp, datetime]
    action: str  # 'buy', 'sell', 'hold'
    shares: float
    price: float
    cost: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            'timestamp': self.timestamp,
            'action': self.action,
            'shares': self.shares,
            'price': self.price,
            'cost': self.cost,
        }
    
    def __repr__(self) -> str:
        return f"Trade({self.action} {self.shares:.4f} shares @ ${self.price:.2f}, cost=${self.cost:.2f})"


# Test the Trade dataclass
test_trade = Trade(
    timestamp=pd.Timestamp('2024-01-01 09:30:00'),
    action='buy',
    shares=100.0,
    price=150.50,
    cost=15.05
)
print('Test Trade:', test_trade)
print('Trade dict:', test_trade.to_dict())

# %% [markdown] cell 18
### 6.2 Market Regime Enum

Defines market regimes for regime-aware trading.

# %% [code] cell 19
class MarketRegime(Enum):
    """Market regime classification."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRISIS = "crisis"


@dataclass
class RegimeConfig:
    """Configuration for regime thresholds and behavior."""
    # Turbulence thresholds (percentiles)
    normal_threshold: float = 0.75
    elevated_threshold: float = 0.90
    
    # Temperature adjustments
    normal_temp_mult: float = 1.0
    elevated_temp_mult: float = 1.3
    crisis_temp_mult: float = 1.8
    
    # Position sizing adjustments
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
    """
    
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self._history: list = []
        self._percentiles: Dict[float, float] = {}
    
    def update(self, returns: np.ndarray) -> float:
        """Calculate turbulence for current return vector."""
        self._history.append(returns)
        
        if len(self._history) < self.lookback:
            return 0.0
        
        self._history = self._history[-self.lookback:]
        
        history_array = np.array(self._history)
        mean = np.mean(history_array, axis=0)
        
        cov = np.cov(history_array.T)
        cov += np.eye(cov.shape[0]) * 1e-6
        
        diff = returns - mean
        try:
            inv_cov = np.linalg.inv(cov)
            turbulence = np.sqrt(diff @ inv_cov @ diff)
        except np.linalg.LinAlgError:
            turbulence = np.linalg.norm(diff)
        
        return float(turbulence)


class MarketRegimeDetector:
    """
    Detects market regime based on turbulence and volatility indicators.
    """
    
    def __init__(self, config: RegimeConfig = None):
        self.config = config or RegimeConfig()
        self.turbulence_calc = TurbulenceIndexCalculator(self.config.lookback)
        self._atr_history: list = []
        self._current_regime: MarketRegime = MarketRegime.NORMAL
    
    def detect_regime(self, returns: np.ndarray, atr: float) -> MarketRegime:
        """Detect current market regime."""
        turbulence = self.turbulence_calc.update(returns)
        
        self._atr_history.append(atr)
        self._atr_history = self._atr_history[-self.config.lookback:]
        
        if len(self._atr_history) < self.config.lookback:
            return MarketRegime.NORMAL
        
        atr_percentile = np.mean(np.array(self._atr_history) <= atr)
        
        turb_percentile = 0.5
        if self.turbulence_calc._percentiles:
            turb_percentile = np.mean([
                turbulence <= v for v in self.turbulence_calc._percentiles.values()
            ])
        
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
    
    def get_regime_indicator(self) -> float:
        """Get current regime as numeric indicator."""
        regime_map = {'normal': 0.0, 'elevated': 0.5, 'crisis': 1.0}
        return regime_map.get(self._current_regime.value, 0.0)

# %% [markdown] cell 20
### 6.3 Stock Trading Environment (Gym)

Custom Gym environment for RL training with:
- State: Model predictions + market state + portfolio state + regime
- Action: Continuous position sizing [-1, 1]
- Reward: Differential Sharpe Ratio

# %% [code] cell 21
class StockTradingEnv:
    """
    Stock trading environment for RL training.
    
    State: Model predictions + market state + portfolio state + regime
    Action: Continuous position sizing [-1, 1] (short to long)
    Reward: Differential Sharpe Ratio (DSR)
    
    Note: This is a custom environment compatible with RL training.
    It implements the core gym interface without external dependencies.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        max_position: float = 1.0,
        lookback: int = 60,
        use_turbulence: bool = True,
        dsr_eta: float = 0.1
    ):
        """
        Initialize trading environment.
        
        Args:
            df: DataFrame with OHLCV data and technical indicators
            predictions: Model predictions [time, horizon, 4]
            initial_balance: Starting cash balance
            transaction_cost: Transaction cost as fraction (0.001 = 0.1%)
            max_position: Maximum position size as fraction of portfolio
            lookback: Lookback window for state calculation
            use_turbulence: Whether to use regime detection
            dsr_eta: DSR decay rate for EMA calculations
        """
        self.df = df.reset_index(drop=True)
        self.predictions = predictions
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.lookback = lookback
        self.use_turbulence = use_turbulence
        self.dsr_eta = dsr_eta
        
        # State and action dimensions
        self.state_dim = self._calculate_state_dim()
        self.action_dim = 1
        
        # Action bounds
        self.action_low = -1.0
        self.action_high = 1.0
        
        # Portfolio state
        self.balance = initial_balance
        self.shares = 0.0
        self.portfolio_value = initial_balance
        self.trades: List[Trade] = []
        
        # For DSR calculation
        self.returns_history: List[float] = []
        self.A = 0.0  # EMA of returns
        self.B = 0.0  # EMA of squared returns
        
        # Turbulence detector
        self.regime_detector = None
        if use_turbulence:
            self.regime_detector = MarketRegimeDetector()
        
        self.current_step = 0
        self.max_steps = len(df) - 1
        
        # Track portfolio value history for metrics
        self.portfolio_history: List[float] = []
        self.reward_history: List[float] = []
    
    def _calculate_state_dim(self) -> int:
        """Calculate state vector dimension."""
        # Model predictions: horizon * 4 (OHLC)
        pred_dim = self.predictions.shape[1] * 4 if len(self.predictions.shape) >= 2 else 0
        
        # Market state: technical indicators
        market_dim = 10
        
        # Portfolio state
        portfolio_dim = 4
        
        # Regime
        regime_dim = 1 if self.use_turbulence else 0
        
        return pred_dim + market_dim + portfolio_dim + regime_dim
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.balance = self.initial_balance
        self.shares = 0.0
        self.portfolio_value = self.initial_balance
        self.trades = []
        self.returns_history = []
        self.A = 0.0
        self.B = 0.0
        self.current_step = self.lookback
        self.portfolio_history = [self.initial_balance]
        self.reward_history = []
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Construct state vector from predictions, market, portfolio, and regime."""
        # Model predictions (flattened)
        if len(self.predictions) > self.current_step:
            pred = self.predictions[self.current_step].flatten()
        else:
            pred = np.zeros(self.predictions.shape[1] * 4 if len(self.predictions.shape) >= 2 else 0)
        
        # Market state
        market = self._get_market_state()
        
        # Portfolio state (normalized)
        current_price = self.df['Close'].iloc[self.current_step] if self.current_step < len(self.df) else 0
        portfolio_value = self.balance + self.shares * current_price
        position_pct = (self.shares * current_price) / portfolio_value if portfolio_value > 0 else 0
        
        portfolio = np.array([
            self.balance / self.initial_balance,
            self.shares / 1000.0,
            portfolio_value / self.initial_balance,
            position_pct
        ])
        
        # Regime indicator
        if self.use_turbulence:
            regime = np.array([self._get_regime_indicator()])
            return np.concatenate([pred, market, portfolio, regime])
        
        return np.concatenate([pred, market, portfolio])
    
    def _get_market_state(self) -> np.ndarray:
        """Extract current market state features."""
        idx = self.current_step
        if idx >= len(self.df):
            return np.zeros(10)
        
        row = self.df.iloc[idx]
        
        # Get technical indicators with fallbacks
        rsi = row.get('rsi_14', 50) / 100.0 if 'rsi_14' in row else 0.5
        macd = row.get('macd_histogram', 0) if 'macd_histogram' in row else 0
        bb_pos = row.get('bb_position', 0.5) if 'bb_position' in row else 0.5
        atr_pct = row.get('atr_14_pct', 0.01) if 'atr_14_pct' in row else 0.01
        mom5 = row.get('price_momentum_5', 0) if 'price_momentum_5' in row else 0
        mom20 = row.get('price_momentum_20', 0) if 'price_momentum_20' in row else 0
        vwap_dev = row.get('vwap_20_dev', 0) if 'vwap_20_dev' in row else 0
        obv_slope = row.get('obv_slope', 0) / 1e6 if 'obv_slope' in row else 0
        direction = row.get('direction', 0) if 'direction' in row else 0
        
        # Volume ratio
        if 'Volume' in row and len(self.df) > 20:
            vol_ma = self.df['Volume'].iloc[max(0, idx-20):idx+1].mean()
            vol_ratio = row['Volume'] / vol_ma if vol_ma > 0 else 1.0
        else:
            vol_ratio = 1.0
        
        return np.array([
            rsi, macd, bb_pos, atr_pct, mom5,
            mom20, vwap_dev, obv_slope, direction, vol_ratio
        ])
    
    def _get_regime_indicator(self) -> float:
        """Get current regime as numeric indicator."""
        if self.regime_detector is None:
            return 0.0
        return self.regime_detector.get_regime_indicator()
    
    def step(self, action: Union[np.ndarray, float]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one trading step.
        
        Args:
            action: Position target [-1, 1] where -1 = max short, 1 = max long
        
        Returns:
            (observation, reward, done, info)
        """
        # Convert action to scalar
        if isinstance(action, np.ndarray):
            action_val = float(action[0]) if len(action) > 0 else float(action)
        else:
            action_val = float(action)
        
        # Clip action to valid range
        action_val = np.clip(action_val, self.action_low, self.action_high)
        
        # Current price
        current_price = self.df['Close'].iloc[self.current_step]
        
        # Apply regime-based position limiting
        if self.regime_detector:
            position_mult = self.regime_detector.get_position_multiplier()
            action_val *= position_mult
        
        # Target position value
        portfolio_value = self.balance + self.shares * current_price
        target_position_value = action_val * self.max_position * portfolio_value
        
        # Current position value
        current_position_value = self.shares * current_price
        
        # Trade needed
        trade_value = target_position_value - current_position_value
        
        # Execute trade
        if abs(trade_value) > 10:  # Min trade size $10
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
        self.portfolio_history.append(self.portfolio_value)
        
        # Calculate return
        if len(self.returns_history) > 0:
            prev_value = self.portfolio_history[-2] if len(self.portfolio_history) > 1 else self.initial_balance
            ret = (self.portfolio_value / prev_value) - 1 if prev_value > 0 else 0
        else:
            ret = 0.0
        
        self.returns_history.append(ret)
        
        # Calculate DSR reward
        reward = self._calculate_dsr_reward(ret)
        self.reward_history.append(reward)
        
        # Update regime detector
        if self.use_turbulence and self.regime_detector:
            returns = np.array([ret])
            atr = self.df.get('atr_14', pd.Series([0] * len(self.df))).iloc[self.current_step]
            self.regime_detector.detect_regime(returns, atr)
        
        # Advance
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        info = {
            'portfolio_value': self.portfolio_value,
            'return': ret,
            'shares': self.shares,
            'balance': self.balance,
            'position_pct': (self.shares * current_price) / self.portfolio_value if self.portfolio_value > 0 else 0,
            'regime': self.regime_detector._current_regime.value if self.regime_detector else 'normal'
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
        
        return float(dsr)
    
    def get_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        if len(self.portfolio_history) < 2:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'num_trades': len(self.trades)
            }
        
        # Total return
        total_return = (self.portfolio_history[-1] / self.initial_balance) - 1
        
        # Returns series
        returns = np.array(self.returns_history)
        returns = returns[returns != 0]  # Remove zeros
        
        # Sharpe ratio (assuming 252 trading days, scaling for intraday)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 390)  # 390 min/day
        else:
            sharpe = 0.0
        
        # Max drawdown
        portfolio = np.array(self.portfolio_history)
        peak = np.maximum.accumulate(portfolio)
        drawdown = (portfolio - peak) / peak
        max_drawdown = np.min(drawdown)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trades),
            'final_value': self.portfolio_history[-1]
        }

