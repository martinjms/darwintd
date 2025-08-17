"""
Trade Execution Module for DarwinTD.

This module contains engines for executing trades based on validated setups.
Trade execution is separate from setup detection and quality evaluation to allow
independent optimization of entry/exit timing, position sizing, and risk management.

Available Execution Engines:
- Scalping: Quick entries with tight stops and fast exits
- Swing: Longer-term execution with wider stops and profit targets
- Adaptive: Adjusts execution style based on market conditions
- Risk_Managed: Focus on capital preservation and risk management

Each engine takes validated SetupData and QualityScore to execute trades.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np

from ..setup_detection import SetupData
from ..quality_evaluation import QualityScore

@dataclass
class TradeExecution:
    """Trade execution result data."""
    setup_id: str
    execution_timestamp: datetime
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    stop_loss: float
    take_profit: float
    execution_type: str  # 'market', 'limit', 'stop'
    status: str  # 'open', 'closed', 'cancelled'
    pnl: Optional[float]
    fees: float
    metadata: Dict[str, Any]

@dataclass
class RiskParameters:
    """Risk management parameters for trade execution."""
    max_position_size: float = 0.25  # 25% of portfolio
    max_daily_risk: float = 0.02  # 2% of portfolio per day
    max_concurrent_trades: int = 5
    stop_loss_percent: float = 0.02  # 2% stop loss
    take_profit_percent: float = 0.06  # 6% take profit (3:1 RR)
    trailing_stop: bool = False
    break_even_level: float = 0.5  # Move stop to break-even at 50% to target


class BaseTradeExecutor(ABC):
    """Abstract base class for all trade execution engines."""
    
    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self.risk_params = RiskParameters()
        self.open_trades = []
        self.portfolio_value = 100000.0  # Starting portfolio value
    
    @abstractmethod
    def execute_setup(self, setup: SetupData, quality: QualityScore, market_data: pd.DataFrame, parameters: Dict[str, Any]) -> Optional[TradeExecution]:
        """Execute a trade based on setup and quality evaluation."""
        pass
    
    @abstractmethod
    def manage_open_trades(self, market_data: pd.DataFrame, current_time: datetime) -> List[TradeExecution]:
        """Manage open trades (exits, stop adjustments, etc.)."""
        pass
    
    @abstractmethod
    def get_parameter_ranges(self) -> Dict[str, tuple]:
        """Get parameter ranges for optimization."""
        pass
    
    def calculate_position_size(self, setup: SetupData, quality: QualityScore) -> float:
        """Calculate position size based on risk and quality."""
        # Base position size on quality score
        base_size = self.risk_params.max_position_size * quality.overall_score
        
        # Adjust for risk/reward ratio
        if setup.risk_reward_ratio and setup.risk_reward_ratio > 0:
            rr_multiplier = min(1.5, setup.risk_reward_ratio / 2)  # Cap at 1.5x
            base_size *= rr_multiplier
        
        # Ensure within risk limits
        max_risk_size = self.risk_params.max_daily_risk / (abs(setup.entry_price - setup.stop_loss) / setup.entry_price)
        
        return min(base_size, max_risk_size)
    
    def update_portfolio_value(self, pnl: float):
        """Update portfolio value after trade close."""
        self.portfolio_value += pnl
    
    def get_current_risk(self) -> float:
        """Calculate current portfolio risk from open trades."""
        total_risk = 0
        for trade in self.open_trades:
            if trade.status == 'open':
                risk_amount = abs(trade.entry_price - trade.stop_loss) * trade.position_size
                total_risk += risk_amount / self.portfolio_value
        return total_risk