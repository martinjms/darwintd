"""
Scalping Trade Executor V1 - Basic Implementation.

Fast execution with tight stops and quick profit-taking for scalping setups.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .. import BaseTradeExecutor, TradeExecution, RiskParameters
from ...setup_detection import SetupData
from ...quality_evaluation import QualityScore


class ScalpingExecutorV1(BaseTradeExecutor):
    """Basic scalping execution engine with tight risk management."""
    
    def __init__(self):
        super().__init__("ScalpingExecutorV1", "1.0")
        # Scalping-specific risk parameters
        self.risk_params = RiskParameters(
            max_position_size=0.1,  # Smaller positions for scalping
            max_daily_risk=0.015,   # Lower daily risk
            max_concurrent_trades=3,
            stop_loss_percent=0.01,   # Tight stops
            take_profit_percent=0.02, # Quick profits (2:1 RR)
            trailing_stop=True,
            break_even_level=0.3      # Move to break-even quickly
        )
    
    def execute_setup(self, setup: SetupData, quality: QualityScore, market_data: pd.DataFrame, parameters: Dict[str, Any]) -> Optional[TradeExecution]:
        """Execute scalping trade with tight risk management."""
        # Only execute high-quality setups for scalping
        if quality.overall_score < parameters.get('min_quality_threshold', 0.7):
            return None
        
        # Check if we have room for more trades
        if len(self.open_trades) >= self.risk_params.max_concurrent_trades:
            return None
        
        # Check daily risk limits
        if self.get_current_risk() >= self.risk_params.max_daily_risk:
            return None
        
        # Calculate scalping-specific entry and exits
        entry_price = setup.entry_price
        
        # Tight stop loss for scalping
        if setup.setup_type.endswith('_long'):
            stop_loss = entry_price * (1 - self.risk_params.stop_loss_percent)
            take_profit = entry_price * (1 + self.risk_params.take_profit_percent)
        else:
            stop_loss = entry_price * (1 + self.risk_params.stop_loss_percent)
            take_profit = entry_price * (1 - self.risk_params.take_profit_percent)
        
        # Calculate position size
        position_size = self.calculate_position_size(setup, quality)
        
        # Adjust for scalping (smaller positions)
        scalping_multiplier = parameters.get('scalping_size_multiplier', 0.5)
        position_size *= scalping_multiplier
        
        # Create trade execution
        execution = TradeExecution(
            setup_id=f"{setup.timestamp}_{setup.setup_type}",
            execution_timestamp=setup.timestamp,
            entry_price=entry_price,
            exit_price=None,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            execution_type='market',  # Fast market orders for scalping
            status='open',
            pnl=None,
            fees=position_size * entry_price * 0.001,  # 0.1% fees
            metadata={
                'execution_engine': self.name,
                'version': self.version,
                'quality_score': quality.overall_score,
                'original_setup_confidence': setup.confidence,
                'scalping_style': True,
                'tight_management': True
            }
        )
        
        self.open_trades.append(execution)
        return execution
    
    def manage_open_trades(self, market_data: pd.DataFrame, current_time: datetime) -> List[TradeExecution]:
        """Manage open scalping trades with aggressive profit-taking."""
        closed_trades = []
        current_price = market_data.loc[current_time, 'close'] if current_time in market_data.index else None
        
        if current_price is None:
            return closed_trades
        
        for trade in self.open_trades[:]:  # Copy list to avoid modification issues
            if trade.status != 'open':
                continue
            
            # Check for stop loss
            if self._check_stop_loss(trade, current_price):
                closed_trade = self._close_trade(trade, current_price, current_time, 'stop_loss')
                closed_trades.append(closed_trade)
                continue
            
            # Check for take profit
            if self._check_take_profit(trade, current_price):
                closed_trade = self._close_trade(trade, current_price, current_time, 'take_profit')
                closed_trades.append(closed_trade)
                continue
            
            # Scalping-specific: Aggressive profit-taking
            profit_percent = self._calculate_profit_percent(trade, current_price)
            if profit_percent >= 0.5:  # Take partial profits at 50% of target
                # Reduce position size and move stop to break-even
                trade.position_size *= 0.5
                trade.stop_loss = trade.entry_price
                trade.metadata['partial_profit_taken'] = True
            
            # Time-based exit for scalping (don't hold too long)
            time_held = current_time - trade.execution_timestamp
            max_hold_time = timedelta(hours=2)  # Max 2 hours for scalping
            
            if time_held > max_hold_time:
                closed_trade = self._close_trade(trade, current_price, current_time, 'time_exit')
                closed_trades.append(closed_trade)
                continue
            
            # Trailing stop for profitable trades
            if self.risk_params.trailing_stop and profit_percent > 0:
                self._update_trailing_stop(trade, current_price)
        
        return closed_trades
    
    def _check_stop_loss(self, trade: TradeExecution, current_price: float) -> bool:
        """Check if stop loss should be triggered."""
        if 'long' in trade.setup_id:
            return current_price <= trade.stop_loss
        else:
            return current_price >= trade.stop_loss
    
    def _check_take_profit(self, trade: TradeExecution, current_price: float) -> bool:
        """Check if take profit should be triggered."""
        if 'long' in trade.setup_id:
            return current_price >= trade.take_profit
        else:
            return current_price <= trade.take_profit
    
    def _calculate_profit_percent(self, trade: TradeExecution, current_price: float) -> float:
        """Calculate current profit as percentage of target."""
        if 'long' in trade.setup_id:
            current_profit = current_price - trade.entry_price
            target_profit = trade.take_profit - trade.entry_price
        else:
            current_profit = trade.entry_price - current_price
            target_profit = trade.entry_price - trade.take_profit
        
        return current_profit / target_profit if target_profit > 0 else 0
    
    def _update_trailing_stop(self, trade: TradeExecution, current_price: float):
        """Update trailing stop for profitable positions."""
        if 'long' in trade.setup_id:
            # For long positions, trail stop up
            new_stop = current_price * 0.995  # 0.5% trailing distance
            trade.stop_loss = max(trade.stop_loss, new_stop)
        else:
            # For short positions, trail stop down
            new_stop = current_price * 1.005  # 0.5% trailing distance
            trade.stop_loss = min(trade.stop_loss, new_stop)
    
    def _close_trade(self, trade: TradeExecution, exit_price: float, exit_time: datetime, exit_reason: str) -> TradeExecution:
        """Close a trade and calculate PnL."""
        trade.exit_price = exit_price
        trade.status = 'closed'
        trade.metadata['exit_time'] = exit_time
        trade.metadata['exit_reason'] = exit_reason
        
        # Calculate PnL
        if 'long' in trade.setup_id:
            pnl = (exit_price - trade.entry_price) * trade.position_size
        else:
            pnl = (trade.entry_price - exit_price) * trade.position_size
        
        pnl -= trade.fees  # Subtract entry fees
        pnl -= trade.position_size * exit_price * 0.001  # Exit fees
        
        trade.pnl = pnl
        self.update_portfolio_value(pnl)
        
        # Remove from open trades
        if trade in self.open_trades:
            self.open_trades.remove(trade)
        
        return trade
    
    def get_parameter_ranges(self) -> Dict[str, tuple]:
        """Get parameter ranges for optimization."""
        return {
            'min_quality_threshold': (0.6, 0.9),
            'scalping_size_multiplier': (0.3, 0.8),
            'max_hold_hours': (1, 4),
            'trailing_distance': (0.003, 0.01),
            'partial_profit_threshold': (0.3, 0.7)
        }