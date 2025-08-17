"""
Swing Trade Executor V1 - Basic Implementation.

Longer-term execution with wider stops and larger profit targets for swing trading.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .. import BaseTradeExecutor, TradeExecution, RiskParameters
from ...setup_detection import SetupData
from ...quality_evaluation import QualityScore


class SwingExecutorV1(BaseTradeExecutor):
    """Basic swing trading execution engine."""
    
    def __init__(self):
        super().__init__("SwingExecutorV1", "1.0")
        # Swing trading risk parameters
        self.risk_params = RiskParameters(
            max_position_size=0.25,   # Larger positions for swing
            max_daily_risk=0.02,      # Standard daily risk
            max_concurrent_trades=5,
            stop_loss_percent=0.03,   # Wider stops
            take_profit_percent=0.09, # Larger targets (3:1 RR)
            trailing_stop=True,
            break_even_level=0.4      # Move to break-even at 40% to target
        )
    
    def execute_setup(self, setup: SetupData, quality: QualityScore, market_data: pd.DataFrame, parameters: Dict[str, Any]) -> Optional[TradeExecution]:
        """Execute swing trade with patient profit-taking."""
        # Lower quality threshold for swing trading (more time to work out)
        if quality.overall_score < parameters.get('min_quality_threshold', 0.5):
            return None
        
        # Check trade limits
        if len(self.open_trades) >= self.risk_params.max_concurrent_trades:
            return None
        
        if self.get_current_risk() >= self.risk_params.max_daily_risk:
            return None
        
        # Use setup's original stop/target if available, otherwise calculate
        entry_price = setup.entry_price
        
        if setup.stop_loss and setup.take_profit:
            # Use setup's levels
            stop_loss = setup.stop_loss
            take_profit = setup.take_profit
        else:
            # Calculate swing trading levels
            if setup.setup_type.endswith('_long'):
                stop_loss = entry_price * (1 - self.risk_params.stop_loss_percent)
                take_profit = entry_price * (1 + self.risk_params.take_profit_percent)
            else:
                stop_loss = entry_price * (1 + self.risk_params.stop_loss_percent)
                take_profit = entry_price * (1 - self.risk_params.take_profit_percent)
        
        # Calculate position size (larger for swing trades)
        position_size = self.calculate_position_size(setup, quality)
        
        # Swing trading multiplier (can be larger positions)
        swing_multiplier = parameters.get('swing_size_multiplier', 1.2)
        position_size *= swing_multiplier
        
        # Create trade execution
        execution = TradeExecution(
            setup_id=f"{setup.timestamp}_{setup.setup_type}",
            execution_timestamp=setup.timestamp,
            entry_price=entry_price,
            exit_price=None,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            execution_type=parameters.get('execution_type', 'limit'),  # Prefer limit orders
            status='open',
            pnl=None,
            fees=position_size * entry_price * 0.001,
            metadata={
                'execution_engine': self.name,
                'version': self.version,
                'quality_score': quality.overall_score,
                'original_setup_confidence': setup.confidence,
                'swing_style': True,
                'patient_management': True,
                'original_rr': setup.risk_reward_ratio
            }
        )
        
        self.open_trades.append(execution)
        return execution
    
    def manage_open_trades(self, market_data: pd.DataFrame, current_time: datetime) -> List[TradeExecution]:
        """Manage swing trades with patient profit-taking."""
        closed_trades = []
        current_price = market_data.loc[current_time, 'close'] if current_time in market_data.index else None
        
        if current_price is None:
            return closed_trades
        
        for trade in self.open_trades[:]:
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
            
            # Move stop to break-even when profitable
            profit_percent = self._calculate_profit_percent(trade, current_price)
            if profit_percent >= self.risk_params.break_even_level and not trade.metadata.get('break_even_set'):
                trade.stop_loss = trade.entry_price
                trade.metadata['break_even_set'] = True
                trade.metadata['break_even_time'] = current_time
            
            # Partial profit-taking for swing trades
            if profit_percent >= 0.7 and not trade.metadata.get('partial_profit_taken'):
                # Take 30% profits, let 70% run
                trade.position_size *= 0.7
                trade.metadata['partial_profit_taken'] = True
                trade.metadata['partial_profit_time'] = current_time
            
            # Time-based management (much longer for swing trades)
            time_held = current_time - trade.execution_timestamp
            max_hold_time = timedelta(days=parameters.get('max_hold_days', 14))
            
            # Warning at 80% of max time
            if time_held > max_hold_time * 0.8:
                trade.metadata['time_warning'] = True
            
            # Force exit at max time
            if time_held > max_hold_time:
                closed_trade = self._close_trade(trade, current_price, current_time, 'time_exit')
                closed_trades.append(closed_trade)
                continue
            
            # Advanced trailing stop for swing trades
            if self.risk_params.trailing_stop and profit_percent > 0.2:
                self._update_swing_trailing_stop(trade, current_price, market_data, current_time)
        
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
    
    def _update_swing_trailing_stop(self, trade: TradeExecution, current_price: float, market_data: pd.DataFrame, current_time: datetime):
        """Update trailing stop for swing trades using ATR-based approach."""
        try:
            # Calculate ATR for dynamic trailing distance
            trade_idx = market_data.index.get_loc(current_time, method='nearest')
            recent_data = market_data.iloc[max(0, trade_idx-14):trade_idx+1]
            
            if len(recent_data) > 1:
                # Simple ATR calculation
                high_low = recent_data['high'] - recent_data['low']
                high_close = abs(recent_data['high'] - recent_data['close'].shift(1))
                low_close = abs(recent_data['low'] - recent_data['close'].shift(1))
                
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.mean()
                
                # Use 2x ATR for trailing distance
                trailing_distance = 2 * atr
            else:
                # Fallback to percentage-based
                trailing_distance = current_price * 0.02  # 2%
            
            if 'long' in trade.setup_id:
                new_stop = current_price - trailing_distance
                trade.stop_loss = max(trade.stop_loss, new_stop)
            else:
                new_stop = current_price + trailing_distance
                trade.stop_loss = min(trade.stop_loss, new_stop)
                
        except Exception:
            # Fallback to simple percentage trailing
            if 'long' in trade.setup_id:
                new_stop = current_price * 0.98  # 2% trail
                trade.stop_loss = max(trade.stop_loss, new_stop)
            else:
                new_stop = current_price * 1.02
                trade.stop_loss = min(trade.stop_loss, new_stop)
    
    def _close_trade(self, trade: TradeExecution, exit_price: float, exit_time: datetime, exit_reason: str) -> TradeExecution:
        """Close a swing trade and calculate PnL."""
        trade.exit_price = exit_price
        trade.status = 'closed'
        trade.metadata['exit_time'] = exit_time
        trade.metadata['exit_reason'] = exit_reason
        
        # Calculate PnL
        if 'long' in trade.setup_id:
            pnl = (exit_price - trade.entry_price) * trade.position_size
        else:
            pnl = (trade.entry_price - exit_price) * trade.position_size
        
        # Subtract fees
        pnl -= trade.fees
        pnl -= trade.position_size * exit_price * 0.001
        
        trade.pnl = pnl
        self.update_portfolio_value(pnl)
        
        # Remove from open trades
        if trade in self.open_trades:
            self.open_trades.remove(trade)
        
        return trade
    
    def get_parameter_ranges(self) -> Dict[str, tuple]:
        """Get parameter ranges for optimization."""
        return {
            'min_quality_threshold': (0.4, 0.8),
            'swing_size_multiplier': (0.8, 1.5),
            'max_hold_days': (7, 21),
            'execution_type': ('limit', 'market'),
            'trailing_atr_multiplier': (1.5, 3.0),
            'partial_profit_threshold': (0.6, 0.8)
        }