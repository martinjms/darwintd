"""
VectorBT-powered backtesting engine for DarwinTD evolutionary trading system.

This module provides high-performance backtesting capabilities optimized for 
genetic algorithm parameter optimization and strategy evolution.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings

# Suppress VectorBT warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='vectorbt')


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    initial_cash: float = 100000.0
    commission: float = 0.001  # 0.1% commission
    slippage: float = 0.0005   # 0.05% slippage
    max_position_size: float = 0.25  # 25% of portfolio per position
    leverage: float = 1.0
    freq: str = '1H'  # Default frequency
    
    # Risk management
    max_drawdown_limit: float = 0.2  # 20% max drawdown
    max_consecutive_losses: int = 5
    position_timeout: int = 100  # Maximum bars to hold position


@dataclass
class SignalData:
    """Container for trading signals and associated data."""
    entries: np.ndarray  # Boolean array for entry signals
    exits: np.ndarray    # Boolean array for exit signals
    prices: pd.Series    # Price data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate signal data consistency."""
        if len(self.entries) != len(self.exits) != len(self.prices):
            raise ValueError("Entries, exits, and prices must have same length")
        
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            raise ValueError("Price data must have DatetimeIndex")


@dataclass
class PortfolioResults:
    """Results from portfolio backtesting."""
    portfolio: vbt.Portfolio
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    
    # Advanced metrics
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    tail_ratio: float = 0.0
    value_at_risk: float = 0.0
    
    @classmethod
    def from_portfolio(cls, portfolio: vbt.Portfolio) -> 'PortfolioResults':
        """Create PortfolioResults from VectorBT Portfolio object."""
        try:
            # Basic metrics
            total_return = portfolio.total_return()
            sharpe_ratio = portfolio.sharpe_ratio()
            max_drawdown = portfolio.max_drawdown()
            
            # Trade statistics
            trades = portfolio.trades
            total_trades = trades.count()
            win_rate = trades.win_rate() if total_trades > 0 else 0.0
            profit_factor = trades.profit_factor() if total_trades > 0 else 0.0
            
            # Average trade duration
            if total_trades > 0:
                durations = trades.duration
                avg_trade_duration = durations.mean() if len(durations) > 0 else 0.0
            else:
                avg_trade_duration = 0.0
            
            # Advanced metrics
            returns = portfolio.returns()
            calmar_ratio = total_return / max_drawdown if max_drawdown != 0 else 0.0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                sortino_ratio = returns.mean() / downside_std if downside_std != 0 else 0.0
            else:
                sortino_ratio = float('inf') if returns.mean() > 0 else 0.0
            
            # Tail ratio (95th percentile / 5th percentile)
            if len(returns) > 20:  # Need sufficient data
                tail_ratio = np.percentile(returns, 95) / abs(np.percentile(returns, 5))
                tail_ratio = tail_ratio if not np.isnan(tail_ratio) and np.isfinite(tail_ratio) else 0.0
            else:
                tail_ratio = 0.0
            
            # Value at Risk (5% VaR)
            value_at_risk = np.percentile(returns, 5) if len(returns) > 0 else 0.0
            
            return cls(
                portfolio=portfolio,
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=total_trades,
                avg_trade_duration=avg_trade_duration,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                tail_ratio=tail_ratio,
                value_at_risk=value_at_risk
            )
            
        except Exception as e:
            # Fallback for any calculation errors
            return cls(
                portfolio=portfolio,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=1.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                avg_trade_duration=0.0
            )


class VectorBTEngine:
    """
    High-performance backtesting engine using VectorBT.
    
    Optimized for genetic algorithm parameter optimization with support for
    vectorized operations across multiple parameter combinations.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """Initialize the VectorBT backtesting engine."""
        self.config = config or BacktestConfig()
        self._last_portfolio = None
        
    def run_backtest(self, signal_data: SignalData) -> PortfolioResults:
        """
        Run a single backtest with the given signals.
        
        Args:
            signal_data: SignalData containing entries, exits, and prices
            
        Returns:
            PortfolioResults with comprehensive performance metrics
        """
        try:
            # Create VectorBT portfolio
            portfolio = vbt.Portfolio.from_signals(
                close=signal_data.prices,
                entries=signal_data.entries,
                exits=signal_data.exits,
                init_cash=self.config.initial_cash,
                fees=self.config.commission,
                slippage=self.config.slippage,
                freq=self.config.freq
            )
            
            self._last_portfolio = portfolio
            return PortfolioResults.from_portfolio(portfolio)
            
        except Exception as e:
            # Return failed backtest results
            print(f"Backtest failed: {e}")
            return PortfolioResults(
                portfolio=None,
                total_return=-1.0,
                sharpe_ratio=-10.0,
                max_drawdown=1.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                avg_trade_duration=0.0
            )
    
    def run_parameter_optimization(
        self, 
        signal_generator,
        data: pd.DataFrame,
        parameter_space: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Run optimization across a parameter space using VectorBT's vectorized operations.
        
        Args:
            signal_generator: Function that generates signals from data and parameters
            data: OHLCV price data
            parameter_space: Dictionary of parameter arrays to optimize over
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations(parameter_space)
            
            # Generate signals for all parameter combinations
            all_entries = []
            all_exits = []
            
            for params in param_combinations:
                entries, exits = signal_generator(data, params)
                all_entries.append(entries)
                all_exits.append(exits)
            
            # Stack signals into matrices for vectorized backtesting
            entries_matrix = np.column_stack(all_entries)
            exits_matrix = np.column_stack(all_exits)
            
            # Run vectorized backtest (handle case where we have multiple columns)
            if entries_matrix.ndim == 1:
                # Single parameter set
                portfolios = vbt.Portfolio.from_signals(
                    close=data['close'],
                    entries=entries_matrix,
                    exits=exits_matrix,
                    init_cash=self.config.initial_cash,
                    fees=self.config.commission,
                    slippage=self.config.slippage,
                    freq=self.config.freq
                )
                results = [PortfolioResults.from_portfolio(portfolios)]
                results[0].parameters = param_combinations[0]
            else:
                # Multiple parameter sets - run individually for stability
                results = []
                for i, params in enumerate(param_combinations):
                    try:
                        portfolio = vbt.Portfolio.from_signals(
                            close=data['close'],
                            entries=entries_matrix[:, i],
                            exits=exits_matrix[:, i],
                            init_cash=self.config.initial_cash,
                            fees=self.config.commission,
                            slippage=self.config.slippage,
                            freq=self.config.freq
                        )
                        result = PortfolioResults.from_portfolio(portfolio)
                        result.parameters = params
                        results.append(result)
                    except Exception as e:
                        # Create failed result for this parameter set
                        failed_result = PortfolioResults(
                            portfolio=None,
                            total_return=-1.0,
                            sharpe_ratio=-10.0,
                            max_drawdown=1.0,
                            win_rate=0.0,
                            profit_factor=0.0,
                            total_trades=0,
                            avg_trade_duration=0.0
                        )
                        failed_result.parameters = params
                        results.append(failed_result)
            
            # Find best parameters based on Sharpe ratio
            best_idx = np.argmax([r.sharpe_ratio for r in results])
            best_result = results[best_idx]
            
            return {
                'best_parameters': best_result.parameters,
                'best_result': best_result,
                'all_results': results,
                'parameter_combinations': param_combinations
            }
            
        except Exception as e:
            print(f"Parameter optimization failed: {e}")
            return {
                'best_parameters': {},
                'best_result': None,
                'all_results': [],
                'parameter_combinations': []
            }
    
    def _generate_parameter_combinations(self, parameter_space: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters for optimization."""
        import itertools
        
        param_names = list(parameter_space.keys())
        param_values = list(parameter_space.values())
        
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
        
        return combinations
    
    def calculate_fitness_metrics(self, results: PortfolioResults) -> Dict[str, float]:
        """
        Calculate fitness metrics for genetic algorithm optimization.
        
        Args:
            results: PortfolioResults from backtesting
            
        Returns:
            Dictionary of fitness metrics
        """
        # Multi-objective fitness considering return, risk, and stability
        return_score = max(0, results.total_return)  # Positive returns preferred
        risk_score = max(0, 1 - results.max_drawdown)  # Lower drawdown preferred
        trade_score = min(1, results.win_rate)  # Higher win rate preferred
        
        # Composite fitness score
        fitness = (return_score * 0.4 + risk_score * 0.3 + trade_score * 0.3)
        
        return {
            'fitness': fitness,
            'return_score': return_score,
            'risk_score': risk_score,
            'trade_score': trade_score,
            'sharpe_ratio': results.sharpe_ratio,
            'calmar_ratio': results.calmar_ratio,
            'sortino_ratio': results.sortino_ratio
        }
    
    def validate_signals(self, signal_data: SignalData) -> bool:
        """
        Validate trading signals for consistency and correctness.
        
        Args:
            signal_data: SignalData to validate
            
        Returns:
            True if signals are valid, False otherwise
        """
        try:
            # Check for basic consistency
            if len(signal_data.entries) != len(signal_data.exits):
                return False
            
            # Check for simultaneous entry and exit signals
            simultaneous = signal_data.entries & signal_data.exits
            if simultaneous.any():
                print(f"Warning: {simultaneous.sum()} simultaneous entry/exit signals found")
            
            # Check for reasonable signal frequency (not too frequent)
            entry_frequency = signal_data.entries.sum() / len(signal_data.entries)
            if entry_frequency > 0.1:  # More than 10% of bars
                print(f"Warning: High signal frequency ({entry_frequency:.1%})")
            
            # Check for any signals at all
            if not signal_data.entries.any():
                print("Warning: No entry signals found")
                return False
            
            return True
            
        except Exception as e:
            print(f"Signal validation failed: {e}")
            return False
    
    def get_last_portfolio(self) -> Optional[vbt.Portfolio]:
        """Get the last portfolio object for detailed analysis."""
        return self._last_portfolio


# Utility functions for common backtesting operations

def create_simple_signals(prices: pd.Series, buy_threshold: float = 0.02, sell_threshold: float = -0.02) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create simple momentum-based entry/exit signals for testing.
    
    Args:
        prices: Price series
        buy_threshold: Percentage change threshold for buy signals
        sell_threshold: Percentage change threshold for sell signals
        
    Returns:
        Tuple of (entries, exits) as numpy arrays
    """
    returns = prices.pct_change()
    
    entries = returns > buy_threshold
    exits = returns < sell_threshold
    
    return entries.values, exits.values


def benchmark_backtest_performance(engine: VectorBTEngine, data: pd.DataFrame, num_runs: int = 100) -> Dict[str, float]:
    """
    Benchmark backtesting performance for optimization planning.
    
    Args:
        engine: VectorBTEngine instance
        data: OHLCV data for testing
        num_runs: Number of backtests to run for timing
        
    Returns:
        Performance metrics dictionary
    """
    import time
    
    # Generate test signals
    entries, exits = create_simple_signals(data['close'])
    signal_data = SignalData(entries=entries, exits=exits, prices=data['close'])
    
    # Time multiple runs
    start_time = time.time()
    
    for _ in range(num_runs):
        results = engine.run_backtest(signal_data)
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    
    return {
        'avg_backtest_time_ms': avg_time * 1000,
        'backtests_per_second': 1 / avg_time,
        'estimated_optimization_time_minutes': (avg_time * 1000) / 60  # For 1000 parameter combinations
    }