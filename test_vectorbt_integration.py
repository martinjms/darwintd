#!/usr/bin/env python3
"""
VectorBT Integration Test for DarwinTD.

Tests the core VectorBT backtesting engine and signal generation framework.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, '/app/src')

from darwintd.backtesting.vectorbt_engine import VectorBTEngine, BacktestConfig, SignalData
from darwintd.signals.base import BacktestData, PriceActionSignalGenerator


def create_test_data(days: int = 365) -> pd.DataFrame:
    """Create realistic test OHLCV data."""
    np.random.seed(42)  # For reproducible results
    
    # Generate date range
    dates = pd.date_range(start='2023-01-01', periods=days * 24, freq='h')
    
    # Generate realistic price movement (geometric brownian motion)
    base_price = 50000  # Starting price like BTC
    returns = np.random.normal(0, 0.005, len(dates))  # 0.5% hourly volatility
    
    # Add some trend and mean reversion
    trend = np.linspace(0, 0.2, len(dates))  # 20% upward trend over period
    returns += trend / len(dates)
    
    # Calculate cumulative prices
    log_prices = np.log(base_price) + np.cumsum(returns)
    prices = np.exp(log_prices)
    
    # Generate OHLC from prices with realistic spreads
    close_prices = prices
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    # Generate highs and lows with some noise
    high_noise = np.random.uniform(1.0, 1.02, len(dates))
    low_noise = np.random.uniform(0.98, 1.0, len(dates))
    
    high_prices = np.maximum(open_prices, close_prices) * high_noise
    low_prices = np.minimum(open_prices, close_prices) * low_noise
    
    # Generate volume (correlated with price volatility)
    volume_base = 1000
    volatility = np.abs(returns)
    volumes = volume_base * (1 + volatility * 10) * np.random.lognormal(0, 0.5, len(dates))
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    return data


class TestSignalGenerator(PriceActionSignalGenerator):
    """Simple test signal generator for momentum trading."""
    
    def __init__(self):
        super().__init__("TestMomentumStrategy")
    
    def generate_signals(self, data: BacktestData, parameters: dict):
        """Generate simple momentum signals."""
        # Get parameters
        momentum_window = parameters.get('momentum_window', 20)
        entry_threshold = parameters.get('entry_threshold', 0.02)
        exit_threshold = parameters.get('exit_threshold', -0.01)
        
        # Calculate momentum
        returns = data.close.pct_change(momentum_window)
        
        # Generate signals
        entries = returns > entry_threshold
        exits = returns < exit_threshold
        
        return entries.values, exits.values
    
    def get_parameter_ranges(self):
        """Define parameter ranges for optimization."""
        return {
            'momentum_window': (5, 50),
            'entry_threshold': (0.01, 0.05),
            'exit_threshold': (-0.05, -0.005)
        }


def test_basic_backtesting():
    """Test basic VectorBT backtesting functionality."""
    print("=== Testing Basic Backtesting ===")
    
    # Create test data
    ohlcv_data = create_test_data(days=100)
    print(f"Created test data: {len(ohlcv_data)} rows")
    
    # Create simple buy-and-hold signals for testing
    entries = np.zeros(len(ohlcv_data), dtype=bool)
    exits = np.zeros(len(ohlcv_data), dtype=bool)
    entries[10] = True  # Buy on day 10
    exits[-10] = True   # Sell 10 days before end
    
    signal_data = SignalData(
        entries=entries,
        exits=exits,
        prices=ohlcv_data['close']
    )
    
    # Run backtest
    engine = VectorBTEngine()
    results = engine.run_backtest(signal_data)
    
    print(f"✅ Backtest completed successfully")
    print(f"   Total Return: {results.total_return:.3f}")
    print(f"   Sharpe Ratio: {results.sharpe_ratio:.3f}")
    print(f"   Max Drawdown: {results.max_drawdown:.3f}")
    print(f"   Win Rate: {results.win_rate:.3f}")
    print(f"   Total Trades: {results.total_trades}")
    
    return True


def test_signal_generation():
    """Test price action signal generation."""
    print("\n=== Testing Signal Generation ===")
    
    # Create test data
    ohlcv_data = create_test_data(days=50)
    
    # Create BacktestData
    backtest_data = BacktestData(
        ohlcv=ohlcv_data,
        features=pd.DataFrame(index=ohlcv_data.index),  # Empty features for now
        metadata={'symbol': 'TEST', 'timeframe': '1H'}
    )
    
    # Create signal generator
    signal_gen = TestSignalGenerator()
    
    # Generate random parameters
    params = signal_gen.generate_random_parameters(1)[0]
    print(f"Test parameters: {params}")
    
    # Generate signals
    entries, exits = signal_gen.generate_signals(backtest_data, params)
    
    print(f"✅ Signals generated successfully")
    print(f"   Entry signals: {np.sum(entries)}")
    print(f"   Exit signals: {np.sum(exits)}")
    print(f"   Signal frequency: {np.sum(entries) / len(entries):.3%}")
    
    # Test signal quality metrics
    quality = signal_gen.calculate_signal_quality(backtest_data, (entries, exits))
    print(f"   Signal quality metrics: {quality}")
    
    return True


def test_parameter_optimization():
    """Test parameter optimization functionality."""
    print("\n=== Testing Parameter Optimization ===")
    
    # Create test data
    ohlcv_data = create_test_data(days=30)  # Smaller dataset for speed
    
    # Create BacktestData
    backtest_data = BacktestData(
        ohlcv=ohlcv_data,
        features=pd.DataFrame(index=ohlcv_data.index),
        metadata={'symbol': 'TEST', 'timeframe': '1H'}
    )
    
    # Create signal generator
    signal_gen = TestSignalGenerator()
    
    # Define parameter space for optimization
    parameter_space = {
        'momentum_window': np.array([10, 20, 30]),
        'entry_threshold': np.array([0.015, 0.025, 0.035]),
        'exit_threshold': np.array([-0.02, -0.015, -0.01])
    }
    
    # Create signal generator function for optimization
    def signal_generator_func(data, params):
        return signal_gen.generate_signals(backtest_data, params)
    
    # Run optimization
    engine = VectorBTEngine()
    optimization_results = engine.run_parameter_optimization(
        signal_generator=signal_generator_func,
        data=ohlcv_data,
        parameter_space=parameter_space
    )
    
    if optimization_results['best_result']:
        best_params = optimization_results['best_parameters']
        best_result = optimization_results['best_result']
        
        print(f"✅ Parameter optimization completed")
        print(f"   Best parameters: {best_params}")
        print(f"   Best Sharpe ratio: {best_result.sharpe_ratio:.3f}")
        print(f"   Best total return: {best_result.total_return:.3f}")
        print(f"   Total parameter combinations tested: {len(optimization_results['all_results'])}")
    else:
        print("❌ Parameter optimization failed")
        return False
    
    return True


def test_performance_benchmarking():
    """Test backtesting performance for genetic algorithm planning."""
    print("\n=== Testing Performance Benchmarking ===")
    
    # Create test data
    ohlcv_data = create_test_data(days=10)  # Small dataset for speed testing
    
    # Import the benchmark function
    from darwintd.backtesting.vectorbt_engine import benchmark_backtest_performance
    
    # Benchmark performance
    engine = VectorBTEngine()
    benchmark_results = benchmark_backtest_performance(engine, ohlcv_data, num_runs=10)
    
    print(f"✅ Performance benchmarking completed")
    print(f"   Average backtest time: {benchmark_results['avg_backtest_time_ms']:.2f} ms")
    print(f"   Backtests per second: {benchmark_results['backtests_per_second']:.1f}")
    print(f"   Estimated time for 1000 parameter optimization: {benchmark_results['estimated_optimization_time_minutes']:.1f} minutes")
    
    return True


def test_fitness_calculation():
    """Test fitness metric calculation for genetic algorithms."""
    print("\n=== Testing Fitness Calculation ===")
    
    # Create test data and run a backtest
    ohlcv_data = create_test_data(days=30)
    
    # Create simple test signals
    entries = np.zeros(len(ohlcv_data), dtype=bool)
    exits = np.zeros(len(ohlcv_data), dtype=bool)
    
    # Create multiple entries/exits for more realistic testing
    for i in range(5, len(ohlcv_data) - 30, 50):  # Leave more buffer at the end
        entries[i] = True
        if i + 20 < len(ohlcv_data):
            exits[i + 20] = True
    
    signal_data = SignalData(
        entries=entries,
        exits=exits,
        prices=ohlcv_data['close']
    )
    
    # Run backtest
    engine = VectorBTEngine()
    results = engine.run_backtest(signal_data)
    
    # Calculate fitness metrics
    fitness_metrics = engine.calculate_fitness_metrics(results)
    
    print(f"✅ Fitness calculation completed")
    print(f"   Composite fitness: {fitness_metrics['fitness']:.3f}")
    print(f"   Return score: {fitness_metrics['return_score']:.3f}")
    print(f"   Risk score: {fitness_metrics['risk_score']:.3f}")
    print(f"   Trade score: {fitness_metrics['trade_score']:.3f}")
    
    return True


def main():
    """Run all integration tests."""
    print("DarwinTD VectorBT Integration Testing")
    print("=" * 50)
    print(f"Test started at: {datetime.now()}")
    print()
    
    tests = [
        ("Basic Backtesting", test_basic_backtesting),
        ("Signal Generation", test_signal_generation),
        ("Parameter Optimization", test_parameter_optimization),
        ("Performance Benchmarking", test_performance_benchmarking),
        ("Fitness Calculation", test_fitness_calculation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("VECTORBT INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
    
    print("-" * 50)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All VectorBT integration tests passed!")
        print("\nNext steps:")
        print("1. Implement Fibonacci price action strategies")
        print("2. Add support/resistance detection")
        print("3. Build genetic algorithm optimization")
    else:
        print("⚠️  Some tests failed. Check implementation and fix issues.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)