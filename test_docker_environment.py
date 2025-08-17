#!/usr/bin/env python3
"""
Docker Environment Testing Script for DarwinTD
Tests that all required libraries work correctly in Docker environment.
"""

import sys
import traceback
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def test_core_libraries():
    """Test core scientific computing libraries."""
    print("=== Testing Core Libraries ===")
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} imported successfully")
        
        # Test basic numpy operations
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.sum() == 15, "NumPy computation failed"
        print("   NumPy operations working correctly")
        
    except Exception as e:
        print(f"❌ NumPy test failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__} imported successfully")
        
        # Test basic pandas operations
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        assert len(df) == 3, "Pandas DataFrame creation failed"
        print("   Pandas operations working correctly")
        
    except Exception as e:
        print(f"❌ Pandas test failed: {e}")
        return False
        
    try:
        import scipy
        print(f"✅ SciPy {scipy.__version__} imported successfully")
        
    except Exception as e:
        print(f"❌ SciPy test failed: {e}")
        return False
    
    return True

def test_vectorbt():
    """Test VectorBT installation and basic functionality."""
    print("\n=== Testing VectorBT ===")
    
    try:
        import vectorbt as vbt
        print(f"✅ VectorBT {vbt.__version__} imported successfully")
        
        # Test basic VectorBT functionality
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
        data = pd.Series(prices, index=dates, name='price')
        
        # Test moving average (basic VectorBT operation)
        ma = vbt.MA.run(data, window=10)
        assert len(ma.ma) == len(data), "VectorBT moving average failed"
        print("   VectorBT moving average calculation working")
        
        # Test portfolio construction (core functionality we'll use)
        entries = data.pct_change() > 0.01  # Simple entry signal
        exits = data.pct_change() < -0.01   # Simple exit signal
        
        portfolio = vbt.Portfolio.from_signals(data, entries, exits, init_cash=10000)
        assert hasattr(portfolio, 'total_return'), "Portfolio construction failed"
        print("   VectorBT portfolio construction working")
        
        # Test performance metrics
        sharpe = portfolio.sharpe_ratio()
        max_dd = portfolio.max_drawdown()
        print(f"   Sample portfolio metrics: Sharpe={sharpe:.3f}, MaxDD={max_dd:.3f}")
        
    except Exception as e:
        print(f"❌ VectorBT test failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_numba():
    """Test numba JIT compilation (VectorBT dependency)."""
    print("\n=== Testing Numba ===")
    
    try:
        import numba
        from numba import jit
        print(f"✅ Numba {numba.__version__} imported successfully")
        
        # Test JIT compilation
        @jit(nopython=True)
        def test_function(x):
            return x * 2 + 1
        
        result = test_function(5)
        assert result == 11, "Numba JIT compilation failed"
        print("   Numba JIT compilation working correctly")
        
    except Exception as e:
        print(f"❌ Numba test failed: {e}")
        return False
    
    return True

def test_genetic_algorithms():
    """Test DEAP genetic algorithm library."""
    print("\n=== Testing Genetic Algorithm Libraries ===")
    
    try:
        import numpy as np
        import deap
        from deap import base, creator, tools, algorithms
        print(f"✅ DEAP imported successfully")
        
        # Test basic DEAP functionality
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_float", np.random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=5)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Create test population
        pop = toolbox.population(n=10)
        assert len(pop) == 10, "DEAP population creation failed"
        assert len(pop[0]) == 5, "DEAP individual creation failed"
        print("   DEAP population creation working correctly")
        
    except Exception as e:
        print(f"❌ DEAP test failed: {e}")
        return False
    
    return True

def test_exchange_apis():
    """Test CCXT exchange API library."""
    print("\n=== Testing Exchange APIs ===")
    
    try:
        import ccxt
        print(f"✅ CCXT {ccxt.__version__} imported successfully")
        print(f"   Supports {len(ccxt.exchanges)} exchanges")
        
        # Test exchange instantiation (without API keys)
        exchange = ccxt.binance({'sandbox': True, 'enableRateLimit': True})
        assert hasattr(exchange, 'load_markets'), "CCXT exchange creation failed"
        print("   CCXT exchange instantiation working")
        
        # Test that our target symbols are supported
        target_exchanges = ['binance', 'coinbase', 'kraken']
        for exchange_id in target_exchanges:
            if exchange_id in ccxt.exchanges:
                print(f"   ✅ {exchange_id} supported")
            else:
                print(f"   ❌ {exchange_id} not found")
        
    except Exception as e:
        print(f"❌ CCXT test failed: {e}")
        return False
    
    return True

def test_visualization():
    """Test plotly visualization library."""
    print("\n=== Testing Visualization ===")
    
    try:
        import plotly
        import plotly.graph_objects as go
        import plotly.express as px
        print(f"✅ Plotly {plotly.__version__} imported successfully")
        
        # Test basic chart creation
        fig = go.Figure(data=go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13]))
        fig.update_layout(title="Test Chart")
        
        # Test if we can convert to HTML (for saving)
        html_str = fig.to_html()
        assert len(html_str) > 100, "Plotly HTML generation failed"
        print("   Plotly chart creation and HTML export working")
        
    except Exception as e:
        print(f"❌ Plotly test failed: {e}")
        return False
    
    return True

def test_database_connection():
    """Test database connectivity."""
    print("\n=== Testing Database ===")
    
    try:
        import sqlalchemy
        from sqlalchemy import create_engine, text
        print(f"✅ SQLAlchemy {sqlalchemy.__version__} imported successfully")
        
        # Test SQLite (always available)
        engine = create_engine('sqlite:///:memory:')
        with engine.connect() as conn:
            result = conn.execute(text('SELECT 1 as test'))
            assert result.fetchone()[0] == 1, "SQLite test query failed"
        print("   SQLite database connection working")
        
        # Test PostgreSQL connection (if available)
        try:
            import psycopg2
            print(f"   PostgreSQL adapter (psycopg2) available")
            
            # Try to connect to postgres service (if running in docker-compose)
            try:
                pg_engine = create_engine('postgresql://darwintd:darwintd_pass@postgres:5432/darwintd')
                with pg_engine.connect() as conn:
                    result = conn.execute(text('SELECT 1 as test'))
                    assert result.fetchone()[0] == 1, "PostgreSQL test query failed"
                print("   ✅ PostgreSQL database connection working")
            except Exception as e:
                print(f"   ⚠️  PostgreSQL connection failed (expected if not in docker-compose): {e}")
                
        except ImportError:
            print("   ❌ PostgreSQL adapter not available")
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False
    
    return True

def test_environment_setup():
    """Test overall environment setup."""
    print("\n=== Testing Environment Setup ===")
    
    try:
        # Test Python version
        python_version = sys.version_info
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version < (3, 9):
            print("   ⚠️  Python version < 3.9, may have compatibility issues")
        
        # Test that we can import our package
        try:
            import darwintd
            print(f"✅ DarwinTD package importable (version {darwintd.__version__})")
        except ImportError as e:
            print(f"   ⚠️  DarwinTD package not installed in development mode: {e}")
        
        # Test data directories exist
        import os
        data_dir = os.environ.get('DATA_DIR', './data')
        cache_dir = os.environ.get('CACHE_DIR', './cache')
        
        print(f"   Data directory: {data_dir}")
        print(f"   Cache directory: {cache_dir}")
        
    except Exception as e:
        print(f"❌ Environment setup test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests and report results."""
    print("DarwinTD Docker Environment Testing")
    print("=" * 50)
    print(f"Test started at: {datetime.now()}")
    print()
    
    tests = [
        ("Core Libraries", test_core_libraries),
        ("VectorBT", test_vectorbt),
        ("Numba JIT", test_numba),
        ("Genetic Algorithms", test_genetic_algorithms),
        ("Exchange APIs", test_exchange_apis),
        ("Visualization", test_visualization),
        ("Database", test_database_connection),
        ("Environment", test_environment_setup),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("DOCKER ENVIRONMENT TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print("-" * 50)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Docker environment is ready for development.")
        print("\nNext steps:")
        print("1. Start working on VectorBT core integration")
        print("2. Begin implementing price action algorithms")
        print("3. Set up genetic algorithm framework")
    else:
        print("⚠️  Some tests failed. Check installation and configuration.")
        print("\nTroubleshooting:")
        print("1. Rebuild Docker image: docker-compose build --no-cache")
        print("2. Check Docker logs: docker-compose logs darwintd")
        print("3. Run interactive shell: docker-compose run darwintd bash")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)