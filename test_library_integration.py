#!/usr/bin/env python3
"""
Library Integration Testing Script for DarwinTD
Tests core trading libraries to validate they work together.
"""

import sys
import traceback
from datetime import datetime, timedelta

def test_library_import():
    """Test if core libraries can be imported."""
    print("=== Testing Library Imports ===")
    
    # Core data libraries
    try:
        import pandas as pd
        import numpy as np
        print("✅ pandas and numpy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import pandas/numpy: {e}")
        return False
    
    # CCXT for exchange APIs
    try:
        import ccxt
        print(f"✅ CCXT imported successfully (version: {ccxt.__version__})")
        print(f"   Supported exchanges: {len(ccxt.exchanges)} exchanges")
    except ImportError as e:
        print(f"❌ Failed to import CCXT: {e}")
        print("   Install with: pip install ccxt")
        return False
    
    # pandas-ta for technical analysis
    try:
        import pandas_ta as ta
        print(f"✅ pandas-ta imported successfully")
        print(f"   Available indicators: {len(ta.CommonKeys.INDICATORS)} indicators")
    except ImportError as e:
        print(f"❌ Failed to import pandas-ta: {e}")
        print("   Install with: pip install pandas-ta")
        return False
    
    # plotly for visualization
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        print("✅ plotly imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import plotly: {e}")
        print("   Install with: pip install plotly")
        return False
    
    # Optional: TA-Lib (may not be installed)
    try:
        import talib
        print(f"✅ TA-Lib imported successfully")
    except ImportError as e:
        print(f"⚠️  TA-Lib not available: {e}")
        print("   This is optional for MVP")
    
    # Optional: Jesse Framework
    try:
        import jesse
        print(f"✅ Jesse Framework imported successfully")
    except ImportError as e:
        print(f"⚠️  Jesse Framework not available: {e}")
        print("   Install with: pip install jesse")
    
    return True

def test_ccxt_exchanges():
    """Test CCXT exchange connectivity (paper trading)."""
    print("\n=== Testing CCXT Exchange Connectivity ===")
    
    try:
        import ccxt
        
        # Test major exchanges we'll use
        exchanges_to_test = ['binance', 'coinbase', 'kraken']
        
        for exchange_id in exchanges_to_test:
            try:
                if exchange_id in ccxt.exchanges:
                    exchange_class = getattr(ccxt, exchange_id)
                    exchange = exchange_class({
                        'sandbox': True,  # Use testnet/sandbox if available
                        'enableRateLimit': True,
                    })
                    
                    # Test basic functionality
                    markets = exchange.load_markets()
                    print(f"✅ {exchange_id}: {len(markets)} trading pairs available")
                    
                    # Test if our target symbols are available
                    target_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'LINK/USDT']
                    available_targets = [symbol for symbol in target_symbols if symbol in markets]
                    print(f"   Target symbols available: {len(available_targets)}/{len(target_symbols)}")
                    
                else:
                    print(f"❌ {exchange_id}: Not found in CCXT")
                    
            except Exception as e:
                print(f"❌ {exchange_id}: Connection failed - {str(e)[:100]}")
                
    except Exception as e:
        print(f"❌ CCXT exchange testing failed: {e}")
        return False
    
    return True

def test_technical_analysis():
    """Test technical analysis calculations."""
    print("\n=== Testing Technical Analysis ===")
    
    try:
        import pandas as pd
        import numpy as np
        import pandas_ta as ta
        
        # Create sample OHLCV data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
        np.random.seed(42)  # For reproducible results
        
        # Generate realistic price data
        base_price = 50000  # Starting at $50k (like BTC)
        price_changes = np.random.normal(0, 0.01, len(dates))  # 1% hourly volatility
        prices = base_price * np.exp(np.cumsum(price_changes))
        
        # Create OHLCV DataFrame
        data = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
            'low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
            'close': prices,
            'volume': np.random.uniform(100, 1000, len(dates))
        }, index=dates)
        
        # Ensure OHLC relationships are correct
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        print(f"✅ Created sample data: {len(data)} rows")
        print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        
        # Test basic indicators
        try:
            # Moving averages
            sma_20 = ta.sma(data['close'], length=20)
            ema_12 = ta.ema(data['close'], length=12)
            print(f"✅ Moving averages: SMA(20), EMA(12)")
            
            # VWAP
            vwap = ta.vwap(data['high'], data['low'], data['close'], data['volume'])
            print(f"✅ VWAP calculated")
            
            # Bollinger Bands
            bbands = ta.bbands(data['close'], length=20)
            print(f"✅ Bollinger Bands calculated")
            
            # RSI
            rsi = ta.rsi(data['close'], length=14)
            print(f"✅ RSI calculated")
            
            # Volume indicators
            obv = ta.obv(data['close'], data['volume'])
            print(f"✅ On-Balance Volume calculated")
            
        except Exception as e:
            print(f"❌ Indicator calculation failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Technical analysis testing failed: {e}")
        return False
    
    return True

def test_visualization():
    """Test visualization capabilities."""
    print("\n=== Testing Visualization ===")
    
    try:
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        np.random.seed(42)
        
        prices = 50000 * np.exp(np.cumsum(np.random.normal(0, 0.02, 30)))
        volumes = np.random.uniform(100, 1000, 30)
        
        # Test candlestick chart creation
        fig = go.Figure(data=go.Candlestick(
            x=dates,
            open=prices,
            high=prices * 1.02,
            low=prices * 0.98,
            close=prices,
            name="BTC/USDT"
        ))
        
        fig.update_layout(
            title="Sample BTC/USDT Chart",
            xaxis_title="Date",
            yaxis_title="Price (USDT)",
            height=400
        )
        
        print("✅ Candlestick chart created successfully")
        
        # Test if we can save the chart
        try:
            fig.write_html("/tmp/test_chart.html")
            print("✅ Chart saved to HTML successfully")
        except Exception as e:
            print(f"⚠️  Chart saving failed: {e}")
        
    except Exception as e:
        print(f"❌ Visualization testing failed: {e}")
        return False
    
    return True

def test_data_pipeline():
    """Test basic data pipeline integration."""
    print("\n=== Testing Data Pipeline Integration ===")
    
    try:
        import pandas as pd
        import pandas_ta as ta
        
        # Simulate a basic data pipeline
        # 1. Data collection (simulated)
        print("1. Simulating data collection...")
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'open': np.random.uniform(49000, 51000, 100),
            'high': np.random.uniform(50000, 52000, 100),
            'low': np.random.uniform(48000, 50000, 100),
            'close': np.random.uniform(49000, 51000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        })
        
        # 2. Data validation
        print("2. Validating data...")
        # Check OHLC relationships
        valid_ohlc = (
            (data['high'] >= data['open']) &
            (data['high'] >= data['close']) &
            (data['low'] <= data['open']) &
            (data['low'] <= data['close']) &
            (data['volume'] >= 0)
        ).all()
        
        if not valid_ohlc:
            print("❌ OHLC validation failed")
            return False
        
        print("✅ OHLC validation passed")
        
        # 3. Technical analysis
        print("3. Calculating indicators...")
        data['sma_20'] = ta.sma(data['close'], length=20)
        data['rsi'] = ta.rsi(data['close'], length=14)
        
        # 4. Data storage simulation
        print("4. Simulating data storage...")
        # In real implementation, this would save to database
        print(f"   Data shape: {data.shape}")
        print(f"   Indicators calculated: SMA(20), RSI(14)")
        
        print("✅ Basic data pipeline test completed")
        
    except Exception as e:
        print(f"❌ Data pipeline testing failed: {e}")
        return False
    
    return True

def main():
    """Run all integration tests."""
    print("DarwinTD Library Integration Testing")
    print("=" * 50)
    
    tests = [
        test_library_import,
        test_ccxt_exchanges,
        test_technical_analysis,
        test_visualization,
        test_data_pipeline
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 50)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All integration tests passed! Library stack is ready.")
    else:
        print("⚠️  Some tests failed. Check installation and configuration.")
        print("\nRecommended actions:")
        print("1. Install missing libraries: pip install ccxt pandas-ta plotly")
        print("2. Check internet connectivity for exchange tests")
        print("3. Review error messages above")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)