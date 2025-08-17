"""
Base Fibonacci detection engine.
"""

from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from .. import BaseSetupDetector, SetupData


class FibonacciDetector(BaseSetupDetector):
    """Base class for Fibonacci retracement/extension detection."""
    
    def __init__(self, name: str = "FibonacciDetector", version: str = "base"):
        super().__init__(name, version)
        self.fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.extension_levels = [1.272, 1.414, 1.618, 2.0, 2.618]
    
    def find_swing_points(self, highs: pd.Series, lows: pd.Series, lookback: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Find swing highs and lows in price data."""
        swing_highs = pd.Series(False, index=highs.index)
        swing_lows = pd.Series(False, index=lows.index)
        
        for i in range(lookback, len(highs) - lookback):
            # Check for swing high
            window_highs = highs.iloc[i-lookback:i+lookback+1]
            if highs.iloc[i] == window_highs.max():
                swing_highs.iloc[i] = True
            
            # Check for swing low
            window_lows = lows.iloc[i-lookback:i+lookback+1]
            if lows.iloc[i] == window_lows.min():
                swing_lows.iloc[i] = True
        
        return swing_highs, swing_lows
    
    def calculate_fibonacci_levels(self, high_price: float, low_price: float, direction: str = "retracement") -> Dict[str, float]:
        """Calculate Fibonacci levels between high and low prices."""
        price_range = high_price - low_price
        
        if direction == "retracement":
            levels = {}
            for level in self.fib_levels:
                levels[f"fib_{level}"] = high_price - (price_range * level)
            return levels
        
        elif direction == "extension":
            levels = {}
            for level in self.extension_levels:
                levels[f"ext_{level}"] = high_price + (price_range * (level - 1))
            return levels
        
        return {}
    
    def detect_fibonacci_confluence(self, price: float, fib_levels: Dict[str, float], tolerance: float = 0.005) -> List[str]:
        """Detect if price is near multiple Fibonacci levels (confluence)."""
        confluences = []
        
        for level_name, level_price in fib_levels.items():
            if abs(price - level_price) / price <= tolerance:
                confluences.append(level_name)
        
        return confluences