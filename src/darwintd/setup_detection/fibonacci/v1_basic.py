"""
Fibonacci Detector V1 - Basic Implementation.

Simple Fibonacci retracement detection with basic swing point identification.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime

from .base import FibonacciDetector
from .. import SetupData


class FibonacciDetectorV1(FibonacciDetector):
    """Basic Fibonacci retracement detector."""
    
    def __init__(self):
        super().__init__("FibonacciDetectorV1", "1.0")
    
    def detect_setups(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> List[SetupData]:
        """Detect basic Fibonacci retracement setups."""
        setups = []
        
        # Get parameters
        swing_lookback = parameters.get('swing_lookback', 20)
        min_swing_size = parameters.get('min_swing_size', 0.02)  # 2% minimum swing
        fib_tolerance = parameters.get('fib_tolerance', 0.005)  # 0.5% tolerance
        
        # Find swing points
        swing_highs, swing_lows = self.find_swing_points(
            data['high'], data['low'], swing_lookback
        )
        
        # Get swing points as lists
        high_points = [(idx, price) for idx, price in zip(data.index[swing_highs], data['high'][swing_highs])]
        low_points = [(idx, price) for idx, price in zip(data.index[swing_lows], data['low'][swing_lows])]
        
        # Look for retracement opportunities
        for i in range(len(data) - 1):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            
            # Find recent swing high and low
            recent_high = self._find_recent_swing(high_points, current_time, lookback_bars=100)
            recent_low = self._find_recent_swing(low_points, current_time, lookback_bars=100)
            
            if recent_high and recent_low:
                high_time, high_price = recent_high
                low_time, low_price = recent_low
                
                # Check if swing is significant enough
                swing_size = abs(high_price - low_price) / low_price
                if swing_size < min_swing_size:
                    continue
                
                # Determine trend direction
                if high_time > low_time:  # Uptrend, look for long setup
                    fib_levels = self.calculate_fibonacci_levels(high_price, low_price, "retracement")
                    confluences = self.detect_fibonacci_confluence(current_price, fib_levels, fib_tolerance)
                    
                    if confluences and current_price < high_price * 0.95:  # Must be in retracement
                        setup = self._create_long_setup(
                            current_time, data.iloc[i], confluences, high_price, low_price, fib_levels
                        )
                        if setup:
                            setups.append(setup)
                
                elif low_time > high_time:  # Downtrend, look for short setup
                    fib_levels = self.calculate_fibonacci_levels(low_price, high_price, "retracement")
                    confluences = self.detect_fibonacci_confluence(current_price, fib_levels, fib_tolerance)
                    
                    if confluences and current_price > low_price * 1.05:  # Must be in retracement
                        setup = self._create_short_setup(
                            current_time, data.iloc[i], confluences, high_price, low_price, fib_levels
                        )
                        if setup:
                            setups.append(setup)
        
        return setups
    
    def _find_recent_swing(self, swing_points: List[tuple], current_time: datetime, lookback_bars: int = 100) -> tuple:
        """Find the most recent swing point within lookback period."""
        for swing_time, swing_price in reversed(swing_points):
            if swing_time < current_time:
                return (swing_time, swing_price)
        return None
    
    def _create_long_setup(self, timestamp, bar_data, confluences, swing_high, swing_low, fib_levels):
        """Create a long setup from Fibonacci retracement."""
        entry_price = bar_data['close']
        stop_loss = swing_low * 0.99  # 1% below swing low
        take_profit = swing_high * 1.01  # 1% above swing high
        
        confidence = min(len(confluences) * 0.3, 1.0)  # More confluences = higher confidence
        risk_reward = (take_profit - entry_price) / (entry_price - stop_loss) if entry_price > stop_loss else None
        
        return SetupData(
            timestamp=timestamp,
            symbol=bar_data.get('symbol', 'UNKNOWN'),
            timeframe='1H',
            setup_type='fibonacci_long',
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward,
            metadata={
                'confluences': confluences,
                'swing_high': swing_high,
                'swing_low': swing_low,
                'fib_levels': fib_levels,
                'version': self.version
            }
        )
    
    def _create_short_setup(self, timestamp, bar_data, confluences, swing_high, swing_low, fib_levels):
        """Create a short setup from Fibonacci retracement."""
        entry_price = bar_data['close']
        stop_loss = swing_high * 1.01  # 1% above swing high
        take_profit = swing_low * 0.99  # 1% below swing low
        
        confidence = min(len(confluences) * 0.3, 1.0)
        risk_reward = (entry_price - take_profit) / (stop_loss - entry_price) if stop_loss > entry_price else None
        
        return SetupData(
            timestamp=timestamp,
            symbol=bar_data.get('symbol', 'UNKNOWN'),
            timeframe='1H',
            setup_type='fibonacci_short',
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward,
            metadata={
                'confluences': confluences,
                'swing_high': swing_high,
                'swing_low': swing_low,
                'fib_levels': fib_levels,
                'version': self.version
            }
        )
    
    def get_parameter_ranges(self) -> Dict[str, tuple]:
        """Get parameter ranges for optimization."""
        return {
            'swing_lookback': (10, 50),
            'min_swing_size': (0.01, 0.05),
            'fib_tolerance': (0.001, 0.01)
        }