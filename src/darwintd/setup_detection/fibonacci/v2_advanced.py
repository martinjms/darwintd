"""
Fibonacci Detector V2 - Advanced Implementation.

Enhanced Fibonacci detection with multiple timeframe analysis and extension levels.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime

from .base import FibonacciDetector
from .. import SetupData


class FibonacciDetectorV2(FibonacciDetector):
    """Advanced Fibonacci detector with multiple timeframe analysis."""
    
    def __init__(self):
        super().__init__("FibonacciDetectorV2", "2.0")
    
    def detect_setups(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> List[SetupData]:
        """Detect advanced Fibonacci setups with multiple confirmations."""
        setups = []
        
        # Parameters
        swing_lookback = parameters.get('swing_lookback', 20)
        min_swing_size = parameters.get('min_swing_size', 0.02)
        fib_tolerance = parameters.get('fib_tolerance', 0.005)
        confluence_threshold = parameters.get('confluence_threshold', 2)
        volume_confirmation = parameters.get('volume_confirmation', True)
        
        # Find swing points with multiple lookback periods
        swing_data = self._multi_timeframe_swings(data, [swing_lookback, swing_lookback*2])
        
        # Calculate volume profile for additional confluence
        volume_levels = self._calculate_volume_levels(data) if volume_confirmation else {}
        
        for i in range(len(data) - 1):
            current_time = data.index[i]
            current_bar = data.iloc[i]
            
            # Analyze potential setups
            setup = self._analyze_fibonacci_setup(
                current_time, current_bar, swing_data, volume_levels, parameters
            )
            
            if setup and setup.confidence >= 0.3:  # Minimum confidence threshold
                setups.append(setup)
        
        return setups
    
    def _multi_timeframe_swings(self, data: pd.DataFrame, lookbacks: List[int]) -> Dict[str, Any]:
        """Find swing points across multiple timeframes."""
        all_swings = {}
        
        for lookback in lookbacks:
            swing_highs, swing_lows = self.find_swing_points(
                data['high'], data['low'], lookback
            )
            
            all_swings[f'tf_{lookback}'] = {
                'highs': [(idx, price) for idx, price in zip(data.index[swing_highs], data['high'][swing_highs])],
                'lows': [(idx, price) for idx, price in zip(data.index[swing_lows], data['low'][swing_lows])]
            }
        
        return all_swings
    
    def _calculate_volume_levels(self, data: pd.DataFrame, bins: int = 50) -> Dict[str, float]:
        """Calculate volume-weighted price levels."""
        min_price = data['low'].min()
        max_price = data['high'].max()
        price_bins = np.linspace(min_price, max_price, bins + 1)
        
        volume_dist = np.zeros(bins)
        
        for i in range(len(data)):
            bar_low = data['low'].iloc[i]
            bar_high = data['high'].iloc[i]
            bar_volume = data['volume'].iloc[i]
            
            start_bin = np.searchsorted(price_bins, bar_low)
            end_bin = np.searchsorted(price_bins, bar_high)
            
            start_bin = max(0, start_bin - 1)
            end_bin = min(bins, end_bin)
            
            if end_bin > start_bin:
                volume_per_bin = bar_volume / (end_bin - start_bin)
                volume_dist[start_bin:end_bin] += volume_per_bin
        
        # Find high volume nodes
        volume_threshold = np.percentile(volume_dist, 80)
        high_volume_levels = {}
        
        for i, volume in enumerate(volume_dist):
            if volume > volume_threshold:
                price_level = (price_bins[i] + price_bins[i + 1]) / 2
                high_volume_levels[f'vol_level_{i}'] = price_level
        
        return high_volume_levels
    
    def _analyze_fibonacci_setup(self, timestamp, bar_data, swing_data, volume_levels, parameters):
        """Analyze a potential Fibonacci setup with multiple confirmations."""
        current_price = bar_data['close']
        confluences = []
        confidence_factors = []
        
        # Check each timeframe for Fibonacci levels
        for tf_key, swings in swing_data.items():
            recent_high = self._find_recent_swing(swings['highs'], timestamp, 200)
            recent_low = self._find_recent_swing(swings['lows'], timestamp, 200)
            
            if recent_high and recent_low:
                high_time, high_price = recent_high
                low_time, low_price = recent_low
                
                # Calculate Fibonacci levels
                if high_time > low_time:  # Uptrend
                    fib_levels = self.calculate_fibonacci_levels(high_price, low_price, "retracement")
                    ext_levels = self.calculate_fibonacci_levels(high_price, low_price, "extension")
                    
                    # Check for Fibonacci confluence
                    fib_confluence = self.detect_fibonacci_confluence(
                        current_price, {**fib_levels, **ext_levels}, parameters['fib_tolerance']
                    )
                    
                    if fib_confluence:
                        confluences.extend([f"{tf_key}_{conf}" for conf in fib_confluence])
                        confidence_factors.append(0.3)
                
                elif low_time > high_time:  # Downtrend
                    fib_levels = self.calculate_fibonacci_levels(low_price, high_price, "retracement")
                    ext_levels = self.calculate_fibonacci_levels(low_price, high_price, "extension")
                    
                    fib_confluence = self.detect_fibonacci_confluence(
                        current_price, {**fib_levels, **ext_levels}, parameters['fib_tolerance']
                    )
                    
                    if fib_confluence:
                        confluences.extend([f"{tf_key}_{conf}" for conf in fib_confluence])
                        confidence_factors.append(0.3)
        
        # Check volume level confluence
        volume_confluence = []
        for level_name, level_price in volume_levels.items():
            if abs(current_price - level_price) / current_price <= parameters['fib_tolerance']:
                volume_confluence.append(level_name)
                confidence_factors.append(0.2)
        
        # Calculate total confluence score
        total_confluences = len(confluences) + len(volume_confluence)
        
        if total_confluences >= parameters.get('confluence_threshold', 2):
            confidence = min(sum(confidence_factors), 1.0)
            
            # Determine setup direction and create setup
            if any('uptrend' in conf or 'fib_0' in conf for conf in confluences):
                return self._create_advanced_long_setup(
                    timestamp, bar_data, confluences + volume_confluence, confidence
                )
            elif any('downtrend' in conf or 'ext_' in conf for conf in confluences):
                return self._create_advanced_short_setup(
                    timestamp, bar_data, confluences + volume_confluence, confidence
                )
        
        return None
    
    def _find_recent_swing(self, swing_points: List[tuple], current_time: datetime, lookback_bars: int) -> tuple:
        """Find most recent swing point."""
        for swing_time, swing_price in reversed(swing_points):
            if swing_time < current_time:
                return (swing_time, swing_price)
        return None
    
    def _create_advanced_long_setup(self, timestamp, bar_data, confluences, confidence):
        """Create advanced long setup."""
        entry_price = bar_data['close']
        atr = self._calculate_atr(bar_data, 14)  # Simplified ATR calculation
        
        stop_loss = entry_price - (2 * atr)
        take_profit = entry_price + (3 * atr)
        
        risk_reward = (take_profit - entry_price) / (entry_price - stop_loss) if entry_price > stop_loss else None
        
        return SetupData(
            timestamp=timestamp,
            symbol=bar_data.get('symbol', 'UNKNOWN'),
            timeframe='1H',
            setup_type='fibonacci_advanced_long',
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward,
            metadata={
                'confluences': confluences,
                'confluence_count': len(confluences),
                'atr_stop': True,
                'version': self.version
            }
        )
    
    def _create_advanced_short_setup(self, timestamp, bar_data, confluences, confidence):
        """Create advanced short setup."""
        entry_price = bar_data['close']
        atr = self._calculate_atr(bar_data, 14)
        
        stop_loss = entry_price + (2 * atr)
        take_profit = entry_price - (3 * atr)
        
        risk_reward = (entry_price - take_profit) / (stop_loss - entry_price) if stop_loss > entry_price else None
        
        return SetupData(
            timestamp=timestamp,
            symbol=bar_data.get('symbol', 'UNKNOWN'),
            timeframe='1H',
            setup_type='fibonacci_advanced_short',
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward,
            metadata={
                'confluences': confluences,
                'confluence_count': len(confluences),
                'atr_stop': True,
                'version': self.version
            }
        )
    
    def _calculate_atr(self, bar_data, period: int = 14) -> float:
        """Simplified ATR calculation for single bar."""
        # This is a simplified version - in practice, you'd need historical data
        high = bar_data.get('high', bar_data['close'])
        low = bar_data.get('low', bar_data['close'])
        return (high - low) * 0.5  # Simplified calculation
    
    def get_parameter_ranges(self) -> Dict[str, tuple]:
        """Get parameter ranges for optimization."""
        return {
            'swing_lookback': (15, 40),
            'min_swing_size': (0.015, 0.04),
            'fib_tolerance': (0.002, 0.008),
            'confluence_threshold': (2, 5),
            'volume_confirmation': (True, True)  # Boolean parameter
        }