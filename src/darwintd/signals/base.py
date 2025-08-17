"""
Base signal generator classes for DarwinTD evolutionary trading system.

Provides abstract interfaces for creating trading signals that can be optimized
through genetic algorithms and backtested with VectorBT.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Types of trading signals."""
    ENTRY_LONG = "entry_long"
    EXIT_LONG = "exit_long"
    ENTRY_SHORT = "entry_short"
    EXIT_SHORT = "exit_short"


@dataclass
class BacktestData:
    """
    Standardized data container for backtesting.
    
    Contains OHLCV data and derived features needed for signal generation.
    """
    ohlcv: pd.DataFrame  # Open, High, Low, Close, Volume
    features: pd.DataFrame  # Derived features (swing points, levels, etc.)
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate data consistency."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.ohlcv.columns]
        if missing_columns:
            raise ValueError(f"Missing required OHLCV columns: {missing_columns}")
        
        if not isinstance(self.ohlcv.index, pd.DatetimeIndex):
            raise ValueError("OHLCV data must have DatetimeIndex")
    
    @property
    def close(self) -> pd.Series:
        """Convenience property for close prices."""
        return self.ohlcv['close']
    
    @property
    def high(self) -> pd.Series:
        """Convenience property for high prices."""
        return self.ohlcv['high']
    
    @property
    def low(self) -> pd.Series:
        """Convenience property for low prices."""
        return self.ohlcv['low']
    
    @property
    def volume(self) -> pd.Series:
        """Convenience property for volume."""
        return self.ohlcv['volume']


class BaseSignalGenerator(ABC):
    """
    Abstract base class for all signal generators.
    
    Signal generators create entry/exit signals from price data and can be
    optimized using genetic algorithms.
    """
    
    def __init__(self, name: str):
        """Initialize signal generator with a name."""
        self.name = name
        self._last_signals = None
        self._parameter_ranges = {}
    
    @abstractmethod
    def generate_signals(self, data: BacktestData, parameters: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate trading signals from data and parameters.
        
        Args:
            data: BacktestData containing OHLCV and features
            parameters: Dictionary of strategy parameters
            
        Returns:
            Tuple of (entry_signals, exit_signals) as boolean numpy arrays
        """
        pass
    
    @abstractmethod
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Get parameter ranges for genetic algorithm optimization.
        
        Returns:
            Dictionary mapping parameter names to (min, max) tuples
        """
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate that parameters are within acceptable ranges.
        
        Args:
            parameters: Parameter dictionary to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        try:
            ranges = self.get_parameter_ranges()
            
            for param_name, (min_val, max_val) in ranges.items():
                if param_name not in parameters:
                    return False
                
                value = parameters[param_name]
                if not min_val <= value <= max_val:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def generate_random_parameters(self, num_sets: int = 1) -> List[Dict[str, Any]]:
        """
        Generate random parameter sets within valid ranges.
        
        Args:
            num_sets: Number of parameter sets to generate
            
        Returns:
            List of parameter dictionaries
        """
        ranges = self.get_parameter_ranges()
        parameter_sets = []
        
        for _ in range(num_sets):
            params = {}
            for param_name, (min_val, max_val) in ranges.items():
                # Generate random value in range
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    params[param_name] = np.random.uniform(min_val, max_val)
            
            parameter_sets.append(params)
        
        return parameter_sets
    
    def calculate_signal_quality(self, data: BacktestData, signals: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """
        Calculate quality metrics for generated signals.
        
        Args:
            data: BacktestData used for signal generation
            signals: Tuple of (entries, exits) signals
            
        Returns:
            Dictionary of signal quality metrics
        """
        entries, exits = signals
        
        # Basic signal statistics
        total_bars = len(entries)
        entry_count = np.sum(entries)
        exit_count = np.sum(exits)
        
        # Signal frequency
        entry_frequency = entry_count / total_bars if total_bars > 0 else 0
        exit_frequency = exit_count / total_bars if total_bars > 0 else 0
        
        # Signal balance (entries should roughly match exits)
        balance_ratio = min(entry_count, exit_count) / max(entry_count, exit_count) if max(entry_count, exit_count) > 0 else 0
        
        # Consecutive signal clustering (prefer distributed signals)
        entry_changes = np.diff(entries.astype(int))
        entry_clusters = np.sum(entry_changes != 0) / 2 if len(entry_changes) > 0 else 0
        clustering_score = entry_clusters / entry_count if entry_count > 0 else 0
        
        return {
            'entry_frequency': entry_frequency,
            'exit_frequency': exit_frequency,
            'balance_ratio': balance_ratio,
            'clustering_score': clustering_score,
            'total_entries': entry_count,
            'total_exits': exit_count
        }
    
    def get_last_signals(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get the last generated signals for analysis."""
        return self._last_signals


class PriceActionSignalGenerator(BaseSignalGenerator):
    """
    Base class for price action-based signal generators.
    
    Provides common functionality for analyzing price movements, swing points,
    and volume patterns without traditional indicators.
    """
    
    def __init__(self, name: str):
        super().__init__(name)
    
    def find_swing_points(self, prices: pd.Series, lookback: int = 20) -> Tuple[pd.Series, pd.Series]:
        """
        Find swing highs and lows in price data.
        
        Args:
            prices: Price series (typically high for swing highs, low for swing lows)
            lookback: Number of periods to look back/forward for swing validation
            
        Returns:
            Tuple of (swing_highs, swing_lows) as boolean Series
        """
        swing_highs = pd.Series(False, index=prices.index)
        swing_lows = pd.Series(False, index=prices.index)
        
        for i in range(lookback, len(prices) - lookback):
            window = prices.iloc[i-lookback:i+lookback+1]
            center_idx = lookback
            center_price = window.iloc[center_idx]
            
            # Check for swing high
            if center_price == window.max():
                swing_highs.iloc[i] = True
            
            # Check for swing low
            if center_price == window.min():
                swing_lows.iloc[i] = True
        
        return swing_highs, swing_lows
    
    def calculate_price_levels(self, highs: pd.Series, lows: pd.Series, tolerance: float = 0.01) -> List[float]:
        """
        Calculate significant price levels from swing points.
        
        Args:
            highs: Series of swing high prices
            lows: Series of swing low prices
            tolerance: Price tolerance for level clustering (as percentage)
            
        Returns:
            List of significant price levels
        """
        # Combine all swing levels
        all_levels = []
        all_levels.extend(highs[highs.notna()].values)
        all_levels.extend(lows[lows.notna()].values)
        
        if not all_levels:
            return []
        
        # Cluster nearby levels
        levels = sorted(all_levels)
        clustered_levels = []
        
        i = 0
        while i < len(levels):
            cluster = [levels[i]]
            j = i + 1
            
            # Find all levels within tolerance of current level
            while j < len(levels) and abs(levels[j] - levels[i]) / levels[i] <= tolerance:
                cluster.append(levels[j])
                j += 1
            
            # Use average of cluster as representative level
            clustered_levels.append(np.mean(cluster))
            i = j
        
        return clustered_levels
    
    def detect_breakouts(self, prices: pd.Series, levels: List[float], tolerance: float = 0.005) -> pd.Series:
        """
        Detect price breakouts above/below significant levels.
        
        Args:
            prices: Price series
            levels: List of significant price levels
            tolerance: Breakout confirmation tolerance
            
        Returns:
            Series with breakout signals (1 for upward breakout, -1 for downward, 0 for none)
        """
        breakouts = pd.Series(0, index=prices.index)
        
        if not levels:
            return breakouts
        
        for i in range(1, len(prices)):
            current_price = prices.iloc[i]
            previous_price = prices.iloc[i-1]
            
            for level in levels:
                # Upward breakout
                if previous_price <= level and current_price > level * (1 + tolerance):
                    breakouts.iloc[i] = 1
                    break
                
                # Downward breakout
                if previous_price >= level and current_price < level * (1 - tolerance):
                    breakouts.iloc[i] = -1
                    break
        
        return breakouts


class VolumeSignalGenerator(BaseSignalGenerator):
    """
    Base class for volume-based signal generators.
    
    Provides functionality for volume profile analysis, volume anomaly detection,
    and volume-price relationship analysis.
    """
    
    def __init__(self, name: str):
        super().__init__(name)
    
    def calculate_volume_profile(self, ohlc: pd.DataFrame, volume: pd.Series, bins: int = 100) -> Dict[str, Any]:
        """
        Calculate volume profile (volume distributed across price levels).
        
        Args:
            ohlc: OHLC price data
            volume: Volume series
            bins: Number of price bins for volume distribution
            
        Returns:
            Dictionary containing volume profile data
        """
        if len(ohlc) == 0:
            return {'price_levels': [], 'volume_distribution': [], 'poc': None}
        
        # Create price range from low to high
        min_price = ohlc['low'].min()
        max_price = ohlc['high'].max()
        price_bins = np.linspace(min_price, max_price, bins + 1)
        
        # Initialize volume distribution
        volume_dist = np.zeros(bins)
        
        # Distribute volume across price levels for each bar
        for i in range(len(ohlc)):
            bar_low = ohlc['low'].iloc[i]
            bar_high = ohlc['high'].iloc[i]
            bar_volume = volume.iloc[i]
            
            # Find bins that overlap with this price bar
            start_bin = np.searchsorted(price_bins, bar_low, side='left')
            end_bin = np.searchsorted(price_bins, bar_high, side='right')
            
            start_bin = max(0, start_bin - 1)
            end_bin = min(bins, end_bin)
            
            # Distribute volume evenly across overlapping bins
            if end_bin > start_bin:
                volume_per_bin = bar_volume / (end_bin - start_bin)
                volume_dist[start_bin:end_bin] += volume_per_bin
        
        # Calculate Point of Control (price level with highest volume)
        poc_bin = np.argmax(volume_dist)
        poc_price = (price_bins[poc_bin] + price_bins[poc_bin + 1]) / 2
        
        # Calculate price levels (center of each bin)
        price_levels = [(price_bins[i] + price_bins[i + 1]) / 2 for i in range(bins)]
        
        return {
            'price_levels': price_levels,
            'volume_distribution': volume_dist,
            'poc': poc_price,
            'poc_volume': volume_dist[poc_bin]
        }
    
    def detect_volume_anomalies(self, volume: pd.Series, threshold: float = 2.0) -> pd.Series:
        """
        Detect volume anomalies (unusually high or low volume).
        
        Args:
            volume: Volume series
            threshold: Number of standard deviations for anomaly detection
            
        Returns:
            Series with anomaly signals (1 for high volume, -1 for low volume, 0 for normal)
        """
        # Calculate rolling mean and standard deviation
        window = min(50, len(volume) // 4)  # Adaptive window size
        rolling_mean = volume.rolling(window=window, min_periods=10).mean()
        rolling_std = volume.rolling(window=window, min_periods=10).std()
        
        # Calculate z-scores
        z_scores = (volume - rolling_mean) / rolling_std
        
        # Identify anomalies
        anomalies = pd.Series(0, index=volume.index)
        anomalies[z_scores > threshold] = 1  # High volume
        anomalies[z_scores < -threshold] = -1  # Low volume
        
        return anomalies