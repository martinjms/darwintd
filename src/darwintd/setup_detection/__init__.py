"""
Setup Detection Module for DarwinTD.

This module contains various engines for detecting trading setups in price data.
Each engine focuses on identifying specific price action patterns that indicate
high-probability trading opportunities.

Available Detection Engines:
- Fibonacci: Detects Fibonacci retracement and extension setups
- Support/Resistance: Identifies key horizontal levels and breakouts
- Volume: Analyzes volume-based setup patterns
- Confluence: Combines multiple detection methods for higher probability setups

Each engine outputs standardized setup data that can be evaluated by quality engines.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np

@dataclass
class SetupData:
    """Standardized setup data structure for all detection engines."""
    timestamp: datetime
    symbol: str
    timeframe: str
    setup_type: str
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward_ratio: Optional[float]
    metadata: Dict[str, Any]
    chart_data: Optional[Dict[str, Any]] = None  # For visualization


class BaseSetupDetector(ABC):
    """Abstract base class for all setup detection engines."""
    
    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self.parameters = {}
    
    @abstractmethod
    def detect_setups(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> List[SetupData]:
        """Detect trading setups in the given data."""
        pass
    
    @abstractmethod
    def get_parameter_ranges(self) -> Dict[str, tuple]:
        """Get parameter ranges for optimization."""
        pass
    
    def validate_setup(self, setup: SetupData) -> bool:
        """Validate a detected setup for basic consistency."""
        return (
            0.0 <= setup.confidence <= 1.0 and
            setup.entry_price > 0 and
            (setup.stop_loss is None or setup.stop_loss > 0) and
            (setup.take_profit is None or setup.take_profit > 0)
        )