"""
Quality Evaluation Module for DarwinTD.

This module contains engines for evaluating the quality of detected trading setups.
Quality evaluation is separate from setup detection to allow independent optimization
and testing of different quality assessment methods.

Available Quality Engines:
- Technical: Evaluates setups based on technical analysis criteria
- Statistical: Uses statistical methods and historical performance
- Risk: Focuses on risk management and position sizing
- Market: Considers market conditions and context
- Confluence: Evaluates confluence of multiple factors

Each engine takes SetupData and outputs a quality score and analysis.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

from ..setup_detection import SetupData

@dataclass
class QualityScore:
    """Quality evaluation result for a trading setup."""
    setup_id: str
    overall_score: float  # 0.0 to 1.0
    technical_score: float
    risk_score: float
    market_score: float
    confidence_score: float
    reasons: List[str]  # Explanation of scoring
    recommendations: List[str]  # Trading recommendations
    metadata: Dict[str, Any]


class BaseQualityEvaluator(ABC):
    """Abstract base class for all quality evaluation engines."""
    
    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self.weights = {
            'technical': 0.3,
            'risk': 0.3,
            'market': 0.2,
            'confidence': 0.2
        }
    
    @abstractmethod
    def evaluate_setup(self, setup: SetupData, market_data: pd.DataFrame, parameters: Dict[str, Any]) -> QualityScore:
        """Evaluate the quality of a trading setup."""
        pass
    
    @abstractmethod
    def get_parameter_ranges(self) -> Dict[str, tuple]:
        """Get parameter ranges for optimization."""
        pass
    
    def calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall quality score."""
        return sum(scores.get(key, 0) * weight for key, weight in self.weights.items())
    
    def set_weights(self, weights: Dict[str, float]):
        """Set custom weights for score components."""
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")
        self.weights.update(weights)