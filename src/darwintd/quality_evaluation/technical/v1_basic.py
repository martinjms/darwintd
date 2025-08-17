"""
Technical Quality Evaluator V1 - Basic Implementation.

Evaluates setup quality based on fundamental technical analysis principles.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np

from .. import BaseQualityEvaluator, QualityScore
from ...setup_detection import SetupData


class TechnicalQualityV1(BaseQualityEvaluator):
    """Basic technical quality evaluation engine."""
    
    def __init__(self):
        super().__init__("TechnicalQualityV1", "1.0")
    
    def evaluate_setup(self, setup: SetupData, market_data: pd.DataFrame, parameters: Dict[str, Any]) -> QualityScore:
        """Evaluate setup quality using technical analysis."""
        reasons = []
        recommendations = []
        
        # Technical analysis scores
        trend_score = self._evaluate_trend_alignment(setup, market_data, parameters)
        level_score = self._evaluate_level_strength(setup, market_data, parameters)
        volume_score = self._evaluate_volume_confirmation(setup, market_data, parameters)
        momentum_score = self._evaluate_momentum(setup, market_data, parameters)
        
        technical_score = np.mean([trend_score, level_score, volume_score, momentum_score])
        
        # Risk evaluation
        risk_score = self._evaluate_risk_metrics(setup, parameters)
        
        # Market context
        market_score = self._evaluate_market_conditions(setup, market_data, parameters)
        
        # Confidence based on setup detection
        confidence_score = setup.confidence
        
        # Calculate overall score
        scores = {
            'technical': technical_score,
            'risk': risk_score,
            'market': market_score,
            'confidence': confidence_score
        }
        overall_score = self.calculate_overall_score(scores)
        
        # Generate reasons and recommendations
        if trend_score > 0.7:
            reasons.append("Strong trend alignment")
        if level_score > 0.7:
            reasons.append("High-quality support/resistance level")
        if volume_score > 0.7:
            reasons.append("Volume confirmation present")
        if risk_score > 0.7:
            reasons.append("Favorable risk/reward ratio")
        
        if overall_score > 0.8:
            recommendations.append("High-quality setup - consider full position size")
        elif overall_score > 0.6:
            recommendations.append("Good setup - consider reduced position size")
        else:
            recommendations.append("Low-quality setup - consider passing")
        
        return QualityScore(
            setup_id=f"{setup.timestamp}_{setup.setup_type}",
            overall_score=overall_score,
            technical_score=technical_score,
            risk_score=risk_score,
            market_score=market_score,
            confidence_score=confidence_score,
            reasons=reasons,
            recommendations=recommendations,
            metadata={
                'trend_score': trend_score,
                'level_score': level_score,
                'volume_score': volume_score,
                'momentum_score': momentum_score,
                'version': self.version
            }
        )
    
    def _evaluate_trend_alignment(self, setup: SetupData, market_data: pd.DataFrame, parameters: Dict[str, Any]) -> float:
        """Evaluate how well the setup aligns with the trend."""
        try:
            # Find setup position in data
            setup_idx = market_data.index.get_loc(setup.timestamp, method='nearest')
            
            # Calculate trend using different timeframes
            short_trend = self._calculate_trend(market_data.iloc[max(0, setup_idx-20):setup_idx+1]['close'])
            medium_trend = self._calculate_trend(market_data.iloc[max(0, setup_idx-50):setup_idx+1]['close'])
            
            # Check alignment
            if setup.setup_type.endswith('_long'):
                # For long setups, want upward trends
                score = (max(0, short_trend) + max(0, medium_trend)) / 2
            else:
                # For short setups, want downward trends
                score = (max(0, -short_trend) + max(0, -medium_trend)) / 2
            
            return min(1.0, score)
        except:
            return 0.5  # Neutral if can't calculate
    
    def _evaluate_level_strength(self, setup: SetupData, market_data: pd.DataFrame, parameters: Dict[str, Any]) -> float:
        """Evaluate the strength of the price level."""
        try:
            # Look at how many times price has respected this level
            level_price = setup.entry_price
            tolerance = parameters.get('level_tolerance', 0.01)
            
            # Count touches near this level
            touches = 0
            for price in market_data['close']:
                if abs(price - level_price) / level_price <= tolerance:
                    touches += 1
            
            # More touches = stronger level
            score = min(1.0, touches / 5)  # Normalize to 5 touches = perfect score
            return score
        except:
            return 0.5
    
    def _evaluate_volume_confirmation(self, setup: SetupData, market_data: pd.DataFrame, parameters: Dict[str, Any]) -> float:
        """Evaluate volume confirmation for the setup."""
        try:
            setup_idx = market_data.index.get_loc(setup.timestamp, method='nearest')
            
            # Get recent volume data
            recent_volume = market_data.iloc[max(0, setup_idx-10):setup_idx+1]['volume']
            avg_volume = recent_volume.mean()
            current_volume = market_data.iloc[setup_idx]['volume']
            
            # Higher than average volume is good confirmation
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            score = min(1.0, volume_ratio / 2)  # 2x average volume = perfect score
            
            return score
        except:
            return 0.5
    
    def _evaluate_momentum(self, setup: SetupData, market_data: pd.DataFrame, parameters: Dict[str, Any]) -> float:
        """Evaluate price momentum at setup time."""
        try:
            setup_idx = market_data.index.get_loc(setup.timestamp, method='nearest')
            
            # Calculate short-term momentum
            recent_prices = market_data.iloc[max(0, setup_idx-5):setup_idx+1]['close']
            momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
            
            # For long setups, positive momentum is good
            if setup.setup_type.endswith('_long'):
                score = max(0, momentum * 10)  # Scale momentum
            else:
                score = max(0, -momentum * 10)
            
            return min(1.0, score)
        except:
            return 0.5
    
    def _evaluate_risk_metrics(self, setup: SetupData, parameters: Dict[str, Any]) -> float:
        """Evaluate risk-related metrics."""
        scores = []
        
        # Risk/reward ratio
        if setup.risk_reward_ratio and setup.risk_reward_ratio > 0:
            rr_score = min(1.0, setup.risk_reward_ratio / 3)  # 3:1 RR = perfect score
            scores.append(rr_score)
        
        # Stop loss distance (closer stops are riskier but more precise)
        if setup.stop_loss:
            stop_distance = abs(setup.entry_price - setup.stop_loss) / setup.entry_price
            distance_score = 1.0 - min(1.0, stop_distance / 0.05)  # 5% stop = neutral
            scores.append(distance_score)
        
        return np.mean(scores) if scores else 0.5
    
    def _evaluate_market_conditions(self, setup: SetupData, market_data: pd.DataFrame, parameters: Dict[str, Any]) -> float:
        """Evaluate broader market conditions."""
        try:
            setup_idx = market_data.index.get_loc(setup.timestamp, method='nearest')
            
            # Volatility assessment
            recent_prices = market_data.iloc[max(0, setup_idx-20):setup_idx+1]['close']
            volatility = recent_prices.pct_change().std()
            
            # Moderate volatility is preferred (not too low, not too high)
            optimal_vol = 0.02  # 2% daily volatility
            vol_score = 1.0 - abs(volatility - optimal_vol) / optimal_vol
            vol_score = max(0, min(1, vol_score))
            
            return vol_score
        except:
            return 0.5
    
    def _calculate_trend(self, prices: pd.Series) -> float:
        """Calculate trend strength (-1 to 1)."""
        if len(prices) < 2:
            return 0
        
        # Linear regression slope
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        
        # Normalize by average price
        trend_strength = slope / prices.mean() if prices.mean() > 0 else 0
        return np.clip(trend_strength * 100, -1, 1)  # Scale and clip
    
    def get_parameter_ranges(self) -> Dict[str, tuple]:
        """Get parameter ranges for optimization."""
        return {
            'level_tolerance': (0.005, 0.02),
            'volume_threshold': (1.2, 3.0),
            'momentum_period': (3, 10)
        }