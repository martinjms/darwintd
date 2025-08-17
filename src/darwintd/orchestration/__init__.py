"""
Orchestration Module for DarwinTD.

This module contains engines for orchestrating the complete trading pipeline:
setup detection → quality evaluation → trade execution → backtesting

Available Orchestration Engines:
- Pipeline: Coordinates the full trading pipeline
- Backtesting: Runs comprehensive backtests across all combinations
- Optimization: Genetic algorithm optimization of all parameters
- Live: Real-time trading orchestration

The orchestration layer allows testing different combinations of:
- Setup detection engines (Fibonacci V1/V2, Support/Resistance, Volume)  
- Quality evaluation engines (Technical V1, Statistical, Risk)
- Trade execution engines (Scalping V1, Swing V1, Adaptive)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np

from ..setup_detection import BaseSetupDetector, SetupData
from ..quality_evaluation import BaseQualityEvaluator, QualityScore
from ..trade_execution import BaseTradeExecutor, TradeExecution
from ..backtesting.vectorbt_engine import VectorBTEngine, PortfolioResults

@dataclass
class PipelineConfig:
    """Configuration for trading pipeline orchestration."""
    setup_detector: str  # "fibonacci_v1", "fibonacci_v2", etc.
    quality_evaluator: str  # "technical_v1", "statistical_v1", etc.
    trade_executor: str  # "scalping_v1", "swing_v1", etc.
    setup_parameters: Dict[str, Any] = field(default_factory=dict)
    quality_parameters: Dict[str, Any] = field(default_factory=dict)
    execution_parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineResults:
    """Results from running a complete trading pipeline."""
    config: PipelineConfig
    setups_detected: List[SetupData]
    quality_scores: List[QualityScore]
    trades_executed: List[TradeExecution]
    portfolio_results: Optional[PortfolioResults]
    performance_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

class BasePipelineOrchestrator(ABC):
    """Abstract base class for pipeline orchestration."""
    
    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self.available_detectors = {}
        self.available_evaluators = {}
        self.available_executors = {}
        self._register_engines()
    
    @abstractmethod
    def _register_engines(self):
        """Register available engines for this orchestrator."""
        pass
    
    @abstractmethod
    def run_pipeline(self, data: pd.DataFrame, config: PipelineConfig) -> PipelineResults:
        """Run the complete trading pipeline."""
        pass
    
    def register_detector(self, name: str, detector_class):
        """Register a setup detector engine."""
        self.available_detectors[name] = detector_class
    
    def register_evaluator(self, name: str, evaluator_class):
        """Register a quality evaluator engine."""
        self.available_evaluators[name] = evaluator_class
    
    def register_executor(self, name: str, executor_class):
        """Register a trade executor engine."""
        self.available_executors[name] = executor_class
    
    def get_available_configs(self) -> List[PipelineConfig]:
        """Get all possible pipeline configurations."""
        configs = []
        
        for detector_name in self.available_detectors.keys():
            for evaluator_name in self.available_evaluators.keys():
                for executor_name in self.available_executors.keys():
                    config = PipelineConfig(
                        setup_detector=detector_name,
                        quality_evaluator=evaluator_name,
                        trade_executor=executor_name
                    )
                    configs.append(config)
        
        return configs