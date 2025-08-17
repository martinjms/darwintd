"""
Pipeline Orchestrator V1 - Basic Implementation.

Coordinates the complete trading pipeline: detection → evaluation → execution.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from .. import BasePipelineOrchestrator, PipelineConfig, PipelineResults
from ...setup_detection.fibonacci.v1_basic import FibonacciDetectorV1
from ...setup_detection.fibonacci.v2_advanced import FibonacciDetectorV2
from ...quality_evaluation.technical.v1_basic import TechnicalQualityV1
from ...trade_execution.scalping.v1_basic import ScalpingExecutorV1
from ...trade_execution.swing.v1_basic import SwingExecutorV1
from ...backtesting.vectorbt_engine import VectorBTEngine, SignalData


class PipelineOrchestratorV1(BasePipelineOrchestrator):
    """Basic pipeline orchestrator for DarwinTD trading system."""
    
    def __init__(self):
        super().__init__("PipelineOrchestratorV1", "1.0")
        self.vectorbt_engine = VectorBTEngine()
    
    def _register_engines(self):
        """Register all available engines."""
        # Setup Detection Engines
        self.register_detector("fibonacci_v1", FibonacciDetectorV1)
        self.register_detector("fibonacci_v2", FibonacciDetectorV2)
        
        # Quality Evaluation Engines
        self.register_evaluator("technical_v1", TechnicalQualityV1)
        
        # Trade Execution Engines
        self.register_executor("scalping_v1", ScalpingExecutorV1)
        self.register_executor("swing_v1", SwingExecutorV1)
    
    def run_pipeline(self, data: pd.DataFrame, config: PipelineConfig) -> PipelineResults:
        """Run the complete trading pipeline."""
        results = PipelineResults(
            config=config,
            setups_detected=[],
            quality_scores=[],
            trades_executed=[],
            portfolio_results=None,
            performance_metrics={}
        )
        
        try:
            # Step 1: Setup Detection
            detector = self._create_detector(config.setup_detector)
            setups = detector.detect_setups(data, config.setup_parameters)
            results.setups_detected = setups
            
            if not setups:
                results.metadata['pipeline_status'] = 'no_setups_detected'
                return results
            
            # Step 2: Quality Evaluation
            evaluator = self._create_evaluator(config.quality_evaluator)
            quality_scores = []
            
            for setup in setups:
                try:
                    quality = evaluator.evaluate_setup(setup, data, config.quality_parameters)
                    quality_scores.append(quality)
                except Exception as e:
                    print(f"Quality evaluation failed for setup {setup.timestamp}: {e}")
                    continue
            
            results.quality_scores = quality_scores
            
            if not quality_scores:
                results.metadata['pipeline_status'] = 'no_quality_scores'
                return results
            
            # Step 3: Trade Execution
            executor = self._create_executor(config.trade_executor)
            trades = []
            
            for setup, quality in zip(setups, quality_scores):
                try:
                    trade = executor.execute_setup(setup, quality, data, config.execution_parameters)
                    if trade:
                        trades.append(trade)
                except Exception as e:
                    print(f"Trade execution failed for setup {setup.timestamp}: {e}")
                    continue
            
            # Simulate trade management throughout the data period
            for i, timestamp in enumerate(data.index[1:], 1):
                closed_trades = executor.manage_open_trades(data, timestamp)
                trades.extend(closed_trades)
            
            # Close any remaining open trades at the end
            for open_trade in executor.open_trades[:]:
                if open_trade.status == 'open':
                    final_price = data['close'].iloc[-1]
                    final_time = data.index[-1]
                    closed_trade = executor._close_trade(open_trade, final_price, final_time, 'end_of_data')
                    trades.append(closed_trade)
            
            results.trades_executed = trades
            
            # Step 4: Convert to VectorBT format and calculate performance
            if trades:
                vectorbt_results = self._convert_to_vectorbt_results(trades, data)
                results.portfolio_results = vectorbt_results
                results.performance_metrics = self._calculate_performance_metrics(trades, data)
            
            results.metadata['pipeline_status'] = 'completed'
            results.metadata['setups_count'] = len(setups)
            results.metadata['quality_scores_count'] = len(quality_scores)
            results.metadata['trades_count'] = len(trades)
            results.metadata['avg_quality_score'] = np.mean([q.overall_score for q in quality_scores]) if quality_scores else 0
            
        except Exception as e:
            results.metadata['pipeline_status'] = 'error'
            results.metadata['error'] = str(e)
            print(f"Pipeline execution failed: {e}")
        
        return results
    
    def _create_detector(self, detector_name: str):
        """Create setup detector instance."""
        if detector_name not in self.available_detectors:
            raise ValueError(f"Unknown detector: {detector_name}")
        return self.available_detectors[detector_name]()
    
    def _create_evaluator(self, evaluator_name: str):
        """Create quality evaluator instance."""
        if evaluator_name not in self.available_evaluators:
            raise ValueError(f"Unknown evaluator: {evaluator_name}")
        return self.available_evaluators[evaluator_name]()
    
    def _create_executor(self, executor_name: str):
        """Create trade executor instance."""
        if executor_name not in self.available_executors:
            raise ValueError(f"Unknown executor: {executor_name}")
        return self.available_executors[executor_name]()
    
    def _convert_to_vectorbt_results(self, trades: List, data: pd.DataFrame):
        """Convert trade results to VectorBT format for analysis."""
        # Create entry/exit signals from trades
        entries = np.zeros(len(data), dtype=bool)
        exits = np.zeros(len(data), dtype=bool)
        
        for trade in trades:
            if trade.status == 'closed':
                try:
                    # Find entry index
                    entry_idx = data.index.get_loc(trade.execution_timestamp, method='nearest')
                    entries[entry_idx] = True
                    
                    # Find exit index
                    if hasattr(trade.metadata, 'exit_time') and trade.metadata.get('exit_time'):
                        exit_idx = data.index.get_loc(trade.metadata['exit_time'], method='nearest')
                        exits[exit_idx] = True
                except (KeyError, ValueError):
                    continue  # Skip trades that can't be mapped to data
        
        # Create SignalData and run backtest
        signal_data = SignalData(
            entries=entries,
            exits=exits,
            prices=data['close']
        )
        
        return self.vectorbt_engine.run_backtest(signal_data)
    
    def _calculate_performance_metrics(self, trades: List, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if not trades:
            return {}
        
        closed_trades = [t for t in trades if t.status == 'closed' and t.pnl is not None]
        
        if not closed_trades:
            return {}
        
        pnls = [t.pnl for t in closed_trades]
        
        # Basic metrics
        total_pnl = sum(pnls)
        total_trades = len(closed_trades)
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')
        
        # Risk metrics
        returns = np.array(pnls) / 100000  # Assuming 100k portfolio
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        max_drawdown = self._calculate_max_drawdown(pnls)
        
        return {
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'return_percent': total_pnl / 100000 * 100  # Assuming 100k portfolio
        }
    
    def _calculate_max_drawdown(self, pnls: List[float]) -> float:
        """Calculate maximum drawdown from PnL series."""
        if not pnls:
            return 0
        
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / 100000  # Normalize by portfolio
        
        return abs(min(drawdown)) if len(drawdown) > 0 else 0
    
    def run_bulk_testing(self, data: pd.DataFrame, configs: List[PipelineConfig]) -> List[PipelineResults]:
        """Run bulk testing across multiple pipeline configurations."""
        results = []
        
        for i, config in enumerate(configs):
            print(f"Running pipeline {i+1}/{len(configs)}: {config.setup_detector} + {config.quality_evaluator} + {config.trade_executor}")
            
            try:
                result = self.run_pipeline(data, config)
                results.append(result)
            except Exception as e:
                print(f"Pipeline {i+1} failed: {e}")
                # Create failed result
                failed_result = PipelineResults(
                    config=config,
                    setups_detected=[],
                    quality_scores=[],
                    trades_executed=[],
                    portfolio_results=None,
                    performance_metrics={},
                    metadata={'pipeline_status': 'failed', 'error': str(e)}
                )
                results.append(failed_result)
        
        return results