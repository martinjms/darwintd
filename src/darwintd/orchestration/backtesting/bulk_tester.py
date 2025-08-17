"""
Bulk Backtesting Engine for DarwinTD.

Tests all combinations of detection, evaluation, and execution engines to find
the optimal configuration for different market conditions.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

from .. import PipelineConfig, PipelineResults
from ..pipeline.v1_basic import PipelineOrchestratorV1


class BulkBacktestEngine:
    """Engine for running bulk backtests across all pipeline combinations."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.orchestrator = PipelineOrchestratorV1()
    
    def run_comprehensive_backtest(
        self, 
        data: pd.DataFrame, 
        parameter_ranges: Optional[Dict[str, Dict[str, tuple]]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive backtesting across all engine combinations.
        
        Args:
            data: OHLCV market data
            parameter_ranges: Optional parameter ranges for each engine type
            
        Returns:
            Dictionary containing all results and analysis
        """
        print("Starting comprehensive backtest...")
        print(f"Data period: {data.index[0]} to {data.index[-1]}")
        print(f"Total bars: {len(data)}")
        
        # Generate all pipeline configurations
        base_configs = self.orchestrator.get_available_configs()
        
        # Expand with parameter variations if provided
        if parameter_ranges:
            expanded_configs = self._expand_configs_with_parameters(base_configs, parameter_ranges)
        else:
            expanded_configs = base_configs
        
        print(f"Testing {len(expanded_configs)} pipeline configurations...")
        
        # Run backtests
        all_results = []
        
        if self.max_workers > 1:
            # Parallel execution
            all_results = self._run_parallel_backtests(data, expanded_configs)
        else:
            # Sequential execution
            all_results = self._run_sequential_backtests(data, expanded_configs)
        
        # Analyze results
        analysis = self._analyze_results(all_results)
        
        return {
            'all_results': all_results,
            'analysis': analysis,
            'metadata': {
                'total_configs_tested': len(expanded_configs),
                'successful_runs': len([r for r in all_results if r.metadata.get('pipeline_status') == 'completed']),
                'data_period': f"{data.index[0]} to {data.index[-1]}",
                'data_bars': len(data)
            }
        }
    
    def _expand_configs_with_parameters(self, base_configs: List[PipelineConfig], parameter_ranges: Dict[str, Dict[str, tuple]]) -> List[PipelineConfig]:
        """Expand base configurations with parameter variations."""
        expanded_configs = []
        
        for base_config in base_configs:
            # Get parameter ranges for this configuration
            setup_ranges = parameter_ranges.get('setup_detection', {}).get(base_config.setup_detector, {})
            quality_ranges = parameter_ranges.get('quality_evaluation', {}).get(base_config.quality_evaluator, {})
            execution_ranges = parameter_ranges.get('trade_execution', {}).get(base_config.trade_executor, {})
            
            # Generate parameter combinations (limited to avoid explosion)
            setup_params = self._generate_param_combinations(setup_ranges, max_combinations=3)
            quality_params = self._generate_param_combinations(quality_ranges, max_combinations=3)
            execution_params = self._generate_param_combinations(execution_ranges, max_combinations=3)
            
            # Create configs for all combinations
            for s_params, q_params, e_params in itertools.product(setup_params, quality_params, execution_params):
                config = PipelineConfig(
                    setup_detector=base_config.setup_detector,
                    quality_evaluator=base_config.quality_evaluator,
                    trade_executor=base_config.trade_executor,
                    setup_parameters=s_params,
                    quality_parameters=q_params,
                    execution_parameters=e_params
                )
                expanded_configs.append(config)
        
        return expanded_configs
    
    def _generate_param_combinations(self, param_ranges: Dict[str, tuple], max_combinations: int = 3) -> List[Dict[str, Any]]:
        """Generate parameter combinations from ranges."""
        if not param_ranges:
            return [{}]  # Empty parameters
        
        param_sets = []
        param_names = list(param_ranges.keys())
        
        # Generate limited combinations to avoid exponential explosion
        for _ in range(max_combinations):
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, bool):
                    params[param_name] = np.random.choice([True, False])
                elif isinstance(min_val, int):
                    params[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    params[param_name] = np.random.uniform(min_val, max_val)
            param_sets.append(params)
        
        return param_sets
    
    def _run_parallel_backtests(self, data: pd.DataFrame, configs: List[PipelineConfig]) -> List[PipelineResults]:
        """Run backtests in parallel."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_config = {
                executor.submit(self._run_single_backtest, data, config): config 
                for config in configs
            }
            
            # Collect results
            for i, future in enumerate(as_completed(future_to_config)):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Completed {i+1}/{len(configs)}: {config.setup_detector}+{config.quality_evaluator}+{config.trade_executor}")
                except Exception as e:
                    print(f"Failed {i+1}/{len(configs)}: {e}")
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
    
    def _run_sequential_backtests(self, data: pd.DataFrame, configs: List[PipelineConfig]) -> List[PipelineResults]:
        """Run backtests sequentially."""
        results = []
        
        for i, config in enumerate(configs):
            print(f"Running {i+1}/{len(configs)}: {config.setup_detector}+{config.quality_evaluator}+{config.trade_executor}")
            
            try:
                result = self._run_single_backtest(data, config)
                results.append(result)
            except Exception as e:
                print(f"Failed: {e}")
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
    
    def _run_single_backtest(self, data: pd.DataFrame, config: PipelineConfig) -> PipelineResults:
        """Run a single backtest configuration."""
        orchestrator = PipelineOrchestratorV1()  # Create fresh instance for each run
        return orchestrator.run_pipeline(data, config)
    
    def _analyze_results(self, results: List[PipelineResults]) -> Dict[str, Any]:
        """Analyze backtest results to find best configurations."""
        successful_results = [r for r in results if r.metadata.get('pipeline_status') == 'completed' and r.performance_metrics]
        
        if not successful_results:
            return {'error': 'No successful backtests completed'}
        
        # Performance analysis
        performance_df = pd.DataFrame([r.performance_metrics for r in successful_results])
        
        # Add configuration info
        for i, result in enumerate(successful_results):
            performance_df.loc[i, 'setup_detector'] = result.config.setup_detector
            performance_df.loc[i, 'quality_evaluator'] = result.config.quality_evaluator
            performance_df.loc[i, 'trade_executor'] = result.config.trade_executor
        
        # Find best configurations by different metrics
        best_configs = {}
        
        if 'total_pnl' in performance_df.columns:
            best_configs['highest_profit'] = self._get_best_config(performance_df, 'total_pnl', successful_results)
        
        if 'sharpe_ratio' in performance_df.columns:
            best_configs['best_sharpe'] = self._get_best_config(performance_df, 'sharpe_ratio', successful_results)
        
        if 'win_rate' in performance_df.columns:
            best_configs['best_winrate'] = self._get_best_config(performance_df, 'win_rate', successful_results)
        
        if 'max_drawdown' in performance_df.columns:
            best_configs['lowest_drawdown'] = self._get_best_config(performance_df, 'max_drawdown', successful_results, ascending=True)
        
        # Engine performance analysis
        engine_analysis = self._analyze_engine_performance(performance_df)
        
        # Statistical summary
        stats_summary = {
            'mean_performance': performance_df.mean().to_dict() if not performance_df.empty else {},
            'std_performance': performance_df.std().to_dict() if not performance_df.empty else {},
            'best_performance': performance_df.max().to_dict() if not performance_df.empty else {},
            'worst_performance': performance_df.min().to_dict() if not performance_df.empty else {}
        }
        
        return {
            'best_configurations': best_configs,
            'engine_analysis': engine_analysis,
            'statistical_summary': stats_summary,
            'performance_data': performance_df.to_dict('records') if not performance_df.empty else [],
            'total_successful_runs': len(successful_results)
        }
    
    def _get_best_config(self, performance_df: pd.DataFrame, metric: str, results: List[PipelineResults], ascending: bool = False) -> Dict[str, Any]:
        """Get the best configuration for a specific metric."""
        if metric not in performance_df.columns:
            return {}
        
        best_idx = performance_df[metric].idxmin() if ascending else performance_df[metric].idxmax()
        best_result = results[best_idx]
        
        return {
            'config': {
                'setup_detector': best_result.config.setup_detector,
                'quality_evaluator': best_result.config.quality_evaluator,
                'trade_executor': best_result.config.trade_executor,
                'parameters': {
                    'setup': best_result.config.setup_parameters,
                    'quality': best_result.config.quality_parameters,
                    'execution': best_result.config.execution_parameters
                }
            },
            'performance': best_result.performance_metrics,
            'metric_value': performance_df.loc[best_idx, metric]
        }
    
    def _analyze_engine_performance(self, performance_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by engine type."""
        if performance_df.empty:
            return {}
        
        analysis = {}
        
        # Analyze by setup detector
        if 'setup_detector' in performance_df.columns:
            setup_analysis = performance_df.groupby('setup_detector').agg({
                'total_pnl': ['mean', 'std', 'count'],
                'sharpe_ratio': ['mean', 'std'],
                'win_rate': ['mean', 'std']
            }).round(4)
            analysis['setup_detectors'] = setup_analysis.to_dict()
        
        # Analyze by quality evaluator
        if 'quality_evaluator' in performance_df.columns:
            quality_analysis = performance_df.groupby('quality_evaluator').agg({
                'total_pnl': ['mean', 'std', 'count'],
                'sharpe_ratio': ['mean', 'std'],
                'win_rate': ['mean', 'std']
            }).round(4)
            analysis['quality_evaluators'] = quality_analysis.to_dict()
        
        # Analyze by trade executor
        if 'trade_executor' in performance_df.columns:
            executor_analysis = performance_df.groupby('trade_executor').agg({
                'total_pnl': ['mean', 'std', 'count'],
                'sharpe_ratio': ['mean', 'std'],
                'win_rate': ['mean', 'std']
            }).round(4)
            analysis['trade_executors'] = executor_analysis.to_dict()
        
        return analysis