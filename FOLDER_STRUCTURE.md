# DarwinTD Modular Architecture

## Super Modular Folder Structure

DarwinTD implements a revolutionary modular architecture that separates setup detection from trading execution, allowing independent optimization of each component.

```
src/darwintd/
├── setup_detection/                    # 🎯 SETUP DETECTION ENGINES
│   ├── __init__.py                    # Base classes and StandardizedSetup data
│   ├── fibonacci/                     # Fibonacci retracement/extension setups
│   │   ├── __init__.py               # Fibonacci engine registry
│   │   ├── base.py                   # Common Fibonacci calculations
│   │   ├── v1_basic.py               # ✅ Basic Fibonacci detection
│   │   └── v2_advanced.py            # ✅ Multi-timeframe + volume confluence
│   ├── support_resistance/            # Horizontal level breakout setups
│   │   ├── __init__.py
│   │   ├── v1_basic.py               # 🔄 Planned: Basic S/R detection
│   │   └── v2_dynamic.py             # 🔄 Planned: Dynamic level detection
│   ├── volume/                        # Volume-based setup detection
│   │   ├── __init__.py
│   │   ├── v1_profile.py             # 🔄 Planned: Volume profile setups
│   │   └── v2_anomaly.py             # 🔄 Planned: Volume anomaly detection
│   └── confluence/                    # Combined multi-factor setups
│       ├── __init__.py
│       ├── v1_basic.py               # 🔄 Planned: Basic confluence
│       └── v2_weighted.py            # 🔄 Planned: Weighted confluence
│
├── quality_evaluation/                 # 📊 QUALITY ASSESSMENT ENGINES
│   ├── __init__.py                    # Base classes and QualityScore data
│   ├── technical/                     # Technical analysis evaluation
│   │   ├── __init__.py
│   │   ├── v1_basic.py               # ✅ Comprehensive technical evaluation
│   │   └── v2_advanced.py            # 🔄 Planned: Multi-timeframe technical
│   ├── statistical/                   # Statistical quality metrics
│   │   ├── __init__.py
│   │   ├── v1_historical.py          # 🔄 Planned: Historical performance
│   │   └── v2_bayesian.py            # 🔄 Planned: Bayesian probability
│   ├── risk/                         # Risk-focused evaluation
│   │   ├── __init__.py
│   │   ├── v1_basic.py               # 🔄 Planned: Basic risk assessment
│   │   └── v2_portfolio.py           # 🔄 Planned: Portfolio risk context
│   └── market/                       # Market condition assessment
│       ├── __init__.py
│       ├── v1_volatility.py          # 🔄 Planned: Volatility regime analysis
│       └── v2_correlation.py         # 🔄 Planned: Cross-asset correlation
│
├── trade_execution/                   # ⚡ TRADE EXECUTION ENGINES
│   ├── __init__.py                   # Base classes and TradeExecution data
│   ├── scalping/                     # Fast execution with tight management
│   │   ├── __init__.py
│   │   ├── v1_basic.py               # ✅ Basic scalping execution
│   │   └── v2_adaptive.py            # 🔄 Planned: Adaptive scalping
│   ├── swing/                        # Patient execution with wider targets
│   │   ├── __init__.py
│   │   ├── v1_basic.py               # ✅ Basic swing execution
│   │   └── v2_trend_following.py     # 🔄 Planned: Trend-following swing
│   ├── adaptive/                     # Market-condition-based execution
│   │   ├── __init__.py
│   │   ├── v1_volatility.py          # 🔄 Planned: Volatility-adaptive
│   │   └── v2_regime.py              # 🔄 Planned: Market regime adaptive
│   └── risk_managed/                 # Conservative execution strategies
│       ├── __init__.py
│       ├── v1_conservative.py        # 🔄 Planned: Conservative execution
│       └── v2_defensive.py           # 🔄 Planned: Defensive execution
│
├── orchestration/                     # 🎼 PIPELINE COORDINATION
│   ├── __init__.py                   # Base orchestration classes
│   ├── pipeline/                     # Complete trading pipeline
│   │   ├── __init__.py
│   │   ├── v1_basic.py               # ✅ Basic pipeline orchestration
│   │   └── v2_parallel.py            # 🔄 Planned: Parallel processing
│   ├── backtesting/                  # Bulk testing infrastructure
│   │   ├── __init__.py
│   │   ├── bulk_tester.py            # ✅ Test all engine combinations
│   │   └── performance_analyzer.py   # 🔄 Planned: Advanced analytics
│   └── optimization/                 # Genetic algorithm optimization
│       ├── __init__.py
│       ├── genetic_optimizer.py      # 🔄 Planned: GA parameter optimization
│       └── multi_objective.py        # 🔄 Planned: Multi-objective optimization
│
├── backtesting/                      # 🏃 HIGH-PERFORMANCE BACKTESTING
│   ├── __init__.py
│   └── vectorbt_engine.py            # ✅ VectorBT integration (103+ backtests/sec)
│
└── signals/                          # 📈 LEGACY SIGNAL GENERATION
    ├── __init__.py
    └── base.py                       # ✅ Base signal classes (legacy)
```

## Key Design Principles

### 1. **Separation of Concerns**
- **Setup Detection**: "I see a pattern" (Fibonacci retracement at support)
- **Quality Evaluation**: "This looks good" (confluence analysis)
- **Trade Execution**: "How should I trade this?" (scalping vs swing)

### 2. **Version Control for Algorithms**
- Multiple implementations of each engine type (v1_basic, v2_advanced)
- Independent optimization and A/B testing
- Evolutionary improvement through versioning

### 3. **Bulk Testing Capability**
```python
# Test ALL combinations systematically
fibonacci_v1 + technical_v1 + scalping_v1
fibonacci_v1 + technical_v1 + swing_v1  
fibonacci_v2 + technical_v1 + scalping_v1
# ... all possible combinations
```

### 4. **Professional Trading Workflow**
Mirrors how institutional traders work:
1. Pattern recognition specialists find setups
2. Risk managers evaluate quality
3. Execution traders implement the trades
4. Performance analysts optimize the system

### 5. **Independent Evolution**
Each layer evolves separately:
- Setup detection optimizes for pattern accuracy
- Quality evaluation optimizes for risk assessment
- Trade execution optimizes for profit/risk management
- Orchestration optimizes for overall system performance

## Current Implementation Status

✅ **Completed**:
- Modular architecture foundation
- Fibonacci detection engines (v1, v2)
- Technical quality evaluation (v1)
- Scalping & swing execution engines (v1)
- Pipeline orchestration (v1)
- Bulk testing infrastructure
- VectorBT integration with 103+ backtests/second

🔄 **Next Phase**:
- Support/resistance detection engines
- Statistical quality evaluation
- Adaptive execution engines
- Genetic algorithm optimization layer

This modular architecture enables systematic testing of all engine combinations to find optimal configurations for different market conditions and trading objectives.