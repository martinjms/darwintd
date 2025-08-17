# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DarwinTD is an evolutionary trading system that uses genetic algorithms to develop optimized cryptocurrency trading strategies. The system focuses on price action and volume analysis rather than traditional indicators.

### Core Concept
- Each cryptocurrency develops its own trading strategies through evolutionary algorithms
- Uses price action patterns (Fibonacci, support/resistance, volume patterns)
- Implements comprehensive safety systems including distribution shift detection and consecutive loss protection
- Validates strategies through paper trading before live deployment

### Target Assets
- Primary focus: 4-5 major cryptocurrencies (BTC, ETH, SOL, ADA, LINK)
- MVP scope covers these assets with potential for scaling

## Development Status

**Current Phase**: Foundation and library research completed. Ready to begin implementation.

**Completed**: 
- Technology stack research and selection
- Project structure and development environment setup
- GitHub issues created for all implementation phases

## Technology Stack (Final Decisions)

### **Core Architecture Decision: No Traditional Indicators**
**Important**: This project focuses on **price action and volume patterns only**. We deliberately avoid traditional technical indicators (moving averages, RSI, MACD, etc.) in favor of pure price action analysis.

### **Selected Technology Stack**

#### **1. VectorBT - High-Performance Backtesting Engine**
**Decision**: Use VectorBT as the primary backtesting framework instead of Jesse Framework or traditional libraries.

**Rationale**:
- **Performance**: 10x+ faster than traditional backtesting for parameter optimization
- **Evolutionary Algorithm Synergy**: Built for hyperparameter optimization and genetic algorithms
- **Cost Effectiveness**: Faster optimization = lower cloud computing costs = higher net profitability
- **Multi-Objective Optimization**: Native support for Pareto frontier evolution
- **Parameter Grid Testing**: Can test thousands of parameter combinations simultaneously

**Trade-off**: Steeper learning curve, but essential for evolutionary trading system performance

#### **2. Custom Price Action Implementation**
**Decision**: Build custom price action algorithms instead of using pandas-ta or TA-Lib.

**Rationale**:
- **Alignment**: Traditional TA libraries are indicator-heavy, we need pure price action
- **Performance**: Custom implementations optimized for our specific use case
- **Transparency**: Full understanding of every calculation for genetic optimization
- **Flexibility**: Easy to modify for evolutionary algorithm requirements

**Core Components**:
```python
# What we build ourselves:
- Fibonacci retracements/extensions calculation
- Support/resistance level detection
- Volume profile and Point of Control analysis
- Swing point identification algorithms
- Breakout detection logic
```

#### **3. CCXT - Exchange Integration**
**Decision**: Use CCXT for all cryptocurrency exchange interactions.

**Rationale**: Industry standard, supports 100+ exchanges, unified API, active maintenance.

#### **4. plotly - Interactive Visualization**
**Decision**: Use plotly for all charting and dashboard needs.

**Rationale**: Interactive charts essential for strategy validation, real-time updates, web-based dashboards.

#### **5. Supporting Libraries**
- **numpy/pandas**: Core data manipulation
- **scipy**: Advanced mathematical functions
- **DEAP**: Genetic algorithm framework
- **PostgreSQL/SQLite**: Data storage (SQLite for development, PostgreSQL for production)

### **Architecture Philosophy**

#### **Performance-First Design**
Every component chosen for computational efficiency to support:
- Real-time genetic algorithm optimization
- Large-scale parameter space exploration
- Multi-asset portfolio optimization
- Continuous strategy evolution

#### **Modular Evolution-Ready Design**
```
Data Pipeline → Price Action Analysis → Signal Generation → VectorBT Backtesting → Genetic Algorithm Optimization
```

Each component designed for:
- Independent optimization and replacement
- Genetic algorithm parameter evolution
- Real-time adaptation to market conditions

## Implementation Phases (GitHub Issues - REVISED ARCHITECTURE)

### **Core Architecture: Setup Detection vs Trading Execution**

**Philosophy**: Separate pattern recognition from trade execution, mirroring how professional traders work:
1. **"I see a setup"** (pattern detection)
2. **"This looks good"** (quality assessment)  
3. **"How should I trade this?"** (execution optimization)

### **Phase 1: Setup Detection Engine (4-6 weeks)**
- **Issue #7**: VectorBT Core Integration ✅ **COMPLETED**
- **Issue #13**: Fibonacci Setup Detection Engine
- **Issue #14**: Support/Resistance Level Detection
- **Issue #15**: Volume Profile and POC Analysis
- **Issue #16**: Multi-Timeframe Confluence Analysis
- **Goal**: Detect high-probability trading setups with quality scoring

### **Phase 2: Setup Management System (3-4 weeks)**
- **Issue #17**: Setup Database and Storage System
- **Issue #18**: Setup Visualization and Review Interface
- **Issue #19**: Setup Quality Scoring and Filtering
- **Goal**: Store, review, and validate detected setups

### **Phase 3: Trading Execution Optimization (4-5 weeks)**
- **Issue #20**: Setup Trading Execution Engine
- **Issue #21**: Entry/Exit Timing Optimization
- **Issue #22**: Scalping-to-Swing Trading Logic
- **Goal**: Optimize how to trade validated setups

### **Phase 4: Evolutionary Optimization (6-8 weeks)**
- **Issue #11**: Advanced Genetic Algorithm Engine (adapted)
- **Issue #23**: Setup Detection Parameter Evolution
- **Issue #24**: Trading Execution Parameter Evolution
- **Goal**: Evolve both setup detection and trading execution

## New Architecture Details

### **Setup Detection Workflow**
```
OHLCV Data → Level Detection → Setup Pattern Recognition → Quality Scoring → Setup Database
```

**Example: Fibonacci Setup Detection**
1. **Swing Point Detection**: Find significant highs/lows
2. **Fibonacci Level Calculation**: Calculate retracement/extension levels
3. **Setup Trigger**: Price approaches a Fibonacci level (within tolerance)
4. **Confluence Analysis**: Check for additional supporting levels
5. **Quality Scoring**: Rate setup based on confluences and market context

### **Setup Data Structure**
```python
@dataclass
class TradingSetup:
    timestamp: datetime           # When setup was detected
    symbol: str                  # Asset symbol
    setup_type: str              # 'fib_retracement', 'support_break', etc.
    quality_score: float         # 0.0 to 10.0 
    confluences: List[str]       # ['fib_618', 'weekly_support', 'volume_poc']
    price_at_detection: float    # Price when setup detected
    relevant_levels: Dict        # Key price levels for this setup
    market_context: Dict         # Trend, volume, volatility context
    chart_window: pd.DataFrame   # OHLCV data for visualization
```

### **Trading Execution Workflow**
```
Setup Detection → Entry Timing → Position Management → Exit Strategy → Performance Attribution
```

**Trading Parameters to Optimize:**
- Entry timing (immediate, confirmation wait, scale-in)
- Entry precision (distance from setup trigger)
- Position sizing (within 1% risk limit)
- Stop loss placement
- Take profit targets (multiple levels)
- Scalping-to-swing conversion logic
- Maximum hold time

### **Confluence System**
Multiple detection methods that can reinforce each other:
- **Fibonacci Levels**: 23.6%, 38.2%, 50%, 61.8%, 78.6%
- **Support/Resistance**: Historical price levels
- **Volume Levels**: Point of Control, Value Area
- **Weekly/Monthly Levels**: Higher timeframe pivots
- **Round Numbers**: Psychological levels

**Quality Scoring**: More confluences = higher setup quality

### **Visual Review System**
Generate chart interface showing:
- Detected setup location
- All relevant levels and confluences
- Market context (volume, trend)
- Setup quality score
- Quick navigation through setup list

### **Supporting Infrastructure**
- **Issue #1**: Research and Choose Trading Libraries ✅ **COMPLETED**
- **Issue #2**: Set Up Development Environment
- **Issue #3**: Build Basic Visualization System
- **Issue #4**: Create Unit Testing Framework
- **Issue #5**: Create OHLCV Data Pipeline
- **Issue #6**: Add Data Validation Dashboard

## Development Principles

### **1. Performance-First Implementation**
- Every component optimized for genetic algorithm requirements
- VectorBT's vectorized operations for massive parameter space exploration
- Minimal computational overhead for real-time evolution

### **2. Evolution-Ready Architecture**
- All parameters designed for genetic optimization
- Modular components that can be evolved independently
- Clear fitness function definitions for every strategy component

### **3. No Premature Optimization Philosophy**
- Start with VectorBT (high-performance foundation)
- Custom implementations only where needed
- Learning curve investment pays off in long-term performance

### **4. Validation Strategy**
- Mathematical validation against manual calculations
- Visual verification using plotly dashboards
- Cross-validation across multiple market conditions
- Out-of-sample testing for all evolved strategies

## Timeline and Expectations

**Total Development Time**: 18-24 weeks for complete system
**Approach**: Build for the full version, not MVP
**Rationale**: Computing cost vs profit optimization requires high-performance foundation from day one

### **Phase Completion Criteria**
- **Phase 1**: Can backtest 1000+ parameter combinations in <30 seconds
- **Phase 2**: Price action strategies outperform random parameter selection
- **Phase 3**: Genetic algorithm produces consistently improving strategy performance

## Key Decision Rationale

### **Why VectorBT Over Simpler Solutions?**
1. **Cost-Benefit Analysis**: Higher cloud computing costs offset by significantly better strategy optimization
2. **Scalability**: System designed to handle institutional-scale parameter optimization
3. **Future-Proof**: No migration needed as system complexity grows
4. **Research Quality**: Professional-grade backtesting for strategy validation

### **Why Custom Price Action vs TA Libraries?**
1. **Alignment**: Project specifically avoids traditional indicators
2. **Optimization**: Genetic algorithms need transparent, modifiable calculations
3. **Performance**: Custom implementations optimized for our specific use cases
4. **Innovation**: Enables novel price action pattern discovery through evolution

## Commands for Development

### **Testing and Validation**
```bash
# Test library integration
python test_library_integration.py

# Run unit tests
pytest tests/unit/

# Run performance benchmarks  
python scripts/benchmark_vectorbt.py

# Validate price action calculations
python scripts/validate_price_action.py
```

### **Development Workflow**
```bash
# Install dependencies
pip install -e ".[dev,analysis]"

# Run code quality checks
black src/ tests/
flake8 src/ tests/
mypy src/

# Start development server (when implemented)
python -m darwintd.server
```