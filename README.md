## darwintd

### Project Summary

#### Core Innovation
Evolutionary trading system where each cryptocurrency develops its own optimized strategies through genetic algorithms, using price action and volume data rather than traditional indicators.

#### Goals
- Primary: Create profitable automated trading system for personal use
- Secondary: Validate evolutionary approach works across different market conditions
- Long-term: Potentially scale into educational trading platform

#### MVP Scope
- 4-5 major cryptocurrencies (BTC, ETH, SOL, ADA, LINK)
- Price action strategies (Fibonacci, support/resistance, volume patterns)
- Evolutionary algorithm optimizing strategy parameters
- Comprehensive safety systems (distribution shift detection, consecutive loss protection)
- Paper trading validation before live deployment

#### Post-MVP Roadmap
- Phase 1: Scale to more cryptocurrencies and larger position sizes
- Phase 2: Add AI pattern recognition for complex setups
- Phase 3: Build educational platform features (blind trading, AI mentors, historical challenges)
- Phase 4: Potential SaaS offering for other traders/educators

#### Timeline
8-12 weeks for MVP, 6-12 months for full platform vision.

### TODO Roadmap

### Week 1: Research & Setup + Testing Framework
- Research and choose libraries
- Set up development environment
- Build basic visualization system (matplotlib/plotly for charts)
- Create unit testing framework for level calculations
- Create OHLCV data pipeline
- Add data validation dashboard to spot data issues

#### Trading Libraries for Price Action & Volume Analysis

- **Comprehensive trading frameworks**
  - Jesse Framework - Full algorithmic trading platform with backtesting, live trading, built-in indicators
  - Zipline - Institutional-grade backtesting (Quantopian's engine), good for complex strategies
  - Backtrader - Popular Python backtesting library with extensive documentation
- **Technical analysis libraries**
  - pandas-ta - 150+ indicators, volume profile support, good documentation
  - TA-Lib - Industry standard, C-based (fast), comprehensive technical indicators
  - vectorbt - High-performance backtesting with advanced volume analysis tools
  - Tulip - Lightweight C library with Python bindings, very fast calculations
- **Specialized libraries**
  - ccxt - Cryptocurrency exchange integration (200+ exchanges)
  - yfinance / python-binance - Data fetching from specific sources
  - plotly / mplfinance - Financial charting and visualization
  - numpy / pandas - Core data manipulation (essential foundation)

- **Recommendation**: Start with pandas-ta + Jesse framework combination for comprehensive coverage.

### Week 2: Price Levels + Immediate Verification
- Implement Fibonacci calculations
- Visual test: Plot Fib levels on known chart patterns
- Unit tests: Verify Fib math against manual calculations
- Implement support/resistance detection
- Visual validation: Compare detected levels to TradingView
- Build pivot points with visual verification

### Week 3: Volume Analysis + Validation
- Implement Volume Profile/POC
- Visual test: Compare POC to TradingView volume profile
- Build VWAP calculations
- Unit tests: Verify VWAP against known values
- Create interactive dashboard to explore volume data

### Week 4: Integration + Comprehensive Testing
- Unified level detection system
- End-to-end visual verification on multiple timeframes
- Automated testing suite for all calculations
- Manual validation interface for spot-checking


