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

This is a greenfield project in the planning/research phase. The codebase currently contains only project documentation and roadmap.

## Planned Architecture

### Key Components (Not Yet Implemented)
1. **Data Pipeline**: OHLCV data ingestion with validation dashboard
2. **Technical Analysis Engine**: 
   - Fibonacci calculations
   - Support/resistance detection
   - Volume Profile/POC analysis
   - VWAP calculations
3. **Evolutionary Algorithm**: Genetic algorithm for strategy optimization
4. **Safety Systems**: Risk management and anomaly detection
5. **Visualization System**: Trading charts and performance dashboards

### Recommended Technology Stack
Based on the project README research:

**Core Libraries**:
- pandas-ta + Jesse framework for comprehensive trading analysis
- ccxt for cryptocurrency exchange integration
- plotly/mplfinance for financial visualization
- numpy/pandas for data manipulation

**Alternative Options**:
- TA-Lib for technical indicators
- Backtrader or Zipline for backtesting
- vectorbt for high-performance analysis

## Development Workflow

### Development Phases
1. **Week 1**: Research, setup, testing framework, data pipeline
2. **Week 2**: Price levels implementation with visual verification
3. **Week 3**: Volume analysis with validation against TradingView
4. **Week 4**: System integration and comprehensive testing

### Validation Strategy
- Visual verification against TradingView for all calculations
- Unit tests for mathematical accuracy
- Manual validation interfaces for spot-checking
- Interactive dashboards for data exploration

## Timeline
- MVP target: 8-12 weeks
- Full platform vision: 6-12 months

## Future Development Notes

When implementing this system:
1. Start with robust data validation and visualization before building trading logic
2. Implement comprehensive testing for all mathematical calculations
3. Build safety systems early in the development process
4. Use paper trading extensively before any live trading implementation
5. Focus on visual verification tools to validate technical analysis accuracy