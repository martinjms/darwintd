# DarwinTD Development Setup

## Quick Start

1. **Clone and navigate to the repository:**
   ```bash
   git clone https://github.com/martinjms/darwintd.git
   cd darwintd
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e ".[dev,analysis]"
   ```

4. **Configure environment:**
   ```bash
   cp .env.template .env
   # Edit .env with your API keys and settings
   ```

5. **Test installation:**
   ```bash
   python test_library_integration.py
   ```

## System Requirements

- Python 3.9 or higher
- 4GB+ RAM (for data processing)
- 10GB+ disk space (for historical data)
- Internet connection (for data collection)

## Dependencies

### Core Dependencies
- **pandas/numpy**: Data manipulation and numerical computing
- **ccxt**: Cryptocurrency exchange APIs
- **pandas-ta**: Technical analysis indicators
- **plotly**: Interactive visualization
- **pydantic**: Configuration and data validation

### Optional Dependencies
- **TA-Lib**: High-performance technical analysis (requires C dependencies)
- **Jesse Framework**: Complete trading platform
- **vectorbt**: Advanced backtesting (post-MVP)

## Installation Options

### Option 1: Basic Setup (Recommended for beginners)
```bash
pip install -r requirements.txt
```

### Option 2: Development Setup (Full features)
```bash
pip install -e ".[dev,analysis]"
```

### Option 3: With TA-Lib (Advanced users)
```bash
# Install system dependencies first
# Ubuntu/Debian:
sudo apt-get install build-essential libssl-dev libffi-dev python3-dev

# macOS:
brew install ta-lib

# Then install Python packages
pip install TA-Lib
pip install -e ".[dev,analysis]"
```

## Configuration

### Environment Variables
Copy `.env.template` to `.env` and configure:

**Required for MVP:**
- `DATABASE_URL`: Database connection string
- `DATA_DIR`: Directory for storing market data

**Optional but recommended:**
- Exchange API keys for live data collection
- `DEBUG=True` for development
- `PAPER_TRADING=True` for safe testing

### Database Setup

**SQLite (Default - no setup required):**
```bash
DATABASE_URL=sqlite:///./darwintd.db
```

**PostgreSQL (Production):**
```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib  # Ubuntu
# OR
brew install postgresql  # macOS

# Create database
createdb darwintd

# Configure
DATABASE_URL=postgresql://user:password@localhost:5432/darwintd
```

## Development Tools

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Run tests
pytest
```

### Pre-commit Hooks
```bash
pre-commit install
```

## Project Structure

```
darwintd/
├── src/darwintd/           # Main source code
│   ├── data/               # Data collection and storage
│   ├── analysis/           # Technical analysis
│   ├── strategies/         # Trading strategies
│   ├── evolution/          # Genetic algorithms
│   ├── visualization/      # Charts and dashboards
│   ├── utils/              # Utilities
│   └── config/             # Configuration
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── fixtures/           # Test data
├── docs/                   # Documentation
├── scripts/                # Utility scripts
├── notebooks/              # Jupyter notebooks
└── config/                 # Configuration files
```

## Verification

### Test Library Installation
```bash
python test_library_integration.py
```

Expected output:
```
✅ pandas and numpy imported successfully
✅ CCXT imported successfully
✅ pandas-ta imported successfully
✅ plotly imported successfully
```

### Test Exchange Connectivity
```bash
python -c "import ccxt; print(f'CCXT supports {len(ccxt.exchanges)} exchanges')"
```

### Test Data Pipeline
```bash
python -c "
import pandas as pd
import pandas_ta as ta
data = pd.DataFrame({'close': [1, 2, 3, 4, 5]})
print('SMA:', ta.sma(data['close']).iloc[-1])
"
```

## Common Issues

### TA-Lib Installation Issues
```bash
# If TA-Lib fails to install:
pip install --upgrade pip wheel
pip install --no-cache-dir TA-Lib

# Alternative: Use conda
conda install -c conda-forge ta-lib
```

### Exchange API Rate Limits
- Use paper trading mode initially
- Implement proper rate limiting
- Monitor API usage in dashboards

### Memory Issues with Large Datasets
- Use chunked processing for historical data
- Implement data caching strategies
- Consider using Parquet format for storage

## Next Steps

1. **Complete Issue #1**: Finalize library choices based on testing
2. **Start Issue #5**: Begin OHLCV data pipeline implementation
3. **Set up Issue #4**: Create unit testing framework
4. **Implement Issue #3**: Build visualization system

## Support

- Check `CLAUDE.md` for project context
- Review GitHub issues for current development status
- Run integration tests to verify setup
- Use `pytest -v` for detailed test information