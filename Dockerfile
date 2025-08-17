# DarwinTD Development Environment
# Optimized for VectorBT, genetic algorithms, and price action analysis

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools for compiling dependencies
    build-essential \
    gcc \
    g++ \
    # Git for version control
    git \
    # Required for VectorBT/numba
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    # Required for potential future TA-Lib installation
    wget \
    # Required for PostgreSQL client
    libpq-dev \
    # Utilities
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Add user's local bin to PATH (for pip installed packages)
ENV PATH="/home/app/.local/bin:$PATH"

# Copy requirements first for better caching
COPY --chown=app:app requirements.txt pyproject.toml ./

# Install Python dependencies
# Split into multiple RUN commands for better caching and debugging
RUN pip install --user --upgrade pip setuptools wheel

# Install core scientific computing stack first
RUN pip install --user \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    scipy>=1.11.0

# Install numba (VectorBT dependency) separately for better error handling
RUN pip install --user numba>=0.58.0

# Install VectorBT and related packages
RUN pip install --user \
    vectorbt>=0.25.0 \
    deap>=1.4.0

# Install other dependencies
RUN pip install --user \
    ccxt>=4.0.0 \
    plotly>=5.17.0 \
    sqlalchemy>=2.0.0 \
    psycopg2-binary>=2.9.0 \
    python-dotenv>=1.0.0 \
    pydantic>=2.0.0 \
    pydantic-settings>=2.0.0 \
    requests>=2.31.0 \
    aiohttp>=3.9.0 \
    click>=8.1.0 \
    rich>=13.5.0 \
    loguru>=0.7.0

# Install development dependencies
RUN pip install --user \
    pytest>=7.4.0 \
    pytest-asyncio>=0.21.0 \
    pytest-cov>=4.1.0 \
    black>=23.0.0 \
    flake8>=6.0.0 \
    mypy>=1.5.0 \
    pre-commit>=3.4.0

# Install Jupyter for analysis
RUN pip install --user \
    jupyter>=1.0.0 \
    ipykernel>=6.25.0 \
    matplotlib>=3.7.0

# Copy the application code
COPY --chown=app:app . .

# Install the package in development mode
RUN pip install --user -e .

# Create directories for data and cache
RUN mkdir -p /app/data /app/cache /app/logs

# Expose port for Jupyter notebook
EXPOSE 8888

# Expose port for future web interface
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import vectorbt; import pandas; import numpy; print('OK')" || exit 1

# Default command
CMD ["bash"]