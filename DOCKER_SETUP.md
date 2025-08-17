# Docker Development Environment for DarwinTD

## Quick Start

### 1. Build and Start Services
```bash
# Build the Docker image
docker-compose build

# Start all services (PostgreSQL, Redis, main app)
docker-compose up -d

# Or start just the main development container
docker-compose up darwintd
```

### 2. Test Environment
```bash
# Run environment tests
docker-compose exec darwintd python test_docker_environment.py

# Or run tests during build
docker-compose run darwintd python test_docker_environment.py
```

### 3. Interactive Development
```bash
# Start interactive shell in container
docker-compose exec darwintd bash

# Or start a new container with shell
docker-compose run darwintd bash
```

## Development Workflow

### Code Development
```bash
# All code changes are automatically synced via volume mount
# Edit files on Windows host, run in container

# Run tests
docker-compose exec darwintd pytest tests/

# Run code quality checks
docker-compose exec darwintd black src/ tests/
docker-compose exec darwintd flake8 src/ tests/
docker-compose exec darwintd mypy src/
```

### Jupyter Notebook Development
```bash
# Start Jupyter service
docker-compose up jupyter

# Access Jupyter at http://localhost:8889
# No password required (development only!)
```

### Database Operations
```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U darwintd -d darwintd

# Or from application container
docker-compose exec darwintd python -c "
from sqlalchemy import create_engine
engine = create_engine('postgresql://darwintd:darwintd_pass@postgres:5432/darwintd')
print('Database connection successful')
"
```

## Docker Services

### Main Application Container (`darwintd`)
- **Purpose**: Main development environment
- **Includes**: Python 3.11, VectorBT, all dependencies
- **Ports**: 8888 (Jupyter), 8000 (future web interface)
- **Volumes**: Source code, data, cache, logs

### PostgreSQL Database (`postgres`)
- **Purpose**: Production-like database for development
- **Version**: PostgreSQL 15
- **Access**: `localhost:5432`
- **Credentials**: `darwintd:darwintd_pass@postgres:5432/darwintd`

### Redis Cache (`redis`)
- **Purpose**: Caching and future real-time features
- **Version**: Redis 7
- **Access**: `localhost:6379`

### Jupyter Notebook (`jupyter`)
- **Purpose**: Data analysis and research
- **Access**: `http://localhost:8889`
- **Features**: No authentication (development only)

## Environment Variables

### Application Settings
```bash
DEBUG=True                    # Development mode
LOG_LEVEL=INFO               # Logging level
PYTHONPATH=/app/src          # Python package path
```

### Database Configuration
```bash
DATABASE_URL=postgresql://darwintd:darwintd_pass@postgres:5432/darwintd
```

### Directory Configuration
```bash
DATA_DIR=/app/data           # Market data storage
CACHE_DIR=/app/cache         # Performance cache
```

## Data Persistence

### Docker Volumes
- `darwintd-data`: Market data and databases
- `darwintd-cache`: Performance cache files
- `darwintd-logs`: Application logs
- `postgres-data`: PostgreSQL database files
- `redis-data`: Redis cache files

### Volume Management
```bash
# List volumes
docker volume ls

# Backup data volume
docker run --rm -v darwintd-data:/data -v $(pwd):/backup alpine tar czf /backup/data-backup.tar.gz -C /data .

# Restore data volume
docker run --rm -v darwintd-data:/data -v $(pwd):/backup alpine tar xzf /backup/data-backup.tar.gz -C /data
```

## Common Commands

### Container Management
```bash
# View running containers
docker-compose ps

# View logs
docker-compose logs darwintd
docker-compose logs postgres

# Restart services
docker-compose restart darwintd

# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

### Development Commands
```bash
# Install new package
docker-compose exec darwintd pip install --user package-name

# Update requirements.txt
docker-compose exec darwintd pip freeze --user > requirements.txt

# Rebuild after dependency changes
docker-compose build --no-cache darwintd
```

### Performance Testing
```bash
# Test VectorBT performance
docker-compose exec darwintd python -c "
import vectorbt as vbt
import pandas as pd
import numpy as np
import time

# Generate test data
dates = pd.date_range('2020-01-01', '2024-01-01', freq='1H')
data = pd.DataFrame({
    'open': np.random.randn(len(dates)).cumsum() + 100,
    'high': np.random.randn(len(dates)).cumsum() + 105,
    'low': np.random.randn(len(dates)).cumsum() + 95,
    'close': np.random.randn(len(dates)).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, len(dates))
}, index=dates)

print(f'Test data: {len(data)} rows')

# Test portfolio backtesting speed
start_time = time.time()
entries = data['close'].pct_change() > 0.02
exits = data['close'].pct_change() < -0.02
portfolio = vbt.Portfolio.from_signals(data['close'], entries, exits)
end_time = time.time()

print(f'Backtest time: {end_time - start_time:.3f} seconds')
print(f'Total return: {portfolio.total_return():.3f}')
print(f'Sharpe ratio: {portfolio.sharpe_ratio():.3f}')
"
```

## Troubleshooting

### Common Issues

#### 1. VectorBT Installation Fails
```bash
# Check build logs
docker-compose build --no-cache darwintd

# Common fix: increase Docker memory allocation
# Docker Desktop -> Settings -> Resources -> Memory: 8GB+
```

#### 2. Permission Errors
```bash
# Fix file permissions (run on host)
sudo chown -R $USER:$USER .
```

#### 3. Port Conflicts
```bash
# Check if ports are in use on host
netstat -tlnp | grep 8888
netstat -tlnp | grep 5432

# Change ports in docker-compose.yml if needed
```

#### 4. Database Connection Issues
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Check PostgreSQL logs
docker-compose logs postgres

# Test connection
docker-compose exec postgres pg_isready -U darwintd
```

#### 5. Memory Issues
```bash
# Check container memory usage
docker stats

# Increase Docker memory allocation
# Docker Desktop -> Settings -> Resources -> Memory: 8GB+
```

### Performance Optimization

#### WSL2 Optimization
```bash
# In Windows PowerShell (as Administrator)
# Create or edit C:\Users\{username}\.wslconfig
[wsl2]
memory=8GB
processors=4
swap=2GB
```

#### Docker Optimization
```bash
# Optimize Docker for development
# Docker Desktop -> Settings -> Resources:
# - Memory: 8GB+
# - CPUs: 4+
# - Swap: 2GB+

# Enable BuildKit for faster builds
export DOCKER_BUILDKIT=1
```

## Development Tips

### 1. Efficient Development Cycle
- Edit code on Windows host using your preferred IDE
- Run and test in Docker container
- Use volume mounts for instant code synchronization

### 2. Debugging
```bash
# Interactive debugging
docker-compose exec darwintd python -m pdb script.py

# View container logs
docker-compose logs -f darwintd
```

### 3. Package Management
```bash
# Add new dependency
echo "new-package>=1.0.0" >> requirements.txt
docker-compose build darwintd
```

### 4. Data Management
- Use Docker volumes for persistent data
- Backup volumes regularly
- Keep large datasets in mounted volumes

## Security Notes

- **Development Only**: This setup is for development, not production
- **No Authentication**: Jupyter has no password (localhost access only)
- **Default Passwords**: Database uses default passwords
- **Port Exposure**: Services exposed on localhost only

For production deployment, implement proper security measures:
- Change default passwords
- Enable authentication
- Use environment variables for secrets
- Implement proper network security