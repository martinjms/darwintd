"""
Configuration settings for DarwinTD.
"""

from typing import Optional, List
from pydantic import BaseSettings, Field
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application Info
    app_name: str = "DarwinTD"
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Data Configuration
    data_dir: Path = Field(default=Path("./data"), env="DATA_DIR")
    cache_dir: Path = Field(default=Path("./cache"), env="CACHE_DIR")
    database_url: str = Field(env="DATABASE_URL", default="sqlite:///./darwintd.db")
    
    # Trading Configuration
    paper_trading: bool = Field(default=True, env="PAPER_TRADING")
    max_position_size: float = Field(default=1000.0, env="MAX_POSITION_SIZE")
    risk_tolerance: float = Field(default=0.02, env="RISK_TOLERANCE")
    
    # Exchange API Keys
    binance_api_key: Optional[str] = Field(default=None, env="BINANCE_API_KEY")
    binance_secret_key: Optional[str] = Field(default=None, env="BINANCE_SECRET_KEY")
    coinbase_api_key: Optional[str] = Field(default=None, env="COINBASE_API_KEY")
    coinbase_secret_key: Optional[str] = Field(default=None, env="COINBASE_SECRET_KEY")
    kraken_api_key: Optional[str] = Field(default=None, env="KRAKEN_API_KEY")
    kraken_secret_key: Optional[str] = Field(default=None, env="KRAKEN_SECRET_KEY")
    
    # External API Keys
    alpha_vantage_api_key: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")
    jina_ai_api_key: Optional[str] = Field(default=None, env="JINA_AI_API_KEY")
    github_token: Optional[str] = Field(default=None, env="GITHUB_TOKEN")
    
    # Performance Settings
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    cache_expiry_seconds: int = Field(default=3600, env="CACHE_EXPIRY_SECONDS")
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    
    # Target Assets for MVP
    target_symbols: List[str] = Field(default=[
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "LINK/USDT"
    ])
    
    # Supported Timeframes
    timeframes: List[str] = Field(default=[
        "1m", "5m", "15m", "1h", "4h", "1d"
    ])
    
    # Exchange Configuration
    preferred_exchanges: List[str] = Field(default=[
        "binance", "coinbase", "kraken"
    ])
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()