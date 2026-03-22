"""
Medallion-X Configuration Settings
Production-ready environment variables and system configuration
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class Environment(Enum):
    DEVELOPMENT = "development"
    TESTNET = "testnet"
    PRODUCTION = "production"

@dataclass
class RedisConfig:
    url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    max_connections: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "100"))
    retry_on_timeout: bool = True
    socket_timeout: int = 5
    socket_connect_timeout: int = 5

@dataclass
class ExchangeConfig:
    api_key: str
    secret: str
    sandbox: bool = False
    rate_limit: int = 1200  # requests per minute
    testnet: bool = True

@dataclass
class DatabaseConfig:
    postgres_url: str = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost/medallionx")
    pool_size: int = 20
    max_overflow: int = 30

@dataclass
class AIConfig:
    model_path: str = os.getenv("MODEL_PATH", "./models")
    learning_rate: float = 0.001
    batch_size: int = 64
    episode_length: int = 1000
    memory_size: int = 100000
    target_update_freq: int = 1000

@dataclass
class RiskConfig:
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk
    max_position_size: float = 0.1   # 10% max position size
    kelly_multiplier: float = 0.25   # Conservative Kelly
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0

@dataclass
class ExecutionConfig:
    max_slippage: float = 0.001  # 0.1% max slippage
    min_order_size: float = 0.001
    execution_timeout_ms: int = 50  # 50ms timeout
    retry_attempts: int = 3
    bad_setup_filter_threshold: float = 0.95  # 95% confidence threshold

@dataclass
class MonitoringConfig:
    fastapi_host: str = "0.0.0.0"
    fastapi_port: int = 8000
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    log_level: str = "INFO"
    metrics_retention_days: int = 30

class MedallionXConfig:
    def __init__(self, env: Environment = Environment.TESTNET):
        self.env = env
        self.redis = RedisConfig()
        self.database = DatabaseConfig()
        self.ai = AIConfig()
        self.risk = RiskConfig()
        self.execution = ExecutionConfig()
        self.monitoring = MonitoringConfig()
        
        # Exchange configurations
        self.exchanges = self._load_exchange_configs()
        
        # Trading symbols
        self.symbols = self._load_trading_symbols()
        
        # Data sources
        self.news_apis = self._load_news_apis()
        self.onchain_apis = self._load_onchain_apis()

    def _load_exchange_configs(self) -> Dict[str, ExchangeConfig]:
        """Load exchange configurations from environment variables"""
        exchanges = {}
        
        # Binance
        exchanges['binance'] = ExchangeConfig(
            api_key=os.getenv('BINANCE_API_KEY', ''),
            secret=os.getenv('BINANCE_SECRET', ''),
            testnet=self.env != Environment.PRODUCTION
        )
        
        # Bybit
        exchanges['bybit'] = ExchangeConfig(
            api_key=os.getenv('BYBIT_API_KEY', ''),
            secret=os.getenv('BYBIT_SECRET', ''),
            testnet=self.env != Environment.PRODUCTION
        )
        
        # OKX
        exchanges['okx'] = ExchangeConfig(
            api_key=os.getenv('OKX_API_KEY', ''),
            secret=os.getenv('OKX_SECRET', ''),
            testnet=self.env != Environment.PRODUCTION
        )
        
        return exchanges

    def _load_trading_symbols(self) -> List[str]:
        """Load trading symbols from environment or use defaults"""
        symbols_str = os.getenv('TRADING_SYMBOLS', 'BTC/USDT,ETH/USDT,SOL/USDT')
        return [s.strip() for s in symbols_str.split(',')]

    def _load_news_apis(self) -> Dict[str, str]:
        """Load news API configurations"""
        return {
            'newsapi': os.getenv('NEWSAPI_KEY', ''),
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_KEY', ''),
            'cryptocompare': os.getenv('CRYPTOCOMPARE_KEY', '')
        }

    def _load_onchain_apis(self) -> Dict[str, str]:
        """Load on-chain data API configurations"""
        return {
            'moralis': os.getenv('MORALIS_KEY', ''),
            'alchemy': os.getenv('ALCHEMY_KEY', ''),
            'infura': os.getenv('INFURA_KEY', '')
        }

    def is_production(self) -> bool:
        return self.env == Environment.PRODUCTION

    def is_testnet(self) -> bool:
        return self.env == Environment.TESTNET

# Global configuration instance
config = MedallionXConfig()
