"""
Medallion-X Feature Store Module
Centralized feature management with real-time computation and caching
Production-ready feature engineering for AI/ML models
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
import time
import json
from collections import deque
import talib

import redis.asyncio as redis
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from ..config.settings import config
from ..utils.logging import get_logger

logger = get_logger(__name__)

class FeatureType(Enum):
    """Feature type enumeration"""
    PRICE = "price"
    VOLUME = "volume"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    ONCHAIN = "onchain"
    DERIVED = "derived"
    TARGET = "target"

@dataclass
class FeatureDefinition:
    """Feature definition with metadata"""
    name: str
    feature_type: FeatureType
    description: str
    computation_function: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    dependencies: Optional[List[str]] = None
    ttl_seconds: int = 300  # 5 minutes default TTL
    is_active: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

@dataclass
class FeatureValue:
    """Feature value with metadata"""
    name: str
    value: Union[float, int, str, List[float]]
    timestamp: int
    symbol: str
    exchange: str
    quality_score: float = 1.0  # 0-1 quality indicator
    confidence_interval: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

class TechnicalIndicators:
    """Collection of technical indicator calculations"""
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> float:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0  # Neutral
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """MACD indicator"""
        if len(prices) < slow:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
        
        ema_fast = talib.EMA(prices, timeperiod=fast)
        ema_slow = talib.EMA(prices, timeperiod=slow)
        
        macd_line = ema_fast[-1] - ema_slow[-1] if not np.isnan(ema_fast[-1]) and not np.isnan(ema_slow[-1]) else 0.0
        
        # For simplicity, returning current MACD value
        return {
            'macd': macd_line,
            'signal': 0.0,  # Would need historical MACD values for signal line
            'histogram': 0.0
        }

    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
        """Bollinger Bands"""
        if len(prices) < period:
            return {'upper': 0.0, 'middle': 0.0, 'lower': 0.0, 'bandwidth': 0.0}
        
        middle = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        bandwidth = (upper - lower) / middle if middle > 0 else 0.0
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'bandwidth': bandwidth
        }

    @staticmethod
    def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3) -> Dict[str, float]:
        """Stochastic Oscillator"""
        if len(close) < k_period:
            return {'k': 50.0, 'd': 50.0}
        
        highest_high = np.max(high[-k_period:])
        lowest_low = np.min(low[-k_period:])
        
        if highest_high == lowest_low:
            k_percent = 50.0
        else:
            k_percent = 100 * (close[-1] - lowest_low) / (highest_high - lowest_low)
        
        # Simplified D calculation (would need moving average of K)
        d_percent = k_percent  # Placeholder
        
        return {'k': k_percent, 'd': d_percent}

    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Average True Range"""
        if len(close) < period + 1:
            return 0.0
        
        tr_values = []
        for i in range(1, len(close)):
            high_low = high[i] - low[i]
            high_close = abs(high[i] - close[i-1])
            low_close = abs(low[i] - close[i-1])
            tr = max(high_low, high_close, low_close)
            tr_values.append(tr)
        
        if len(tr_values) >= period:
            atr = np.mean(tr_values[-period:])
        else:
            atr = np.mean(tr_values) if tr_values else 0.0
        
        return atr

class FeatureComputer:
    """Real-time feature computation engine"""
    
    def __init__(self):
        self.technical_indicators = TechnicalIndicators()
        self.price_history: Dict[str, deque] = {}  # symbol -> price history
        self.volume_history: Dict[str, deque] = {}  # symbol -> volume history
        self.max_history_length = 1000
        
        # Scalers for normalization
        self.scalers: Dict[str, StandardScaler] = {}

    def update_price_history(self, symbol: str, price: float, timestamp: int) -> None:
        """Update price history for a symbol"""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.max_history_length)
        
        self.price_history[symbol].append((timestamp, price))

    def update_volume_history(self, symbol: str, volume: float, timestamp: int) -> None:
        """Update volume history for a symbol"""
        if symbol not in self.volume_history:
            self.volume_history[symbol] = deque(maxlen=self.max_history_length)
        
        self.volume_history[symbol].append((timestamp, volume))

    def get_price_array(self, symbol: str, count: int = 100) -> np.ndarray:
        """Get recent price array for technical indicators"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < count:
            return np.array([])
        
        recent_prices = [price for _, price in list(self.price_history[symbol])[-count:]]
        return np.array(recent_prices)

    def get_ohlcv_arrays(self, symbol: str, count: int = 100) -> Dict[str, np.ndarray]:
        """Get OHLCV arrays (simplified - using close prices for all)"""
        prices = self.get_price_array(symbol, count)
        volumes = self.get_volume_array(symbol, count)
        
        if len(prices) == 0:
            return {'open': np.array([]), 'high': np.array([]), 'low': np.array([]), 'close': np.array([]), 'volume': np.array([])}
        
        # Simplified OHLC from close prices
        return {
            'open': prices,
            'high': prices,
            'low': prices,
            'close': prices,
            'volume': volumes
        }

    def get_volume_array(self, symbol: str, count: int = 100) -> np.ndarray:
        """Get recent volume array"""
        if symbol not in self.volume_history or len(self.volume_history[symbol]) < count:
            return np.array([])
        
        recent_volumes = [volume for _, volume in list(self.volume_history[symbol])[-count:]]
        return np.array(recent_volumes)

    def compute_price_features(self, symbol: str, timestamp: int) -> List[FeatureValue]:
        """Compute price-based features"""
        features = []
        prices = self.get_price_array(symbol, 100)
        
        if len(prices) < 2:
            return features
        
        current_price = prices[-1]
        
        # Price change features
        if len(prices) >= 2:
            price_change_1 = (current_price - prices[-2]) / prices[-2]
            features.append(FeatureValue(
                name="price_change_1m",
                value=price_change_1,
                timestamp=timestamp,
                symbol=symbol,
                exchange="combined"
            ))
        
        if len(prices) >= 60:  # 1 hour if 1-minute data
            price_change_60 = (current_price - prices[-60]) / prices[-60]
            features.append(FeatureValue(
                name="price_change_1h",
                value=price_change_60,
                timestamp=timestamp,
                symbol=symbol,
                exchange="combined"
            ))
        
        # Price position relative to recent range
        if len(prices) >= 20:
            price_min = np.min(prices[-20:])
            price_max = np.max(prices[-20:])
            price_position = (current_price - price_min) / (price_max - price_min) if price_max > price_min else 0.5
            
            features.append(FeatureValue(
                name="price_position_20",
                value=price_position,
                timestamp=timestamp,
                symbol=symbol,
                exchange="combined"
            ))
        
        return features

    def compute_technical_features(self, symbol: str, timestamp: int) -> List[FeatureValue]:
        """Compute technical indicator features"""
        features = []
        ohlcv = self.get_ohlcv_arrays(symbol, 100)
        
        if len(ohlcv['close']) < 20:
            return features
        
        close_prices = ohlcv['close']
        
        # RSI
        rsi_value = self.technical_indicators.rsi(close_prices, 14)
        features.append(FeatureValue(
            name="rsi_14",
            value=rsi_value,
            timestamp=timestamp,
            symbol=symbol,
            exchange="combined"
        ))
        
        # MACD
        macd_values = self.technical_indicators.macd(close_prices)
        for key, value in macd_values.items():
            features.append(FeatureValue(
                name=f"macd_{key}",
                value=value,
                timestamp=timestamp,
                symbol=symbol,
                exchange="combined"
            ))
        
        # Bollinger Bands
        bb_values = self.technical_indicators.bollinger_bands(close_prices)
        for key, value in bb_values.items():
            features.append(FeatureValue(
                name=f"bb_{key}",
                value=value,
                timestamp=timestamp,
                symbol=symbol,
                exchange="combined"
            ))
        
        # ATR
        atr_value = self.technical_indicators.atr(ohlcv['high'], ohlcv['low'], ohlcv['close'])
        features.append(FeatureValue(
            name="atr_14",
            value=atr_value,
            timestamp=timestamp,
            symbol=symbol,
            exchange="combined"
        ))
        
        return features

    def compute_volume_features(self, symbol: str, timestamp: int) -> List[FeatureValue]:
        """Compute volume-based features"""
        features = []
        volumes = self.get_volume_array(symbol, 100)
        
        if len(volumes) < 2:
            return features
        
        current_volume = volumes[-1]
        
        # Volume change
        volume_change = (current_volume - volumes[-2]) / volumes[-2] if volumes[-2] > 0 else 0.0
        features.append(FeatureValue(
            name="volume_change",
            value=volume_change,
            timestamp=timestamp,
            symbol=symbol,
            exchange="combined"
        ))
        
        # Volume moving average
        if len(volumes) >= 20:
            volume_ma = np.mean(volumes[-20:])
            volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0
            features.append(FeatureValue(
                name="volume_ratio_20",
                value=volume_ratio,
                timestamp=timestamp,
                symbol=symbol,
                exchange="combined"
            ))
        
        return features

class FeatureStore:
    """
    Centralized feature store with real-time computation and caching
    - Feature definition and management
    - Real-time computation pipeline
    - Redis caching with TTL
    - Feature quality monitoring
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.feature_computer = FeatureComputer()
        self.feature_definitions: Dict[str, FeatureDefinition] = {}
        self.is_running = False
        
        # Initialize default feature definitions
        self._initialize_default_features()
        
        # Performance metrics
        self.metrics = {
            'features_computed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'computation_time_ms': 0.0,
            'active_features': 0,
            'quality_score_avg': 0.0
        }

    def _initialize_default_features(self) -> None:
        """Initialize default feature definitions"""
        default_features = [
            # Price features
            FeatureDefinition(
                name="price_change_1m",
                feature_type=FeatureType.PRICE,
                description="1-minute price change percentage"
            ),
            FeatureDefinition(
                name="price_change_1h",
                feature_type=FeatureType.PRICE,
                description="1-hour price change percentage"
            ),
            FeatureDefinition(
                name="price_position_20",
                feature_type=FeatureType.PRICE,
                description="Price position within 20-period range (0-1)"
            ),
            
            # Technical features
            FeatureDefinition(
                name="rsi_14",
                feature_type=FeatureType.TECHNICAL,
                description="14-period Relative Strength Index"
            ),
            FeatureDefinition(
                name="macd_macd",
                feature_type=FeatureType.TECHNICAL,
                description="MACD line value"
            ),
            FeatureDefinition(
                name="bb_bandwidth",
                feature_type=FeatureType.TECHNICAL,
                description="Bollinger Band bandwidth"
            ),
            FeatureDefinition(
                name="atr_14",
                feature_type=FeatureType.TECHNICAL,
                description="14-period Average True Range"
            ),
            
            # Volume features
            FeatureDefinition(
                name="volume_change",
                feature_type=FeatureType.VOLUME,
                description="Volume change percentage"
            ),
            FeatureDefinition(
                name="volume_ratio_20",
                feature_type=FeatureType.VOLUME,
                description="Current volume relative to 20-period average"
            ),
        ]
        
        for feature_def in default_features:
            self.feature_definitions[feature_def.name] = feature_def

    async def process_market_data(self, symbol: str, exchange: str, price: float, volume: float, timestamp: int) -> List[FeatureValue]:
        """Process market data and compute features"""
        start_time = time.time()
        
        try:
            # Update price and volume history
            self.feature_computer.update_price_history(symbol, price, timestamp)
            self.feature_computer.update_volume_history(symbol, volume, timestamp)
            
            # Compute features
            all_features = []
            
            # Price features
            price_features = self.feature_computer.compute_price_features(symbol, timestamp)
            all_features.extend(price_features)
            
            # Technical features
            technical_features = self.feature_computer.compute_technical_features(symbol, timestamp)
            all_features.extend(technical_features)
            
            # Volume features
            volume_features = self.feature_computer.compute_volume_features(symbol, timestamp)
            all_features.extend(volume_features)
            
            # Store features in Redis
            for feature in all_features:
                await self._store_feature(feature)
            
            # Update metrics
            self.metrics['features_computed'] += len(all_features)
            computation_time = (time.time() - start_time) * 1000
            self.metrics['computation_time_ms'] = (
                (self.metrics['computation_time_ms'] * (self.metrics['features_computed'] - len(all_features)) + computation_time) /
                self.metrics['features_computed']
            )
            
            return all_features
            
        except Exception as e:
            logger.error(f"Error computing features for {symbol}: {e}")
            return []

    async def process_news_sentiment(self, symbol: str, sentiment_score: float, timestamp: int) -> FeatureValue:
        """Process news sentiment as a feature"""
        feature = FeatureValue(
            name="news_sentiment",
            value=sentiment_score,
            timestamp=timestamp,
            symbol=symbol,
            exchange="news",
            quality_score=0.8  # News sentiment has moderate quality
        )
        
        await self._store_feature(feature)
        self.metrics['features_computed'] += 1
        
        return feature

    async def process_onchain_metrics(self, symbol: str, metrics: Dict[str, float], timestamp: int) -> List[FeatureValue]:
        """Process on-chain metrics as features"""
        features = []
        
        for metric_name, value in metrics.items():
            feature = FeatureValue(
                name=f"onchain_{metric_name}",
                value=value,
                timestamp=timestamp,
                symbol=symbol,
                exchange="onchain",
                quality_score=0.7  # On-chain data has good quality
            )
            
            features.append(feature)
            await self._store_feature(feature)
        
        self.metrics['features_computed'] += len(features)
        return features

    async def _store_feature(self, feature: FeatureValue) -> None:
        """Store feature in Redis"""
        key = f"features:{feature.symbol}:{feature.name}:latest"
        
        # Convert to JSON-serializable format
        feature_dict = asdict(feature)
        
        # Store with TTL
        await self.redis_client.setex(
            key,
            ttl=feature.ttl_seconds if hasattr(feature, 'ttl_seconds') else 300,
            value=json.dumps(feature_dict, default=str)
        )
        
        # Store in time series for history
        ts_key = f"features:ts:{feature.symbol}:{feature.name}"
        await self.redis_client.zadd(
            ts_key,
            {json.dumps(feature_dict, default=str): feature.timestamp}
        )
        # Keep only last 1000 records
        await self.redis_client.zremrangebyrank(ts_key, 0, -1001)

    async def get_latest_features(self, symbol: str, feature_names: Optional[List[str]] = None) -> Dict[str, FeatureValue]:
        """Get latest features for a symbol"""
        features = {}
        
        if feature_names is None:
            # Get all features for the symbol
            keys = await self.redis_client.keys(f"features:{symbol}:*:latest")
        else:
            # Get specific features
            keys = [f"features:{symbol}:{name}:latest" for name in feature_names]
        
        for key in keys:
            try:
                data = await self.redis_client.get(key)
                if data:
                    self.metrics['cache_hits'] += 1
                    feature_dict = json.loads(data.decode('utf-8'))
                    feature = FeatureValue(**feature_dict)
                    features[feature.name] = feature
                else:
                    self.metrics['cache_misses'] += 1
            except Exception as e:
                logger.error(f"Error retrieving feature from {key}: {e}")
                self.metrics['cache_misses'] += 1
        
        return features

    async def get_feature_history(self, symbol: str, feature_name: str, limit: int = 100) -> List[FeatureValue]:
        """Get historical feature values"""
        try:
            ts_key = f"features:ts:{symbol}:{feature_name}"
            data_points = await self.redis_client.zrevrange(ts_key, 0, limit - 1)
            
            features = []
            for data in data_points:
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                feature_dict = json.loads(data)
                features.append(FeatureValue(**feature_dict))
            
            return features
            
        except Exception as e:
            logger.error(f"Error retrieving feature history: {e}")
            return []

    async def create_feature_vector(self, symbol: str, timestamp: Optional[int] = None) -> np.ndarray:
        """Create feature vector for ML models"""
        if timestamp is None:
            timestamp = int(time.time() * 1000)
        
        # Get all latest features
        features = await self.get_latest_features(symbol)
        
        if not features:
            return np.array([])
        
        # Order features consistently
        feature_order = sorted(features.keys())
        feature_vector = np.array([features[name].value for name in feature_order])
        
        return feature_vector

    def add_feature_definition(self, feature_def: FeatureDefinition) -> None:
        """Add a new feature definition"""
        self.feature_definitions[feature_def.name] = feature_def
        logger.info(f"Added feature definition: {feature_def.name}")

    def remove_feature_definition(self, feature_name: str) -> None:
        """Remove a feature definition"""
        if feature_name in self.feature_definitions:
            del self.feature_definitions[feature_name]
            logger.info(f"Removed feature definition: {feature_name}")

    def get_feature_definitions(self, feature_type: Optional[FeatureType] = None) -> List[FeatureDefinition]:
        """Get feature definitions, optionally filtered by type"""
        features = list(self.feature_definitions.values())
        
        if feature_type:
            features = [f for f in features if f.feature_type == feature_type]
        
        return features

    async def start(self) -> None:
        """Start feature store"""
        self.is_running = True
        self.metrics['active_features'] = len([f for f in self.feature_definitions.values() if f.is_active])
        logger.info("Feature store started")

    async def stop(self) -> None:
        """Stop feature store"""
        self.is_running = False
        logger.info("Feature store stopped")

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.metrics,
            'total_definitions': len(self.feature_definitions),
            'cache_hit_rate': self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses']) if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0.0
        }
