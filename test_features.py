"""
Test suite for Features Module
Production-ready tests for Kalman filter and feature store
"""

import pytest
import asyncio
import numpy as np
import time
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

import redis.asyncio as redis

from medallion_x.features.kalman_filter import (
    AdaptiveKalmanFilter, KalmanFilterManager, FilteredData, KalmanState
)
from medallion_x.features.feature_store import (
    FeatureStore, FeatureComputer, FeatureDefinition, FeatureValue, FeatureType
)


@pytest.fixture
async def redis_client():
    """Create a test Redis client"""
    client = redis.from_url("redis://localhost:6379/15")  # Use DB 15 for tests
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.close()


@pytest.fixture
def kalman_filter():
    """Create a Kalman filter instance for testing"""
    return AdaptiveKalmanFilter("BTC/USDT", "binance")


@pytest.fixture
def kalman_manager(redis_client):
    """Create a Kalman filter manager for testing"""
    return KalmanFilterManager(redis_client)


@pytest.fixture
def feature_store(redis_client):
    """Create a feature store instance for testing"""
    return FeatureStore(redis_client)


class TestAdaptiveKalmanFilter:
    """Test cases for AdaptiveKalmanFilter class"""
    
    def test_initialization(self, kalman_filter):
        """Test proper initialization of Kalman filter"""
        assert kalman_filter.symbol == "BTC/USDT"
        assert kalman_filter.exchange == "binance"
        assert kalman_filter.state_dim == 3
        assert kalman_filter.state.shape == (3,)
        assert kalman_filter.covariance.shape == (3, 3)
        assert kalman_filter.metrics['filter_updates'] == 0

    def test_predict(self, kalman_filter):
        """Test prediction step"""
        predicted_state, predicted_covariance = kalman_filter.predict()
        
        assert predicted_state.shape == (3,)
        assert predicted_covariance.shape == (3, 3)
        assert not np.array_equal(predicted_state, kalman_filter.state)  # Should be different

    def test_update(self, kalman_filter):
        """Test update step with measurement"""
        measurement = 50000.0
        timestamp = int(time.time() * 1000)
        
        filtered_data = kalman_filter.update(measurement, timestamp)
        
        assert isinstance(filtered_data, FilteredData)
        assert filtered_data.original_price == measurement
        assert filtered_data.symbol == "BTC/USDT"
        assert filtered_data.exchange == "binance"
        assert filtered_data.timestamp == timestamp
        assert kalman_filter.metrics['filter_updates'] == 1

    def test_adaptive_parameters(self, kalman_filter):
        """Test adaptive parameter adjustment"""
        # Add some innovations to the window
        for i in range(20):
            kalman_filter._adapt_parameters(i * 0.1, np.array([[1.0]]))
        
        # Check that adaptation occurred
        assert len(kalman_filter.innovation_window) <= kalman_filter.max_window_size
        assert kalman_filter.R[0, 0] > 0  # Measurement noise should be positive

    def test_get_state(self, kalman_filter):
        """Test getting filter state"""
        state = kalman_filter.get_state()
        
        assert isinstance(state, KalmanState)
        assert state.symbol == "BTC/USDT"
        assert state.exchange == "binance"
        assert state.state_vector.shape == (3,)
        assert state.covariance_matrix.shape == (3, 3)

    def test_reset(self, kalman_filter):
        """Test filter reset"""
        # Update filter to change state
        kalman_filter.update(50000.0, int(time.time() * 1000))
        
        # Reset filter
        kalman_filter.reset()
        
        # Check that state is reset
        assert np.allclose(kalman_filter.state, np.zeros(3))
        assert kalman_filter.metrics['filter_updates'] == 0
        assert len(kalman_filter.innovation_window) == 0


class TestKalmanFilterManager:
    """Test cases for KalmanFilterManager class"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, kalman_manager):
        """Test proper initialization of manager"""
        assert kalman_manager.redis_client is not None
        assert len(kalman_manager.filters) == 0
        assert kalman_manager.metrics['total_filters'] == 0

    @pytest.mark.asyncio
    async def test_get_or_create_filter(self, kalman_manager):
        """Test getting or creating filters"""
        # First call should create new filter
        filter1 = kalman_manager.get_or_create_filter("BTC/USDT", "binance")
        assert isinstance(filter1, AdaptiveKalmanFilter)
        assert kalman_manager.metrics['total_filters'] == 1
        
        # Second call should return existing filter
        filter2 = kalman_manager.get_or_create_filter("BTC/USDT", "binance")
        assert filter1 is filter2
        assert kalman_manager.metrics['total_filters'] == 1

    @pytest.mark.asyncio
    async def test_process_price_update(self, kalman_manager):
        """Test processing price updates"""
        filtered_data = await kalman_manager.process_price_update(
            symbol="BTC/USDT",
            exchange="binance",
            price=50000.0,
            timestamp=int(time.time() * 1000)
        )
        
        assert isinstance(filtered_data, FilteredData)
        assert filtered_data.original_price == 50000.0
        assert filtered_data.symbol == "BTC/USDT"
        assert filtered_data.exchange == "binance"
        assert kalman_manager.metrics['total_updates'] == 1

    @pytest.mark.asyncio
    async def test_process_batch_updates(self, kalman_manager):
        """Test batch processing of updates"""
        updates = [
            {
                'symbol': 'BTC/USDT',
                'exchange': 'binance',
                'price': 50000.0,
                'timestamp': int(time.time() * 1000)
            },
            {
                'symbol': 'BTC/USDT',
                'exchange': 'binance',
                'price': 50100.0,
                'timestamp': int(time.time() * 1000) + 1000
            }
        ]
        
        results = await kalman_manager.process_batch_updates(updates)
        
        assert len(results) == 2
        assert all(isinstance(r, FilteredData) for r in results)
        assert kalman_manager.metrics['total_updates'] == 2

    @pytest.mark.asyncio
    async def test_get_latest_filtered_data(self, kalman_manager):
        """Test retrieving latest filtered data"""
        # Process an update first
        await kalman_manager.process_price_update(
            symbol="BTC/USDT",
            exchange="binance",
            price=50000.0,
            timestamp=int(time.time() * 1000)
        )
        
        # Retrieve the data
        filtered_data = await kalman_manager.get_latest_filtered_data("BTC/USDT", "binance")
        
        assert filtered_data is not None
        assert isinstance(filtered_data, FilteredData)
        assert filtered_data.original_price == 50000.0

    @pytest.mark.asyncio
    async def test_reset_filter(self, kalman_manager):
        """Test resetting a specific filter"""
        # Create and update a filter
        await kalman_manager.process_price_update(
            symbol="BTC/USDT",
            exchange="binance",
            price=50000.0,
            timestamp=int(time.time() * 1000)
        )
        
        # Reset the filter
        await kalman_manager.reset_filter("BTC/USDT", "binance")
        
        # Check that filter was reset (metrics should be 0)
        key = kalman_manager.get_filter_key("BTC/USDT", "binance")
        assert kalman_manager.filters[key].metrics['filter_updates'] == 0


class TestFeatureComputer:
    """Test cases for FeatureComputer class"""
    
    def test_initialization(self, feature_store):
        """Test proper initialization of feature computer"""
        computer = feature_store.feature_computer
        assert hasattr(computer, 'price_history')
        assert hasattr(computer, 'volume_history')
        assert hasattr(computer, 'technical_indicators')

    def test_update_price_history(self, feature_store):
        """Test updating price history"""
        computer = feature_store.feature_computer
        symbol = "BTC/USDT"
        timestamp = int(time.time() * 1000)
        price = 50000.0
        
        computer.update_price_history(symbol, price, timestamp)
        
        assert symbol in computer.price_history
        assert len(computer.price_history[symbol]) == 1
        assert computer.price_history[symbol][0] == (timestamp, price)

    def test_get_price_array(self, feature_store):
        """Test getting price array"""
        computer = feature_store.feature_computer
        symbol = "BTC/USDT"
        timestamp = int(time.time() * 1000)
        
        # Add some price data
        for i in range(10):
            computer.update_price_history(symbol, 50000.0 + i, timestamp + i * 1000)
        
        price_array = computer.get_price_array(symbol, 5)
        
        assert len(price_array) == 5
        assert price_array[0] == 50000.0
        assert price_array[-1] == 50004.0

    def test_compute_price_features(self, feature_store):
        """Test computing price features"""
        computer = feature_store.feature_computer
        symbol = "BTC/USDT"
        timestamp = int(time.time() * 1000)
        
        # Add price data
        for i in range(25):
            computer.update_price_history(symbol, 50000.0 + i, timestamp + i * 1000)
        
        features = computer.compute_price_features(symbol, timestamp)
        
        assert len(features) > 0
        assert all(isinstance(f, FeatureValue) for f in features)
        assert all(f.symbol == symbol for f in features)

    def test_compute_technical_features(self, feature_store):
        """Test computing technical features"""
        computer = feature_store.feature_computer
        symbol = "BTC/USDT"
        timestamp = int(time.time() * 1000)
        
        # Add price data
        for i in range(50):
            computer.update_price_history(symbol, 50000.0 + i * 10, timestamp + i * 1000)
        
        features = computer.compute_technical_features(symbol, timestamp)
        
        assert len(features) > 0
        assert all(isinstance(f, FeatureValue) for f in features)
        
        # Check for specific indicators
        feature_names = [f.name for f in features]
        assert "rsi_14" in feature_names
        assert "atr_14" in feature_names

    def test_compute_volume_features(self, feature_store):
        """Test computing volume features"""
        computer = feature_store.feature_computer
        symbol = "BTC/USDT"
        timestamp = int(time.time() * 1000)
        
        # Add volume data
        for i in range(25):
            computer.update_volume_history(symbol, 1000.0 + i * 10, timestamp + i * 1000)
        
        features = computer.compute_volume_features(symbol, timestamp)
        
        assert len(features) > 0
        assert all(isinstance(f, FeatureValue) for f in features)
        assert all(f.symbol == symbol for f in features)


class TestFeatureStore:
    """Test cases for FeatureStore class"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, feature_store):
        """Test proper initialization of feature store"""
        assert feature_store.redis_client is not None
        assert len(feature_store.feature_definitions) > 0
        assert feature_store.feature_computer is not None

    @pytest.mark.asyncio
    async def test_process_market_data(self, feature_store):
        """Test processing market data"""
        features = await feature_store.process_market_data(
            symbol="BTC/USDT",
            exchange="binance",
            price=50000.0,
            volume=1000.0,
            timestamp=int(time.time() * 1000)
        )
        
        assert len(features) > 0
        assert all(isinstance(f, FeatureValue) for f in features)
        assert feature_store.metrics['features_computed'] > 0

    @pytest.mark.asyncio
    async def test_process_news_sentiment(self, feature_store):
        """Test processing news sentiment"""
        feature = await feature_store.process_news_sentiment(
            symbol="BTC/USDT",
            sentiment_score=0.7,
            timestamp=int(time.time() * 1000)
        )
        
        assert isinstance(feature, FeatureValue)
        assert feature.name == "news_sentiment"
        assert feature.value == 0.7
        assert feature.exchange == "news"

    @pytest.mark.asyncio
    async def test_process_onchain_metrics(self, feature_store):
        """Test processing on-chain metrics"""
        metrics = {
            'transaction_count': 1000,
            'whale_movements': 5,
            'exchange_flow': 100000.0
        }
        
        features = await feature_store.process_onchain_metrics(
            symbol="BTC/USDT",
            metrics=metrics,
            timestamp=int(time.time() * 1000)
        )
        
        assert len(features) == 3
        assert all(isinstance(f, FeatureValue) for f in features)
        assert all(f.name.startswith("onchain_") for f in features)

    @pytest.mark.asyncio
    async def test_get_latest_features(self, feature_store):
        """Test retrieving latest features"""
        # Process some data first
        await feature_store.process_market_data(
            symbol="BTC/USDT",
            exchange="binance",
            price=50000.0,
            volume=1000.0,
            timestamp=int(time.time() * 1000)
        )
        
        # Retrieve features
        features = await feature_store.get_latest_features("BTC/USDT")
        
        assert len(features) > 0
        assert all(isinstance(name, str) for name in features.keys())
        assert all(isinstance(f, FeatureValue) for f in features.values())

    @pytest.mark.asyncio
    async def test_create_feature_vector(self, feature_store):
        """Test creating feature vector"""
        # Process some data first
        await feature_store.process_market_data(
            symbol="BTC/USDT",
            exchange="binance",
            price=50000.0,
            volume=1000.0,
            timestamp=int(time.time() * 1000)
        )
        
        # Create feature vector
        vector = await feature_store.create_feature_vector("BTC/USDT")
        
        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0

    def test_add_feature_definition(self, feature_store):
        """Test adding feature definition"""
        new_feature = FeatureDefinition(
            name="test_feature",
            feature_type=FeatureType.DERIVED,
            description="Test feature for unit testing"
        )
        
        feature_store.add_feature_definition(new_feature)
        
        assert "test_feature" in feature_store.feature_definitions
        assert feature_store.feature_definitions["test_feature"].name == "test_feature"

    def test_remove_feature_definition(self, feature_store):
        """Test removing feature definition"""
        # Add a feature first
        new_feature = FeatureDefinition(
            name="test_feature_remove",
            feature_type=FeatureType.DERIVED,
            description="Test feature for removal"
        )
        feature_store.add_feature_definition(new_feature)
        
        # Remove it
        feature_store.remove_feature_definition("test_feature_remove")
        
        assert "test_feature_remove" not in feature_store.feature_definitions

    def test_get_feature_definitions(self, feature_store):
        """Test getting feature definitions"""
        # Get all definitions
        all_defs = feature_store.get_feature_definitions()
        assert len(all_defs) > 0
        assert all(isinstance(d, FeatureDefinition) for d in all_defs)
        
        # Get filtered definitions
        price_defs = feature_store.get_feature_definitions(FeatureType.PRICE)
        assert all(d.feature_type == FeatureType.PRICE for d in price_defs)


@pytest.mark.integration
class TestFeaturesIntegration:
    """Integration tests for features module"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_feature_pipeline(self, redis_client):
        """Test end-to-end feature processing pipeline"""
        # Create components
        kalman_manager = KalmanFilterManager(redis_client)
        feature_store = FeatureStore(redis_client)
        
        # Process market data through Kalman filter
        filtered_data = await kalman_manager.process_price_update(
            symbol="BTC/USDT",
            exchange="binance",
            price=50000.0,
            timestamp=int(time.time() * 1000)
        )
        
        # Process filtered data through feature store
        features = await feature_store.process_market_data(
            symbol="BTC/USDT",
            exchange="binance",
            price=filtered_data.filtered_price,
            volume=1000.0,
            timestamp=filtered_data.timestamp
        )
        
        # Verify pipeline worked
        assert isinstance(filtered_data, FilteredData)
        assert len(features) > 0
        assert kalman_manager.metrics['total_updates'] > 0
        assert feature_store.metrics['features_computed'] > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
