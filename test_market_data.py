"""
Test suite for Market Data Ingestion Module
Production-ready unit tests with mocking and integration testing
"""

import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from typing import Dict, Any

import redis.asyncio as redis
import ccxt.pro as ccxt

from medallion_x.data_pipeline.market_data_ingestion import (
    MarketDataIngestion, OHLCVData, OrderBookData, TradeData
)
from medallion_x.config.settings import config


@pytest.fixture
async def redis_client():
    """Create a test Redis client"""
    client = redis.from_url("redis://localhost:6379/15")  # Use DB 15 for tests
    # Clear test data
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.close()


@pytest.fixture
def market_data_ingestion(redis_client):
    """Create MarketDataIngestion instance for testing"""
    return MarketDataIngestion(redis_client)


class TestMarketDataIngestion:
    """Test cases for MarketDataIngestion class"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, market_data_ingestion):
        """Test proper initialization of the market data ingestion"""
        assert market_data_ingestion.redis_client is not None
        assert len(market_data_ingestion.exchanges) > 0
        assert market_data_ingestion.is_running is False
        assert isinstance(market_data_ingestion.metrics, dict)
        
        # Check that all exchanges are initialized
        for exchange_id in config.exchanges.keys():
            assert exchange_id in market_data_ingestion.exchanges
            assert exchange_id in market_data_ingestion.active_connections

    @pytest.mark.asyncio
    async def test_store_ohlcv_data(self, market_data_ingestion):
        """Test storing OHLCV data in Redis"""
        test_data = OHLCVData(
            timestamp=int(time.time() * 1000),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0,
            exchange="binance",
            symbol="BTC/USDT"
        )
        
        await market_data_ingestion._store_ohlcv_data(test_data)
        
        # Verify data was stored
        key = f"ohlcv:{test_data.exchange}:{test_data.symbol}:latest"
        stored_data = await market_data_ingestion.redis_client.get(key)
        
        assert stored_data is not None
        parsed_data = json.loads(stored_data.decode('utf-8'))
        assert parsed_data['exchange'] == test_data.exchange
        assert parsed_data['symbol'] == test_data.symbol
        assert parsed_data['close'] == test_data.close

    @pytest.mark.asyncio
    async def test_store_orderbook_data(self, market_data_ingestion):
        """Test storing order book data in Redis"""
        test_data = OrderBookData(
            timestamp=int(time.time() * 1000),
            exchange="binance",
            symbol="BTC/USDT",
            bids=[(50000.0, 1.0), (49999.0, 2.0)],
            asks=[(50001.0, 1.5), (50002.0, 2.5)],
            spread=1.0
        )
        
        await market_data_ingestion._store_orderbook_data(test_data)
        
        # Verify data was stored
        key = f"orderbook:{test_data.exchange}:{test_data.symbol}:latest"
        stored_data = await market_data_ingestion.redis_client.get(key)
        
        assert stored_data is not None
        parsed_data = json.loads(stored_data.decode('utf-8'))
        assert parsed_data['exchange'] == test_data.exchange
        assert parsed_data['symbol'] == test_data.symbol
        assert len(parsed_data['bids']) == 2
        assert len(parsed_data['asks']) == 2

    @pytest.mark.asyncio
    async def test_store_trade_data(self, market_data_ingestion):
        """Test storing trade data in Redis"""
        test_data = TradeData(
            timestamp=int(time.time() * 1000),
            exchange="binance",
            symbol="BTC/USDT",
            price=50000.0,
            quantity=0.1,
            side="buy"
        )
        
        await market_data_ingestion._store_trade_data(test_data)
        
        # Verify data was stored
        key = f"trades:{test_data.exchange}:{test_data.symbol}:latest"
        stored_data = await market_data_ingestion.redis_client.get(key)
        
        assert stored_data is not None
        parsed_data = json.loads(stored_data.decode('utf-8'))
        assert parsed_data['exchange'] == test_data.exchange
        assert parsed_data['symbol'] == test_data.symbol
        assert parsed_data['price'] == test_data.price
        assert parsed_data['side'] == test_data.side

    @pytest.mark.asyncio
    async def test_data_callback(self, market_data_ingestion):
        """Test data callback functionality"""
        callback_called = asyncio.Event()
        callback_data = {}
        
        async def test_callback(data_type: str, data: Any):
            callback_called.set()
            callback_data['type'] = data_type
            callback_data['data'] = data
        
        market_data_ingestion.add_data_callback(test_callback)
        
        # Trigger callback with test data
        test_data = OHLCVData(
            timestamp=int(time.time() * 1000),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0,
            exchange="binance",
            symbol="BTC/USDT"
        )
        
        await market_data_ingestion._trigger_callbacks('ohlcv', test_data)
        
        # Wait for callback to be called (with timeout)
        await asyncio.wait_for(callback_called.wait(), timeout=1.0)
        
        assert callback_data['type'] == 'ohlcv'
        assert isinstance(callback_data['data'], OHLCVData)

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, market_data_ingestion):
        """Test metrics tracking functionality"""
        initial_metrics = market_data_ingestion.get_metrics()
        assert initial_metrics['messages_processed'] == 0
        assert initial_metrics['avg_processing_time_ms'] == 0.0
        
        # Simulate processing times
        market_data_ingestion._update_processing_time(10.0)
        market_data_ingestion._update_processing_time(20.0)
        market_data_ingestion._update_processing_time(15.0)
        
        updated_metrics = market_data_ingestion.get_metrics()
        assert updated_metrics['messages_processed'] == 3
        assert updated_metrics['avg_processing_time_ms'] == 15.0  # (10+20+15)/3

    @pytest.mark.asyncio
    @patch('ccxt.pro.binance')
    async def test_exchange_initialization(self, mock_binance, market_data_ingestion):
        """Test exchange initialization with mocked CCXT"""
        mock_exchange = AsyncMock()
        mock_binance.return_value = mock_exchange
        
        # Re-initialize to test with mock
        market_data_ingestion._initialize_exchanges()
        
        assert 'binance' in market_data_ingestion.exchanges
        assert market_data_ingestion.active_connections['binance'] is False

    @pytest.mark.asyncio
    async def test_connection_monitoring(self, market_data_ingestion):
        """Test connection health monitoring"""
        # Set up a stale connection
        exchange_id = 'binance'
        market_data_ingestion.active_connections[exchange_id] = True
        market_data_ingestion.last_ping[exchange_id] = time.time() - 35  # 35 seconds ago
        
        # Run monitoring (should detect stale connection)
        await market_data_ingestion._monitor_connections()
        
        # Connection should be marked as inactive
        assert market_data_ingestion.active_connections[exchange_id] is False
        assert market_data_ingestion.metrics['connection_drops'] > 0

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, market_data_ingestion):
        """Test graceful shutdown functionality"""
        market_data_ingestion.is_running = True
        
        # Mock exchange close method
        for exchange in market_data_ingestion.exchanges.values():
            exchange.close = AsyncMock()
        
        await market_data_ingestion.stop()
        
        assert market_data_ingestion.is_running is False
        
        # Verify all exchanges were closed
        for exchange in market_data_ingestion.exchanges.values():
            exchange.close.assert_called_once()


class TestDataStructures:
    """Test cases for data structure classes"""
    
    def test_ohlcv_data_structure(self):
        """Test OHLCV data structure"""
        data = OHLCVData(
            timestamp=int(time.time() * 1000),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0,
            exchange="binance",
            symbol="BTC/USDT"
        )
        
        assert data.timestamp > 0
        assert data.open < data.high
        assert data.low < data.close
        assert data.volume > 0
        assert data.exchange == "binance"
        assert data.symbol == "BTC/USDT"

    def test_orderbook_data_structure(self):
        """Test order book data structure"""
        data = OrderBookData(
            timestamp=int(time.time() * 1000),
            exchange="binance",
            symbol="BTC/USDT",
            bids=[(50000.0, 1.0), (49999.0, 2.0)],
            asks=[(50001.0, 1.5), (50002.0, 2.5)],
            spread=1.0
        )
        
        assert data.timestamp > 0
        assert len(data.bids) == 2
        assert len(data.asks) == 2
        assert data.spread == 1.0
        assert data.bids[0][0] < data.asks[0][0]  # Best bid < best ask

    def test_trade_data_structure(self):
        """Test trade data structure"""
        data = TradeData(
            timestamp=int(time.time() * 1000),
            exchange="binance",
            symbol="BTC/USDT",
            price=50000.0,
            quantity=0.1,
            side="buy"
        )
        
        assert data.timestamp > 0
        assert data.price > 0
        assert data.quantity > 0
        assert data.side in ["buy", "sell"]


@pytest.mark.integration
class TestMarketDataIntegration:
    """Integration tests for market data ingestion"""
    
    @pytest.mark.asyncio
    async def test_redis_integration(self, redis_client):
        """Test Redis integration end-to-end"""
        ingestion = MarketDataIngestion(redis_client)
        
        # Test storing and retrieving data
        test_data = OHLCVData(
            timestamp=int(time.time() * 1000),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0,
            exchange="test_exchange",
            symbol="BTC/USDT"
        )
        
        await ingestion._store_ohlcv_data(test_data)
        
        # Verify retrieval
        key = f"ohlcv:{test_data.exchange}:{test_data.symbol}:latest"
        stored_data = await redis_client.get(key)
        
        assert stored_data is not None
        parsed_data = json.loads(stored_data.decode('utf-8'))
        assert parsed_data['close'] == test_data.close


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
