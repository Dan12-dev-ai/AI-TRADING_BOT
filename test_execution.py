"""
Test suite for Execution Engine Module
Production-ready tests for execution engine and bad setup filter
"""

import pytest
import asyncio
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

import redis.asyncio as redis

from medallion_x.execution.execution_engine import (
    ExecutionEngine, ExchangeConnector, BadSetupFilter,
    OrderRequest, OrderResponse, OrderType, OrderSide, OrderStatus,
    ExecutionQuality, ExecutionMetrics
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
def exchange_config():
    """Create exchange configuration for testing"""
    from medallion_x.config.settings import ExchangeConfig
    return ExchangeConfig(
        api_key="test_key",
        secret="test_secret",
        testnet=True
    )


@pytest.fixture
def exchange_connector(exchange_config):
    """Create exchange connector for testing"""
    return ExchangeConnector("binance", exchange_config)


@pytest.fixture
def bad_setup_filter():
    """Create bad setup filter for testing"""
    return BadSetupFilter()


@pytest.fixture
def execution_engine(redis_client):
    """Create execution engine for testing"""
    return ExecutionEngine(redis_client)


@pytest.fixture
def sample_order_request():
    """Create sample order request for testing"""
    return OrderRequest(
        id=str(uuid.uuid4()),
        symbol="BTC/USDT",
        exchange="binance",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1,
        price=50000.0
    )


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing"""
    return {
        'volatility': 0.02,
        'spread': 0.0005,
        'liquidity': 50000.0,
        'volume': 1000000.0,
        'trend': 'neutral',
        'momentum': 0.01,
        'order_book_depth': 1000.0,
        'price_impact': 0.001
    }


class TestExchangeConnector:
    """Test cases for ExchangeConnector class"""
    
    def test_initialization(self, exchange_connector, exchange_config):
        """Test proper initialization of exchange connector"""
        assert exchange_connector.exchange_id == "binance"
        assert exchange_connector.exchange_config == exchange_config
        assert exchange_connector.is_connected is False
        assert exchange_connector.ccxt_exchange is None
        assert len(exchange_connector.order_book) == 0

    @pytest.mark.asyncio
    async def test_connect_success(self, exchange_connector):
        """Test successful exchange connection"""
        # Mock CCXT exchange
        with patch('ccxt.pro.binance') as mock_exchange_class:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets.return_value = None
            mock_exchange_class.return_value = mock_exchange
            
            success = await exchange_connector.connect()
            
            assert success is True
            assert exchange_connector.is_connected is True
            assert exchange_connector.ccxt_exchange == mock_exchange

    @pytest.mark.asyncio
    async def test_connect_failure(self, exchange_connector):
        """Test failed exchange connection"""
        # Mock CCXT exchange to raise exception
        with patch('ccxt.pro.binance') as mock_exchange_class:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets.side_effect = Exception("Connection failed")
            mock_exchange_class.return_value = mock_exchange
            
            success = await exchange_connector.connect()
            
            assert success is False
            assert exchange_connector.is_connected is False

    @pytest.mark.asyncio
    async def test_submit_market_order_success(self, exchange_connector, sample_order_request):
        """Test successful market order submission"""
        # Mock connected exchange
        with patch('ccxt.pro.binance') as mock_exchange_class:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets.return_value = None
            mock_exchange.create_market_order.return_value = {
                'id': 'exchange_order_123',
                'filled': 0.1,
                'remaining': 0.0,
                'price': 50000.0,
                'average': 50000.0,
                'status': 'closed',
                'fee': {'USDT': 5.0}
            }
            mock_exchange_class.return_value = mock_exchange
            
            await exchange_connector.connect()
            
            response = await exchange_connector.submit_order(sample_order_request)
            
            assert isinstance(response, OrderResponse)
            assert response.client_order_id == sample_order_request.id
            assert response.exchange_order_id == 'exchange_order_123'
            assert response.status == OrderStatus.FILLED
            assert response.filled_quantity == 0.1
            assert response.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_submit_limit_order_success(self, exchange_connector):
        """Test successful limit order submission"""
        limit_order = OrderRequest(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            exchange="binance",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=49000.0
        )
        
        # Mock connected exchange
        with patch('ccxt.pro.binance') as mock_exchange_class:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets.return_value = None
            mock_exchange.create_limit_order.return_value = {
                'id': 'limit_order_123',
                'filled': 0.0,
                'remaining': 0.1,
                'price': 49000.0,
                'status': 'open',
                'fee': {}
            }
            mock_exchange_class.return_value = mock_exchange
            
            await exchange_connector.connect()
            
            response = await exchange_connector.submit_order(limit_order)
            
            assert isinstance(response, OrderResponse)
            assert response.status == OrderStatus.SUBMITTED
            assert response.remaining_quantity == 0.1

    @pytest.mark.asyncio
    async def test_submit_order_failure(self, exchange_connector, sample_order_request):
        """Test order submission failure"""
        # Mock connected exchange with failure
        with patch('ccxt.pro.binance') as mock_exchange_class:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets.return_value = None
            mock_exchange.create_market_order.side_effect = Exception("Insufficient balance")
            mock_exchange_class.return_value = mock_exchange
            
            await exchange_connector.connect()
            
            response = await exchange_connector.submit_order(sample_order_request)
            
            assert isinstance(response, OrderResponse)
            assert response.status == OrderStatus.REJECTED
            assert response.error_message is not None
            assert "Insufficient balance" in response.error_message

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, exchange_connector):
        """Test successful order cancellation"""
        # Mock connected exchange
        with patch('ccxt.pro.binance') as mock_exchange_class:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets.return_value = None
            mock_exchange.cancel_order.return_value = {'status': 'canceled'}
            mock_exchange_class.return_value = mock_exchange
            
            await exchange_connector.connect()
            
            success = await exchange_connector.cancel_order('order_123', 'BTC/USDT')
            
            assert success is True

    @pytest.mark.asyncio
    async def test_cancel_order_failure(self, exchange_connector):
        """Test order cancellation failure"""
        # Mock connected exchange with failure
        with patch('ccxt.pro.binance') as mock_exchange_class:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets.return_value = None
            mock_exchange.cancel_order.side_effect = Exception("Order not found")
            mock_exchange_class.return_value = mock_exchange
            
            await exchange_connector.connect()
            
            success = await exchange_connector.cancel_order('order_123', 'BTC/USDT')
            
            assert success is False

    @pytest.mark.asyncio
    async def test_get_order_status(self, exchange_connector):
        """Test getting order status"""
        # Mock connected exchange
        with patch('ccxt.pro.binance') as mock_exchange_class:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets.return_value = None
            mock_exchange.fetch_order.return_value = {
                'id': 'order_123',
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'type': 'market',
                'amount': 0.1,
                'filled': 0.1,
                'remaining': 0.0,
                'price': 50000.0,
                'average': 50000.0,
                'status': 'closed',
                'timestamp': int(time.time() * 1000),
                'fee': {'USDT': 5.0}
            }
            mock_exchange_class.return_value = mock_exchange
            
            await exchange_connector.connect()
            
            status = await exchange_connector.get_order_status('order_123', 'BTC/USDT')
            
            assert isinstance(status, OrderResponse)
            assert status.exchange_order_id == 'order_123'
            assert status.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_get_balance(self, exchange_connector):
        """Test getting account balance"""
        # Mock connected exchange
        with patch('ccxt.pro.binance') as mock_exchange_class:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets.return_value = None
            mock_exchange.fetch_balance.return_value = {
                'total': {'USDT': 10000.0, 'BTC': 0.5}
            }
            mock_exchange_class.return_value = mock_exchange
            
            await exchange_connector.connect()
            
            balance = await exchange_connector.get_balance()
            
            assert isinstance(balance, dict)
            assert 'USDT' in balance
            assert balance['USDT'] == 10000.0

    def test_performance_metrics(self, exchange_connector):
        """Test performance metrics tracking"""
        # Simulate some orders
        exchange_connector.performance_metrics['orders_submitted'] = 10
        exchange_connector.performance_metrics['orders_filled'] = 8
        exchange_connector.performance_metrics['average_latency_ms'] = 25.0
        
        metrics = exchange_connector.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert 'orders_submitted' in metrics
        assert 'average_latency_ms' in metrics

    @pytest.mark.asyncio
    async def test_disconnect(self, exchange_connector):
        """Test exchange disconnection"""
        # Mock connected exchange
        with patch('ccxt.pro.binance') as mock_exchange_class:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets.return_value = None
            mock_exchange.close.return_value = None
            mock_exchange_class.return_value = mock_exchange
            
            await exchange_connector.connect()
            assert exchange_connector.is_connected is True
            
            await exchange_connector.disconnect()
            assert exchange_connector.is_connected is False


class TestBadSetupFilter:
    """Test cases for BadSetupFilter class"""
    
    def test_initialization(self, bad_setup_filter):
        """Test proper initialization of bad setup filter"""
        assert bad_setup_filter.min_confidence_threshold == 0.95
        assert bad_setup_filter.max_risk_score == 0.3
        assert bad_setup_filter.volatility_threshold == 0.05
        assert bad_setup_filter.spread_threshold == 0.001
        assert bad_setup_filter.liquidity_threshold == 10000

    @pytest.mark.asyncio
    async def test_analyze_setup_perfect_conditions(self, bad_setup_filter, sample_order_request, sample_market_data):
        """Test setup analysis with perfect market conditions"""
        # Perfect market conditions
        perfect_market_data = {
            'volatility': 0.02,  # Normal volatility
            'spread': 0.0005,   # Tight spread
            'liquidity': 100000,  # High liquidity
            'volume': 10000000,  # High volume
            'trend': 'neutral',
            'momentum': 0.01,
            'order_book_depth': 10000,
            'price_impact': 0.0001
        }
        
        result = await bad_setup_filter.analyze_setup(sample_order_request, perfect_market_data)
        
        assert result.passed is True
        assert result.confidence >= 0.95
        assert result.risk_score <= 0.3
        assert result.recommendation == "EXECUTE"

    @pytest.mark.asyncio
    async def test_analyze_setup_high_volatility(self, bad_setup_filter, sample_order_request):
        """Test setup analysis with high volatility"""
        # High volatility conditions
        high_vol_data = {
            'volatility': 0.08,  # High volatility
            'spread': 0.0005,
            'liquidity': 100000,
            'volume': 10000000,
            'trend': 'neutral',
            'momentum': 0.01,
            'order_book_depth': 10000,
            'price_impact': 0.0001
        }
        
        result = await bad_setup_filter.analyze_setup(sample_order_request, high_vol_data)
        
        assert result.passed is False
        assert "High volatility" in ", ".join(result.reasons)
        assert result.recommendation == "AVOID"

    @pytest.mark.asyncio
    async def test_analyze_setup_low_liquidity(self, bad_setup_filter, sample_order_request):
        """Test setup analysis with low liquidity"""
        # Low liquidity conditions
        low_liquidity_data = {
            'volatility': 0.02,
            'spread': 0.0005,
            'liquidity': 5000,  # Low liquidity
            'volume': 100000,
            'trend': 'neutral',
            'momentum': 0.01,
            'order_book_depth': 1000,
            'price_impact': 0.001
        }
        
        result = await bad_setup_filter.analyze_setup(sample_order_request, low_liquidity_data)
        
        assert result.passed is False
        assert "Low liquidity" in ", ".join(result.reasons)
        assert result.recommendation == "AVOID"

    @pytest.mark.asyncio
    async def test_analyze_setup_wide_spread(self, bad_setup_filter, sample_order_request):
        """Test setup analysis with wide spread"""
        # Wide spread conditions
        wide_spread_data = {
            'volatility': 0.02,
            'spread': 0.002,  # Wide spread
            'liquidity': 100000,
            'volume': 10000000,
            'trend': 'neutral',
            'momentum': 0.01,
            'order_book_depth': 10000,
            'price_impact': 0.0001
        }
        
        result = await bad_setup_filter.analyze_setup(sample_order_request, wide_spread_data)
        
        assert result.passed is False
        assert "Wide spread" in ", ".join(result.reasons)
        assert result.recommendation == "AVOID"

    @pytest.mark.asyncio
    async def test_analyze_setup_large_order_size(self, bad_setup_filter):
        """Test setup analysis with large order size"""
        # Large order
        large_order = OrderRequest(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            exchange="binance",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10.0,  # Large order
            price=50000.0
        )
        
        # Normal market conditions
        market_data = {
            'volatility': 0.02,
            'spread': 0.0005,
            'liquidity': 50000,  # Moderate liquidity
            'volume': 1000000,
            'trend': 'neutral',
            'momentum': 0.01,
            'order_book_depth': 10000,
            'price_impact': 0.0001
        }
        
        result = await bad_setup_filter.analyze_setup(large_order, market_data)
        
        assert result.passed is False
        assert "Large order size" in ", ".join(result.reasons)

    @pytest.mark.asyncio
    async def test_analyze_setup_adverse_timing(self, bad_setup_filter, sample_order_request):
        """Test setup analysis with adverse timing"""
        # Adverse timing conditions
        adverse_timing_data = {
            'volatility': 0.02,
            'spread': 0.0005,
            'liquidity': 100000,
            'volume': 10000000,
            'trend': 'strong_down',  # Strong downtrend
            'momentum': -0.2,  # Negative momentum
            'order_book_depth': 10000,
            'price_impact': 0.0001
        }
        
        result = await bad_setup_filter.analyze_setup(sample_order_request, adverse_timing_data)
        
        assert result.passed is False
        assert "Strong downtrend" in ", ".join(result.reasons)

    def test_check_volatility(self, bad_setup_filter):
        """Test volatility check function"""
        market_analysis = {'volatility': 0.03}
        result = bad_setup_filter._check_volatility(market_analysis)
        assert result['passed'] is True
        
        market_analysis = {'volatility': 0.08}
        result = bad_setup_filter._check_volatility(market_analysis)
        assert result['passed'] is False

    def test_check_liquidity(self, bad_setup_filter):
        """Test liquidity check function"""
        market_analysis = {'liquidity': 50000, 'volume': 1000000}
        result = bad_setup_filter._check_liquidity(market_analysis)
        assert result['passed'] is True
        
        market_analysis = {'liquidity': 5000, 'volume': 100000}
        result = bad_setup_filter._check_liquidity(market_analysis)
        assert result['passed'] is False

    def test_check_spread(self, bad_setup_filter):
        """Test spread check function"""
        market_analysis = {'spread': 0.0005}
        result = bad_setup_filter._check_spread(market_analysis)
        assert result['passed'] is True
        
        market_analysis = {'spread': 0.002}
        result = bad_setup_filter._check_spread(market_analysis)
        assert result['passed'] is False

    def test_check_order_size(self, bad_setup_filter, sample_order_request):
        """Test order size check function"""
        market_analysis = {'liquidity': 1000000}
        result = bad_setup_filter._check_order_size(sample_order_request, market_analysis)
        assert result['passed'] is True
        
        market_analysis = {'liquidity': 10000}
        result = bad_setup_filter._check_order_size(sample_order_request, market_analysis)
        assert result['passed'] is False

    def test_get_recommendation(self, bad_setup_filter):
        """Test recommendation function"""
        # Execute recommendation
        result = bad_setup_filter._get_recommendation(True, 0.98, 0.1)
        assert result == "EXECUTE"
        
        # Wait recommendation
        result = bad_setup_filter._get_recommendation(True, 0.96, 0.25)
        assert result == "WAIT"
        
        # Avoid recommendation
        result = bad_setup_filter._get_recommendation(False, 0.5, 0.8)
        assert result == "AVOID"

    def test_get_metrics(self, bad_setup_filter):
        """Test getting filter metrics"""
        # Simulate some analysis
        bad_setup_filter.metrics['setups_analyzed'] = 100
        bad_setup_filter.metrics['setups_approved'] = 80
        bad_setup_filter.metrics['setups_rejected'] = 20
        
        metrics = bad_setup_filter.get_metrics()
        
        assert isinstance(metrics, dict)
        assert 'approval_rate' in metrics
        assert 'rejection_rate' in metrics
        assert metrics['approval_rate'] == 0.8
        assert metrics['rejection_rate'] == 0.2


class TestExecutionEngine:
    """Test cases for ExecutionEngine class"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, execution_engine):
        """Test proper initialization of execution engine"""
        assert execution_engine.redis_client is not None
        assert isinstance(execution_engine.bad_setup_filter, BadSetupFilter)
        assert len(execution_engine.exchange_connectors) == 0
        assert len(execution_engine.active_orders) == 0
        assert isinstance(execution_engine.metrics, ExecutionMetrics)

    @pytest.mark.asyncio
    async def test_start(self, execution_engine):
        """Test starting execution engine"""
        # Mock exchange initialization
        with patch.object(execution_engine, '_initialize_exchanges') as mock_init:
            mock_init.return_value = None
            
            await execution_engine.start()
            
            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_order_success(self, execution_engine, sample_order_request, sample_market_data):
        """Test successful order submission"""
        # Mock exchange connector
        mock_connector = AsyncMock()
        mock_connector.is_connected = True
        mock_connector.ccxt_exchange.symbols = ["BTC/USDT"]
        mock_connector.submit_order.return_value = OrderResponse(
            id=str(uuid.uuid4()),
            client_order_id=sample_order_request.id,
            exchange_order_id="exchange_123",
            symbol=sample_order_request.symbol,
            exchange="binance",
            side=sample_order_request.side,
            order_type=sample_order_request.order_type,
            quantity=sample_order_request.quantity,
            filled_quantity=sample_order_request.quantity,
            remaining_quantity=0.0,
            price=sample_order_request.price,
            average_price=sample_order_request.price,
            status=OrderStatus.FILLED,
            timestamp=int(time.time() * 1000),
            execution_time_ms=15.0
        )
        
        execution_engine.exchange_connectors["binance"] = mock_connector
        
        # Mock exchange selection
        with patch.object(execution_engine, '_select_best_exchange', return_value="binance"):
            response = await execution_engine.submit_order(sample_order_request, sample_market_data)
        
        assert isinstance(response, OrderResponse)
        assert response.status == OrderStatus.FILLED
        assert response.client_order_id == sample_order_request.id

    @pytest.mark.asyncio
    async def test_submit_order_rejected_by_filter(self, execution_engine, sample_order_request):
        """Test order submission rejected by bad setup filter"""
        # Bad market conditions
        bad_market_data = {
            'volatility': 0.1,  # Very high volatility
            'spread': 0.005,   # Wide spread
            'liquidity': 1000,  # Low liquidity
            'volume': 10000,
            'trend': 'strong_down',
            'momentum': -0.3,
            'order_book_depth': 100,
            'price_impact': 0.01
        }
        
        response = await execution_engine.submit_order(sample_order_request, bad_market_data)
        
        assert isinstance(response, OrderResponse)
        assert response.status == OrderStatus.REJECTED
        assert "Bad setup filter" in response.error_message

    @pytest.mark.asyncio
    async def test_submit_order_no_exchange_available(self, execution_engine, sample_order_request, sample_market_data):
        """Test order submission with no exchange available"""
        # No exchanges configured
        response = await execution_engine.submit_order(sample_order_request, sample_market_data)
        
        assert isinstance(response, OrderResponse)
        assert response.status == OrderStatus.REJECTED
        assert "No suitable exchange" in response.error_message

    @pytest.mark.asyncio
    async def test_select_best_exchange(self, execution_engine):
        """Test exchange selection logic"""
        # Mock exchange connectors
        connector1 = AsyncMock()
        connector1.is_connected = True
        connector1.ccxt_exchange.symbols = ["BTC/USDT"]
        connector1.performance_metrics = {'average_latency_ms': 20, 'error_rate': 0.01, 'orders_submitted': 100, 'orders_filled': 95}
        
        connector2 = AsyncMock()
        connector2.is_connected = True
        connector2.ccxt_exchange.symbols = ["BTC/USDT"]
        connector2.performance_metrics = {'average_latency_ms': 30, 'error_rate': 0.02, 'orders_submitted': 100, 'orders_filled': 90}
        
        execution_engine.exchange_connectors = {
            "binance": connector1,
            "bybit": connector2
        }
        
        best_exchange = await execution_engine._select_best_exchange("BTC/USDT", OrderSide.BUY)
        
        assert best_exchange == "binance"  # Should select the better performing exchange

    @pytest.mark.asyncio
    async def test_calculate_exchange_score(self, execution_engine):
        """Test exchange score calculation"""
        mock_connector = MagicMock()
        mock_connector.performance_metrics = {
            'average_latency_ms': 25,
            'error_rate': 0.01,
            'orders_submitted': 100,
            'orders_filled': 95
        }
        
        score = execution_engine._calculate_exchange_score(mock_connector, "BTC/USDT", OrderSide.BUY)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_cancel_order(self, execution_engine):
        """Test order cancellation"""
        # Create active order
        order_id = str(uuid.uuid4())
        active_order = OrderResponse(
            id=order_id,
            client_order_id="client_123",
            exchange_order_id="exchange_123",
            symbol="BTC/USDT",
            exchange="binance",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
            filled_quantity=0.0,
            remaining_quantity=0.1,
            price=50000.0,
            average_price=None,
            status=OrderStatus.SUBMITTED,
            timestamp=int(time.time() * 1000),
            execution_time_ms=15.0
        )
        
        execution_engine.active_orders[order_id] = active_order
        
        # Mock exchange connector
        mock_connector = AsyncMock()
        mock_connector.cancel_order.return_value = True
        execution_engine.exchange_connectors["binance"] = mock_connector
        
        success = await execution_engine.cancel_order(order_id)
        
        assert success is True
        assert order_id not in execution_engine.active_orders

    @pytest.mark.asyncio
    async def test_get_order_status(self, execution_engine):
        """Test getting order status"""
        # Create order in Redis
        order_id = str(uuid.uuid4())
        order_data = {
            'id': order_id,
            'client_order_id': 'client_123',
            'exchange_order_id': 'exchange_123',
            'symbol': 'BTC/USDT',
            'exchange': 'binance',
            'side': 'buy',
            'order_type': 'market',
            'quantity': 0.1,
            'filled_quantity': 0.1,
            'remaining_quantity': 0.0,
            'price': 50000.0,
            'average_price': 50000.0,
            'status': 'filled',
            'timestamp': int(time.time() * 1000),
            'execution_time_ms': 15.0,
            'fee': {'USDT': 5.0}
        }
        
        await execution_engine.redis_client.setex(
            f"orders:{order_id}",
            ttl=86400,
            value=str(order_data).replace("'", '"')
        )
        
        status = await execution_engine.get_order_status(order_id)
        
        assert isinstance(status, OrderResponse)
        assert status.id == order_id
        assert status.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_get_portfolio_balance(self, execution_engine):
        """Test getting portfolio balance"""
        # Mock exchange connectors
        mock_connector1 = AsyncMock()
        mock_connector1.get_balance.return_value = {'USDT': 10000.0, 'BTC': 0.5}
        
        mock_connector2 = AsyncMock()
        mock_connector2.get_balance.return_value = {'USDT': 5000.0, 'ETH': 2.0}
        
        execution_engine.exchange_connectors = {
            "binance": mock_connector1,
            "bybit": mock_connector2
        }
        
        balances = await execution_engine.get_portfolio_balance()
        
        assert isinstance(balances, dict)
        assert "binance" in balances
        assert "bybit" in balances
        assert balances["binance"]["USDT"] == 10000.0
        assert balances["bybit"]["USDT"] == 5000.0

    def test_get_execution_metrics(self, execution_engine):
        """Test getting execution metrics"""
        metrics = execution_engine.get_execution_metrics()
        
        assert isinstance(metrics, ExecutionMetrics)
        assert 'total_orders' in metrics
        assert 'successful_orders' in metrics
        assert 'average_execution_time_ms' in metrics

    def test_get_filter_metrics(self, execution_engine):
        """Test getting filter metrics"""
        metrics = execution_engine.get_filter_metrics()
        
        assert isinstance(metrics, dict)
        assert 'setups_analyzed' in metrics
        assert 'approval_rate' in metrics

    @pytest.mark.asyncio
    async def test_stop(self, execution_engine):
        """Test stopping execution engine"""
        # Mock exchange connectors
        mock_connector = AsyncMock()
        execution_engine.exchange_connectors = {"binance": mock_connector}
        
        await execution_engine.stop()
        
        mock_connector.disconnect.assert_called_once()


@pytest.mark.integration
class TestExecutionIntegration:
    """Integration tests for execution module"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_order_execution(self, redis_client):
        """Test end-to-end order execution pipeline"""
        # Create execution engine
        execution_engine = ExecutionEngine(redis_client)
        
        # Create order request
        order_request = OrderRequest(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            exchange="binance",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
            price=50000.0
        )
        
        # Create market data
        market_data = {
            'volatility': 0.02,
            'spread': 0.0005,
            'liquidity': 100000,
            'volume': 1000000,
            'trend': 'neutral',
            'momentum': 0.01,
            'order_book_depth': 10000,
            'price_impact': 0.0001
        }
        
        # Mock exchange connector
        mock_connector = AsyncMock()
        mock_connector.is_connected = True
        mock_connector.ccxt_exchange.symbols = ["BTC/USDT"]
        mock_connector.submit_order.return_value = OrderResponse(
            id=str(uuid.uuid4()),
            client_order_id=order_request.id,
            exchange_order_id="exchange_123",
            symbol=order_request.symbol,
            exchange="binance",
            side=order_request.side,
            order_type=order_request.order_type,
            quantity=order_request.quantity,
            filled_quantity=order_request.quantity,
            remaining_quantity=0.0,
            price=order_request.price,
            average_price=order_request.price,
            status=OrderStatus.FILLED,
            timestamp=int(time.time() * 1000),
            execution_time_ms=15.0
        )
        
        execution_engine.exchange_connectors["binance"] = mock_connector
        
        # Mock exchange selection
        with patch.object(execution_engine, '_select_best_exchange', return_value="binance"):
            response = await execution_engine.submit_order(order_request, market_data)
        
        # Verify pipeline worked
        assert isinstance(response, OrderResponse)
        assert response.status == OrderStatus.FILLED
        assert response.client_order_id == order_request.id
        assert execution_engine.metrics.total_orders == 1
        assert execution_engine.metrics.successful_orders == 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
