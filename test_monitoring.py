"""
Test suite for Monitoring Module
Production-ready tests for FastAPI dashboard and Telegram NLP
"""

import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

import redis.asyncio as redis

from medallion_x.monitoring.fastapi_dashboard import (
    FastAPIDashboard, WebSocketManager, MetricsCollector, AlertManager,
    SystemMetrics, TradingMetrics, Alert, AlertLevel, SystemStatus
)
from medallion_x.monitoring.telegram_nlp import (
    TelegramBot, NLPProcessor, TelegramCommand, NLPIntent, BotResponse,
    CommandCategory, CommandPermission
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
def websocket_manager():
    """Create WebSocket manager for testing"""
    return WebSocketManager()


@pytest.fixture
def metrics_collector(redis_client):
    """Create metrics collector for testing"""
    return MetricsCollector(redis_client)


@pytest.fixture
def alert_manager(redis_client):
    """Create alert manager for testing"""
    return AlertManager(redis_client)


@pytest.fixture
def fastapi_dashboard(redis_client):
    """Create FastAPI dashboard for testing"""
    return FastAPIDashboard(redis_client)


@pytest.fixture
def nlp_processor():
    """Create NLP processor for testing"""
    return NLPProcessor()


@pytest.fixture
def telegram_bot(redis_client):
    """Create Telegram bot for testing"""
    return TelegramBot(redis_client)


class TestWebSocketManager:
    """Test cases for WebSocketManager class"""
    
    def test_initialization(self, websocket_manager):
        """Test proper initialization of WebSocket manager"""
        assert len(websocket_manager.active_connections) == 0
        assert len(websocket_manager.connection_metadata) == 0

    def test_get_connection_count(self, websocket_manager):
        """Test getting connection count"""
        assert websocket_manager.get_connection_count() == 0

    def test_disconnect_nonexistent_connection(self, websocket_manager):
        """Test disconnecting non-existent connection"""
        # Should not raise error
        websocket_manager.disconnect(None)


class TestMetricsCollector:
    """Test cases for MetricsCollector class"""
    
    def test_initialization(self, metrics_collector, redis_client):
        """Test proper initialization of metrics collector"""
        assert metrics_collector.redis_client == redis_client
        assert metrics_collector.collection_interval == 5
        assert len(metrics_collector.system_metrics_history) == 0
        assert len(metrics_collector.trading_metrics_history) == 0

    @pytest.mark.asyncio
    async def test_collect_system_metrics(self, metrics_collector):
        """Test system metrics collection"""
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_io_counters') as mock_network, \
             patch('psutil.net_connections', return_value=10), \
             patch('psutil.boot_time', return_value=time.time() - 86400):
            
            # Mock memory
            mock_memory.return_value.percent = 60.0
            
            # Mock disk
            mock_disk.return_value.used = 1000000
            mock_disk.return_value.total = 2000000
            
            # Mock network
            mock_network.return_value.bytes_sent = 1000
            mock_network.return_value.bytes_recv = 2000
            mock_network.return_value.packets_sent = 10
            mock_network.return_value.packets_recv = 20
            
            metrics = await metrics_collector.collect_system_metrics()
            
            assert isinstance(metrics, SystemMetrics)
            assert metrics.cpu_usage == 50.0
            assert metrics.memory_usage == 60.0
            assert metrics.disk_usage == 50.0
            assert metrics.active_connections == 10
            assert metrics.uptime_seconds > 0

    @pytest.mark.asyncio
    async def test_collect_trading_metrics(self, metrics_collector):
        """Test trading metrics collection"""
        # Mock trading data in Redis
        trading_data = {
            'total_trades': 100,
            'successful_trades': 65,
            'failed_trades': 35,
            'total_pnl': 2500.0,
            'daily_pnl': 150.0,
            'win_rate': 0.65,
            'average_trade_duration': 14400.0,  # 4 hours
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.05,
            'current_positions': 3,
            'portfolio_value': 25000.0,
            'risk_exposure': 0.15
        }
        
        await metrics_collector.redis_client.set(
            "trading:metrics:latest",
            value=json.dumps(trading_data)
        )
        
        metrics = await metrics_collector.collect_trading_metrics()
        
        assert isinstance(metrics, TradingMetrics)
        assert metrics.total_trades == 100
        assert metrics.successful_trades == 65
        assert metrics.total_pnl == 2500.0
        assert metrics.win_rate == 0.65

    @pytest.mark.asyncio
    async def test_get_metrics_history(self, metrics_collector):
        """Test getting metrics history"""
        # Store some test data
        system_metrics = SystemMetrics(
            timestamp=int(time.time() * 1000),
            cpu_usage=50.0,
            memory_usage=60.0,
            disk_usage=70.0,
            network_io={'bytes_sent': 1000, 'bytes_recv': 2000},
            active_connections=10,
            total_requests=100,
            error_rate=0.01,
            average_response_time=100.0,
            uptime_seconds=86400
        )
        
        await metrics_collector._store_system_metrics(system_metrics)
        
        history = await metrics_collector.get_metrics_history("system", 10)
        
        assert isinstance(history, list)
        assert len(history) >= 1
        assert history[0]['cpu_usage'] == 50.0

    def test_get_performance_metrics(self, metrics_collector):
        """Test getting performance metrics"""
        metrics = metrics_collector.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert 'collections_performed' in metrics
        assert 'collection_errors' in metrics
        assert 'average_collection_time_ms' in metrics


class TestAlertManager:
    """Test cases for AlertManager class"""
    
    def test_initialization(self, alert_manager, redis_client):
        """Test proper initialization of alert manager"""
        assert alert_manager.redis_client == redis_client
        assert len(alert_manager.active_alerts) == 0
        assert alert_manager.thresholds['cpu_usage'] == 80.0
        assert alert_manager.thresholds['memory_usage'] == 85.0

    @pytest.mark.asyncio
    async def test_check_system_alerts(self, alert_manager):
        """Test system alert checking"""
        # Create metrics that should trigger alerts
        high_cpu_metrics = SystemMetrics(
            timestamp=int(time.time() * 1000),
            cpu_usage=90.0,  # Above threshold
            memory_usage=70.0,
            disk_usage=60.0,
            network_io={'bytes_sent': 1000, 'bytes_recv': 2000},
            active_connections=10,
            total_requests=100,
            error_rate=0.01,
            average_response_time=100.0,
            uptime_seconds=86400
        )
        
        alerts = await alert_manager.check_system_alerts(high_cpu_metrics)
        
        assert len(alerts) > 0
        assert any("High CPU usage" in alert.message for alert in alerts)
        assert all(alert.level == AlertLevel.WARNING for alert in alerts)

    @pytest.mark.asyncio
    async def test_check_trading_alerts(self, alert_manager):
        """Test trading alert checking"""
        # Create metrics that should trigger alerts
        high_drawdown_metrics = TradingMetrics(
            timestamp=int(time.time() * 1000),
            total_trades=100,
            successful_trades=65,
            failed_trades=35,
            total_pnl=2500.0,
            daily_pnl=150.0,
            win_rate=0.65,
            average_trade_duration=14400.0,
            sharpe_ratio=1.5,
            max_drawdown=-0.15,  # Below threshold
            current_positions=3,
            portfolio_value=25000.0,
            risk_exposure=0.15
        )
        
        alerts = await alert_manager.check_trading_alerts(high_drawdown_metrics)
        
        assert len(alerts) > 0
        assert any("High drawdown" in alert.message for alert in alerts)
        assert all(alert.level == AlertLevel.ERROR for alert in alerts)

    @pytest.mark.asyncio
    async def test_create_alert(self, alert_manager):
        """Test alert creation"""
        alert = await alert_manager._create_alert(
            AlertLevel.WARNING,
            "test",
            "Test alert message",
            {"test": "data"}
        )
        
        assert isinstance(alert, Alert)
        assert alert.level == AlertLevel.WARNING
        assert alert.source == "test"
        assert alert.message == "Test alert message"
        assert alert.resolved is False
        assert alert.id in alert_manager.active_alerts

    @pytest.mark.asyncio
    async def test_resolve_alert(self, alert_manager):
        """Test alert resolution"""
        # Create an alert first
        alert = await alert_manager._create_alert(
            AlertLevel.WARNING,
            "test",
            "Test alert message",
            {"test": "data"}
        )
        alert_id = alert.id
        
        # Resolve the alert
        success = await alert_manager.resolve_alert(alert_id)
        
        assert success is True
        assert alert_id not in alert_manager.active_alerts

    @pytest.mark.asyncio
    async def test_get_active_alerts(self, alert_manager):
        """Test getting active alerts"""
        # Create some alerts
        await alert_manager._create_alert(
            AlertLevel.WARNING,
            "test1",
            "Test alert 1",
            {"test": "data1"}
        )
        await alert_manager._create_alert(
            AlertLevel.ERROR,
            "test2",
            "Test alert 2",
            {"test": "data2"}
        )
        
        alerts = await alert_manager.get_active_alerts()
        
        assert isinstance(alerts, list)
        assert len(alerts) == 2
        assert all(not alert.resolved for alert in alerts)


class TestNLPProcessor:
    """Test cases for NLPProcessor class"""
    
    def test_initialization(self, nlp_processor):
        """Test proper initialization of NLP processor"""
        assert nlp_processor.command_patterns is not None
        assert 'buy' in nlp_processor.command_patterns
        assert 'sell' in nlp_processor.command_patterns
        assert nlp_processor.entity_types is not None

    def test_clean_text(self, nlp_processor):
        """Test text cleaning"""
        dirty_text = "  BUY    BTC/USDT   0.1!!!  "
        clean_text = nlp_processor._clean_text(dirty_text)
        
        assert clean_text == "buy btc/usdt 0.1"

    def test_extract_entities(self, nlp_processor):
        """Test entity extraction"""
        text = "Buy 0.1 BTC/USDT at $50000 with 2% risk"
        entities = nlp_processor._extract_entities(text)
        
        assert 'SYMBOL' in entities
        assert 'AMOUNT' in entities
        assert 'PERCENTAGE' in entities
        assert 'CURRENCY' in entities
        assert entities['SYMBOL'] == ['BTC/USDT']
        assert entities['AMOUNT'] == ['0.1']

    def test_classify_intent(self, nlp_processor):
        """Test intent classification"""
        text = "buy 0.1 btc/usdt"
        entities = nlp_processor._extract_entities(text)
        
        intent = nlp_processor._classify_intent(text, entities)
        
        assert intent.intent == "buy"
        assert intent.confidence > 0.0
        assert intent.command == "buy"
        assert 'symbol' in intent.parameters
        assert 'amount' in intent.parameters

    def test_classify_intent_sell(self, nlp_processor):
        """Test sell intent classification"""
        text = "sell 0.05 eth/usdt"
        entities = nlp_processor._extract_entities(text)
        
        intent = nlp_processor._classify_intent(text, entities)
        
        assert intent.intent == "sell"
        assert intent.command == "sell"
        assert intent.parameters.get('symbol') == "eth/usdt"
        assert intent.parameters.get('amount') == "0.05"

    def test_classify_intent_status(self, nlp_processor):
        """Test status intent classification"""
        text = "what is the current status?"
        entities = nlp_processor._extract_entities(text)
        
        intent = nlp_processor._classify_intent(text, entities)
        
        assert intent.intent == "status"
        assert intent.command == "status"

    def test_get_parameter_names(self, nlp_processor):
        """Test parameter name mapping"""
        param_names = nlp_processor._get_parameter_names("buy")
        assert param_names == ['symbol', 'amount']
        
        param_names = nlp_processor._get_parameter_names("status")
        assert param_names == []

    def test_validate_intent(self, nlp_processor):
        """Test intent validation"""
        # Valid intent
        valid_intent = NLPIntent(
            intent="buy",
            confidence=0.9,
            entities={},
            command="buy",
            parameters={"symbol": "BTC/USDT", "amount": "0.1"}
        )
        
        validated = nlp_processor._validate_intent(valid_intent, 123456789)
        assert validated.intent == "buy"
        assert validated.parameters['symbol'] == "BTC/USDT"
        
        # Invalid symbol
        invalid_intent = NLPIntent(
            intent="buy",
            confidence=0.9,
            entities={},
            command="buy",
            parameters={"symbol": "INVALID", "amount": "0.1"}
        )
        
        validated = nlp_processor._validate_intent(invalid_intent, 123456789)
        assert validated.intent == "invalid_symbol"

    @pytest.mark.asyncio
    async def test_process_message(self, nlp_processor):
        """Test message processing"""
        message_text = "buy 0.1 btc/usdt"
        user_id = 123456789
        
        intent = await nlp_processor.process_message(message_text, user_id)
        
        assert isinstance(intent, NLPIntent)
        assert intent.intent == "buy"
        assert intent.confidence > 0.0


class TestTelegramBot:
    """Test cases for TelegramBot class"""
    
    def test_initialization(self, telegram_bot, redis_client):
        """Test proper initialization of Telegram bot"""
        assert telegram_bot.redis_client == redis_client
        assert telegram_bot.nlp_processor is not None
        assert len(telegram_bot.commands) > 0
        assert len(telegram_bot.allowed_users) > 0

    def test_load_allowed_users(self, telegram_bot):
        """Test loading allowed users"""
        users = telegram_bot._load_allowed_users()
        assert isinstance(users, list)
        assert len(users) > 0

    def test_initialize_commands(self, telegram_bot):
        """Test command initialization"""
        commands = telegram_bot.commands
        
        assert 'buy' in commands
        assert 'sell' in commands
        assert 'status' in commands
        assert 'help' in commands
        
        # Check command structure
        buy_command = commands['buy']
        assert isinstance(buy_command, TelegramCommand)
        assert buy_command.category == CommandCategory.TRADING
        assert buy_command.permission == CommandPermission.TRADER

    def test_check_permissions(self, telegram_bot):
        """Test permission checking"""
        # Allowed user
        allowed_user_id = telegram_bot.allowed_users[0]
        assert telegram_bot._check_permissions(allowed_user_id, CommandPermission.ADMIN) is True
        
        # Non-allowed user
        assert telegram_bot._check_permissions(999999999, CommandPermission.READ_ONLY) is False

    def test_get_parameter_names(self, telegram_bot):
        """Test parameter name mapping"""
        param_names = telegram_bot.commands['buy'].parameters
        assert 'symbol' in param_names
        assert 'amount' in param_names

    @pytest.mark.asyncio
    async def test_handle_buy_command(self, telegram_bot):
        """Test buy command handler"""
        # Mock update
        mock_update = MagicMock()
        mock_update.message.reply_text = AsyncMock()
        
        context = {
            'update': mock_update,
            'parameters': {'symbol': 'BTC/USDT', 'amount': '0.1'},
            'command_name': 'buy'
        }
        
        await telegram_bot.handle_buy(context)
        
        # Check that reply was called
        mock_update.message.reply_text.assert_called_once()
        
        # Check reply content
        call_args = mock_update.message.reply_text.call_args
        assert 'Buy Order Placed' in call_args[0][0]

    @pytest.mark.asyncio
    async def test_handle_sell_command(self, telegram_bot):
        """Test sell command handler"""
        mock_update = MagicMock()
        mock_update.message.reply_text = AsyncMock()
        
        context = {
            'update': mock_update,
            'parameters': {'symbol': 'BTC/USDT', 'amount': '0.1'},
            'command_name': 'sell'
        }
        
        await telegram_bot.handle_sell(context)
        
        mock_update.message.reply_text.assert_called_once()
        assert 'Sell Order Placed' in mock_update.message.reply_text.call_args[0][0]

    @pytest.mark.asyncio
    async def test_handle_status_command(self, telegram_bot):
        """Test status command handler"""
        mock_update = MagicMock()
        mock_update.message.reply_text = AsyncMock()
        
        context = {
            'update': mock_update,
            'parameters': {},
            'command_name': 'status'
        }
        
        await telegram_bot.handle_status(context)
        
        mock_update.message.reply_text.assert_called_once()
        assert 'System Status' in mock_update.message.reply_text.call_args[0][0]

    @pytest.mark.asyncio
    async def test_handle_help_command(self, telegram_bot):
        """Test help command handler"""
        mock_update = MagicMock()
        mock_update.message.reply_text = AsyncMock()
        
        context = {
            'update': mock_update,
            'parameters': {},
            'command_name': 'help'
        }
        
        await telegram_bot.handle_help(context)
        
        mock_update.message.reply_text.assert_called_once()
        assert 'Medallion-X Bot Commands' in mock_update.message.reply_text.call_args[0][0]

    @pytest.mark.asyncio
    async def test_handle_natural_language(self, telegram_bot):
        """Test natural language handling"""
        # Mock update with authorized user
        mock_update = MagicMock()
        mock_update.effective_user.id = telegram_bot.allowed_users[0]
        mock_update.message.text = "buy 0.1 btc/usdt"
        mock_update.message.reply_text = AsyncMock()
        
        # Mock context
        mock_context = MagicMock()
        
        with patch.object(telegram_bot, '_execute_command') as mock_execute:
            await telegram_bot.handle_natural_language(mock_update, mock_context)
            
            # Check that command was executed
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_natural_language_unauthorized(self, telegram_bot):
        """Test natural language handling with unauthorized user"""
        # Mock update with unauthorized user
        mock_update = MagicMock()
        mock_update.effective_user.id = 999999999  # Not in allowed users
        mock_update.message.reply_text = AsyncMock()
        
        # Mock context
        mock_context = MagicMock()
        
        await telegram_bot.handle_natural_language(mock_update, mock_context)
        
        # Check that unauthorized message was sent
        mock_update.message.reply_text.assert_called_once()
        assert 'not authorized' in mock_update.message.reply_text.call_args[0][0]

    def test_get_metrics(self, telegram_bot):
        """Test getting bot metrics"""
        metrics = telegram_bot.get_metrics()
        
        assert isinstance(metrics, dict)
        assert 'messages_processed' in metrics
        assert 'commands_executed' in metrics
        assert 'total_commands' in metrics
        assert 'active_users' in metrics


class TestFastAPIDashboard:
    """Test cases for FastAPIDashboard class"""
    
    def test_initialization(self, fastapi_dashboard, redis_client):
        """Test proper initialization of FastAPI dashboard"""
        assert fastapi_dashboard.redis_client == redis_client
        assert fastapi_dashboard.websocket_manager is not None
        assert fastapi_dashboard.metrics_collector is not None
        assert fastapi_dashboard.alert_manager is not None
        assert fastapi_dashboard.system_status == SystemStatus.HEALTHY

    def test_setup_middleware(self, fastapi_dashboard):
        """Test middleware setup"""
        # Check that CORS middleware is added
        middleware_types = [type(middleware.cls) for middleware in fastapi_dashboard.app.user_middleware]
        from fastapi.middleware.cors import CORSMiddleware
        assert CORSMiddleware in middleware_types

    def test_get_dashboard_html(self, fastapi_dashboard):
        """Test dashboard HTML generation"""
        html = fastapi_dashboard._get_dashboard_html()
        
        assert isinstance(html, str)
        assert '<!DOCTYPE html>' in html
        assert 'Medallion-X Trading Bot Dashboard' in html
        assert 'WebSocket' in html

    def test_get_app(self, fastapi_dashboard):
        """Test getting FastAPI app"""
        app = fastapi_dashboard.get_app()
        
        assert app is not None
        assert app.title == "Medallion-X Dashboard"

    @pytest.mark.asyncio
    async def test_update_response_time(self, fastapi_dashboard):
        """Test response time update"""
        await fastapi_dashboard._update_response_time(0.1)
        
        # Check that metric was stored
        response_time = await fastapi_dashboard.redis_client.get("metrics:http:avg_response_time")
        assert response_time is not None


@pytest.mark.integration
class TestMonitoringIntegration:
    """Integration tests for monitoring module"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring_pipeline(self, redis_client):
        """Test end-to-end monitoring pipeline"""
        # Create components
        metrics_collector = MetricsCollector(redis_client)
        alert_manager = AlertManager(redis_client)
        
        # Collect metrics
        with patch('psutil.cpu_percent', return_value=85.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_io_counters') as mock_network, \
             patch('psutil.net_connections', return_value=10), \
             patch('psutil.boot_time', return_value=time.time() - 86400):
            
            mock_memory.return_value.percent = 70.0
            mock_disk.return_value.used = 1000000
            mock_disk.return_value.total = 2000000
            mock_network.return_value.bytes_sent = 1000
            mock_network.return_value.bytes_recv = 2000
            mock_network.return_value.packets_sent = 10
            mock_network.return_value.packets_recv = 20
            
            system_metrics = await metrics_collector.collect_system_metrics()
            
            # Check for alerts
            alerts = await alert_manager.check_system_alerts(system_metrics)
            
            # Verify pipeline worked
            assert isinstance(system_metrics, SystemMetrics)
            assert isinstance(alerts, list)
            assert len(alerts) > 0  # Should have CPU alert
    
    @pytest.mark.asyncio
    async def test_nlp_to_command_execution(self, redis_client):
        """Test NLP processing to command execution"""
        telegram_bot = TelegramBot(redis_client)
        nlp_processor = NLPProcessor()
        
        # Process natural language message
        message_text = "buy 0.1 btc/usdt"
        user_id = 123456789  # Assume authorized
        
        intent = await nlp_processor.process_message(message_text, user_id)
        
        # Verify intent was classified correctly
        assert intent.command == "buy"
        assert intent.parameters.get('symbol') == "BTC/USDT"
        assert intent.parameters.get('amount') == "0.1"
        
        # Mock command execution
        mock_update = MagicMock()
        mock_update.message.reply_text = AsyncMock()
        
        context = {
            'update': mock_update,
            'parameters': intent.parameters,
            'command_name': intent.command
        }
        
        await telegram_bot.handle_buy(context)
        
        # Verify command was executed
        mock_update.message.reply_text.assert_called_once()
        assert 'Buy Order Placed' in mock_update.message.reply_text.call_args[0][0]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
