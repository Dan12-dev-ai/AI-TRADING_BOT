"""
Test suite for Risk Management Module
Production-ready tests for Dynamic Kelly and Risk Engine
"""

import pytest
import asyncio
import numpy as np
import time
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

import redis.asyncio as redis

from medallion_x.risk.dynamic_kelly import (
    RiskEngine, DynamicKellyCalculator, MarketAnalyzer,
    MarketStatistics, KellyParameters, RiskMetrics, MarketRegime
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
def market_analyzer():
    """Create a market analyzer for testing"""
    return MarketAnalyzer(lookback_window=100)


@pytest.fixture
def kelly_calculator():
    """Create a Kelly calculator for testing"""
    return DynamicKellyCalculator(safety_factor=0.25, max_leverage=3.0)


@pytest.fixture
def risk_engine(redis_client):
    """Create a risk engine for testing"""
    return RiskEngine(redis_client)


class TestMarketAnalyzer:
    """Test cases for MarketAnalyzer class"""
    
    def test_initialization(self, market_analyzer):
        """Test proper initialization of market analyzer"""
        assert market_analyzer.lookback_window == 100
        assert market_analyzer.ewma_lambda == 0.94
        assert market_analyzer.min_samples == 30
        assert len(market_analyzer.price_history) == 0
        assert len(market_analyzer.return_history) == 0

    def test_update_price(self, market_analyzer):
        """Test price update functionality"""
        symbol = "BTC/USDT"
        price = 50000.0
        timestamp = int(time.time() * 1000)
        
        market_analyzer.update_price(symbol, price, timestamp)
        
        assert symbol in market_analyzer.price_history
        assert len(market_analyzer.price_history[symbol]) == 1
        assert market_analyzer.price_history[symbol][0] == (timestamp, price)

    def test_return_calculation(self, market_analyzer):
        """Test return calculation"""
        symbol = "BTC/USDT"
        timestamp = int(time.time() * 1000)
        
        # Add two prices to generate a return
        market_analyzer.update_price(symbol, 50000.0, timestamp)
        market_analyzer.update_price(symbol, 50500.0, timestamp + 60000)
        
        assert symbol in market_analyzer.return_history
        assert len(market_analyzer.return_history[symbol]) == 1
        assert market_analyzer.return_history[symbol][0] > 0  # Positive return

    def test_calculate_market_statistics_insufficient_data(self, market_analyzer):
        """Test statistics calculation with insufficient data"""
        symbol = "BTC/USDT"
        
        # Add insufficient data
        for i in range(10):  # Less than min_samples
            market_analyzer.update_price(symbol, 50000.0 + i, int(time.time() * 1000) + i * 60000)
        
        stats = market_analyzer.calculate_market_statistics(symbol)
        assert stats is None

    def test_calculate_market_statistics_sufficient_data(self, market_analyzer):
        """Test statistics calculation with sufficient data"""
        symbol = "BTC/USDT"
        timestamp = int(time.time() * 1000)
        
        # Add sufficient data with some randomness
        base_price = 50000.0
        for i in range(50):
            price = base_price + np.random.randn() * 100
            market_analyzer.update_price(symbol, price, timestamp + i * 60000)
        
        stats = market_analyzer.calculate_market_statistics(symbol)
        
        assert stats is not None
        assert isinstance(stats, MarketStatistics)
        assert isinstance(stats.mean_return, float)
        assert isinstance(stats.volatility, float)
        assert isinstance(stats.sharpe_ratio, float)
        assert isinstance(stats.regime, MarketRegime)

    def test_regime_detection(self, market_analyzer):
        """Test market regime detection"""
        # Test bull market
        bull_returns = np.random.normal(0.002, 0.02, 100)  # Positive mean, moderate vol
        regime = market_analyzer._detect_regime(bull_returns, 0.002, 0.02)
        assert regime in [MarketRegime.BULL_MARKET, MarketRegime.HIGH_VOLATILITY]
        
        # Test bear market
        bear_returns = np.random.normal(-0.002, 0.02, 100)  # Negative mean, moderate vol
        regime = market_analyzer._detect_regime(bear_returns, -0.002, 0.02)
        assert regime in [MarketRegime.BEAR_MARKET, MarketRegime.HIGH_VOLATILITY]
        
        # Test sideways
        sideways_returns = np.random.normal(0.0001, 0.02, 100)  # Near-zero mean
        regime = market_analyzer._detect_regime(sideways_returns, 0.0001, 0.02)
        assert regime in [MarketRegime.SIDEWAYS, MarketRegime.HIGH_VOLATILITY]

    def test_correlation_matrix_calculation(self, market_analyzer):
        """Test correlation matrix calculation"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        timestamp = int(time.time() * 1000)
        
        # Add correlated data
        for i in range(50):
            base_return = np.random.randn() * 0.02
            btc_price = 50000.0 * (1 + base_return)
            eth_price = 3000.0 * (1 + base_return * 0.8)  # 80% correlation
            
            market_analyzer.update_price(symbols[0], btc_price, timestamp + i * 60000)
            market_analyzer.update_price(symbols[1], eth_price, timestamp + i * 60000)
        
        correlation_matrix = market_analyzer.calculate_correlation_matrix(symbols)
        
        assert correlation_matrix is not None
        assert correlation_matrix.shape == (2, 2)
        assert correlation_matrix[0, 1] > 0.5  # Should be positively correlated

    def test_correlation_matrix_insufficient_data(self, market_analyzer):
        """Test correlation matrix with insufficient data"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        
        correlation_matrix = market_analyzer.calculate_correlation_matrix(symbols)
        assert correlation_matrix is None


class TestDynamicKellyCalculator:
    """Test cases for DynamicKellyCalculator class"""
    
    def test_initialization(self, kelly_calculator):
        """Test proper initialization of Kelly calculator"""
        assert kelly_calculator.safety_factor == 0.25
        assert kelly_calculator.max_leverage == 3.0
        assert kelly_calculator.min_kelly == 0.01
        assert kelly_calculator.max_kelly == 0.25
        assert isinstance(kelly_calculator.market_analyzer, MarketAnalyzer)

    def test_calculate_kelly_fraction_insufficient_data(self, kelly_calculator):
        """Test Kelly calculation with insufficient data"""
        symbol = "BTC/USDT"
        
        # Create minimal market stats
        stats = MarketStatistics(
            mean_return=0.001,
            volatility=0.02,
            skewness=0.0,
            kurtosis=3.0,
            var_95=-0.03,
            var_99=-0.05,
            max_drawdown=-0.1,
            sharpe_ratio=0.5,
            sortino_ratio=0.7,
            calmar_ratio=0.3,
            win_rate=0.55,
            avg_win=0.02,
            avg_loss=-0.015,
            profit_factor=1.2,
            regime=MarketRegime.BULL_MARKET,
            confidence_level=0.5
        )
        
        kelly_params = kelly_calculator.calculate_kelly_fraction(symbol, stats)
        
        assert isinstance(kelly_params, KellyParameters)
        assert 0 <= kelly_params.position_size <= 1
        assert kelly_params.safety_factor == 0.25

    def test_calculate_kelly_fraction_good_conditions(self, kelly_calculator):
        """Test Kelly calculation with good market conditions"""
        symbol = "BTC/USDT"
        
        # Create favorable market stats
        stats = MarketStatistics(
            mean_return=0.002,
            volatility=0.015,
            skewness=0.5,  # Positive skew
            kurtosis=2.5,  # Low kurtosis
            var_95=-0.02,
            var_99=-0.03,
            max_drawdown=-0.05,
            sharpe_ratio=1.5,  # Good Sharpe
            sortino_ratio=2.0,
            calmar_ratio=0.8,
            win_rate=0.6,
            avg_win=0.025,
            avg_loss=-0.015,
            profit_factor=1.5,
            regime=MarketRegime.BULL_MARKET,
            confidence_level=0.9
        )
        
        kelly_params = kelly_calculator.calculate_kelly_fraction(symbol, stats)
        
        assert isinstance(kelly_params, KellyParameters)
        assert kelly_params.position_size > 0
        assert kelly_params.expected_growth > 0
        assert kelly_params.risk_of_ruin < 0.5

    def test_calculate_kelly_fraction_bad_conditions(self, kelly_calculator):
        """Test Kelly calculation with poor market conditions"""
        symbol = "BTC/USDT"
        
        # Create unfavorable market stats
        stats = MarketStatistics(
            mean_return=-0.001,  # Negative return
            volatility=0.04,  # High volatility
            skewness=-0.5,  # Negative skew
            kurtosis=5.0,  # High kurtosis (fat tails)
            var_95=-0.06,
            var_99=-0.08,
            max_drawdown=-0.25,  # Severe drawdown
            sharpe_ratio=-0.5,  # Negative Sharpe
            sortino_ratio=-0.7,
            calmar_ratio=-0.2,
            win_rate=0.4,  # Low win rate
            avg_win=0.015,
            avg_loss=-0.025,
            profit_factor=0.8,
            regime=MarketRegime.HIGH_VOLATILITY,
            confidence_level=0.8
        )
        
        kelly_params = kelly_calculator.calculate_kelly_fraction(symbol, stats)
        
        assert isinstance(kelly_params, KellyParameters)
        assert kelly_params.position_size <= kelly_calculator.max_kelly
        # Should be conservative due to poor conditions

    def test_skew_adjustment(self, kelly_calculator):
        """Test skewness adjustment calculation"""
        # Positive skewness
        pos_adjustment = kelly_calculator._calculate_skew_adjustment(0.5)
        assert pos_adjustment > 1.0
        
        # Negative skewness
        neg_adjustment = kelly_calculator._calculate_skew_adjustment(-0.5)
        assert neg_adjustment < 1.0
        
        # Zero skewness
        zero_adjustment = kelly_calculator._calculate_skew_adjustment(0.0)
        assert zero_adjustment == 1.0

    def test_kurtosis_adjustment(self, kelly_calculator):
        """Test kurtosis adjustment calculation"""
        # Normal kurtosis
        normal_adjustment = kelly_calculator._calculate_kurtosis_adjustment(3.0)
        assert normal_adjustment == 1.0
        
        # High kurtosis (fat tails)
        high_adjustment = kelly_calculator._calculate_kurtosis_adjustment(5.0)
        assert high_adjustment < 1.0
        
        # Low kurtosis
        low_adjustment = kelly_calculator._calculate_kurtosis_adjustment(2.0)
        assert low_adjustment == 1.0

    def test_regime_adjustment(self, kelly_calculator):
        """Test regime adjustment calculation"""
        # Bull market
        bull_adjustment = kelly_calculator._get_regime_adjustment(MarketRegime.BULL_MARKET)
        assert bull_adjustment > 1.0
        
        # Bear market
        bear_adjustment = kelly_calculator._get_regime_adjustment(MarketRegime.BEAR_MARKET)
        assert bear_adjustment < 1.0
        
        # High volatility
        high_vol_adjustment = kelly_calculator._get_regime_adjustment(MarketRegime.HIGH_VOLATILITY)
        assert high_vol_adjustment < 1.0

    def test_leverage_multiplier(self, kelly_calculator):
        """Test leverage multiplier calculation"""
        # Good conditions (high Sharpe, low vol, low drawdown)
        good_stats = MarketStatistics(
            mean_return=0.002, volatility=0.01, skewness=0.0, kurtosis=3.0,
            var_95=-0.02, var_99=-0.03, max_drawdown=-0.05, sharpe_ratio=2.0,
            sortino_ratio=2.5, calmar_ratio=1.0, win_rate=0.6, avg_win=0.02,
            avg_loss=-0.01, profit_factor=1.5, regime=MarketRegime.BULL_MARKET,
            confidence_level=0.9
        )
        
        good_multiplier = kelly_calculator._calculate_leverage_multiplier(good_stats)
        assert good_multiplier > 1.0
        assert good_multiplier <= kelly_calculator.max_leverage
        
        # Poor conditions
        poor_stats = MarketStatistics(
            mean_return=0.001, volatility=0.05, skewness=0.0, kurtosis=3.0,
            var_95=-0.08, var_99=-0.12, max_drawdown=-0.3, sharpe_ratio=0.3,
            sortino_ratio=0.4, calmar_ratio=0.1, win_rate=0.5, avg_win=0.02,
            avg_loss=-0.02, profit_factor=1.0, regime=MarketRegime.HIGH_VOLATILITY,
            confidence_level=0.8
        )
        
        poor_multiplier = kelly_calculator._calculate_leverage_multiplier(poor_stats)
        assert poor_multiplier < 1.0

    def test_portfolio_kelly_calculation(self, kelly_calculator):
        """Test portfolio Kelly calculation"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        
        # Create market stats for both symbols
        stats_list = []
        for _ in symbols:
            stats = MarketStatistics(
                mean_return=0.001, volatility=0.02, skewness=0.0, kurtosis=3.0,
                var_95=-0.03, var_99=-0.05, max_drawdown=-0.1, sharpe_ratio=0.7,
                sortino_ratio=1.0, calmar_ratio=0.3, win_rate=0.55, avg_win=0.02,
                avg_loss=-0.015, profit_factor=1.2, regime=MarketRegime.BULL_MARKET,
                confidence_level=0.8
            )
            stats_list.append(stats)
        
        portfolio_kelly = kelly_calculator.calculate_portfolio_kelly(symbols, stats_list)
        
        assert isinstance(portfolio_kelly, dict)
        assert len(portfolio_kelly) == len(symbols)
        for symbol in symbols:
            assert symbol in portfolio_kelly
            assert isinstance(portfolio_kelly[symbol], KellyParameters)


class TestRiskEngine:
    """Test cases for RiskEngine class"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, risk_engine):
        """Test proper initialization of risk engine"""
        assert risk_engine.redis_client is not None
        assert isinstance(risk_engine.kelly_calculator, DynamicKellyCalculator)
        assert risk_engine.max_portfolio_risk == config.risk.max_portfolio_risk
        assert risk_engine.max_position_size == config.risk.max_position_size
        assert risk_engine.portfolio_value == 100000.0
        assert len(risk_engine.current_positions) == 0

    @pytest.mark.asyncio
    async def test_calculate_position_size_insufficient_data(self, risk_engine):
        """Test position size calculation with insufficient data"""
        symbol = "BTC/USDT"
        signal_strength = 0.8
        confidence = 0.7
        
        result = await risk_engine.calculate_position_size(symbol, signal_strength, confidence)
        
        assert isinstance(result, dict)
        assert 'symbol' in result
        assert 'recommended_position' in result
        assert 'kelly_fraction' in result
        assert 'risk_check' in result
        assert result['symbol'] == symbol

    @pytest.mark.asyncio
    async def test_calculate_position_size_with_market_data(self, risk_engine):
        """Test position size calculation with market data"""
        symbol = "BTC/USDT"
        signal_strength = 0.8
        confidence = 0.7
        
        # Add market data
        for i in range(50):
            price = 50000.0 + np.random.randn() * 100
            timestamp = int(time.time() * 1000) + i * 60000
            risk_engine.update_market_data(symbol, price, timestamp)
        
        result = await risk_engine.calculate_position_size(symbol, signal_strength, confidence)
        
        assert isinstance(result, dict)
        assert 'recommended_position' in result
        assert 0 <= result['recommended_position'] <= risk_engine.max_position_size
        assert 'market_regime' in result
        assert 'sharpe_ratio' in result

    @pytest.mark.asyncio
    async def test_risk_check(self, risk_engine):
        """Test risk check functionality"""
        symbol = "BTC/USDT"
        position_size = 0.05
        
        # Create market stats
        stats = MarketStatistics(
            mean_return=0.001, volatility=0.02, skewness=0.0, kurtosis=3.0,
            var_95=-0.03, var_99=-0.05, max_drawdown=-0.1, sharpe_ratio=0.7,
            sortino_ratio=1.0, calmar_ratio=0.3, win_rate=0.55, avg_win=0.02,
            avg_loss=-0.015, profit_factor=1.2, regime=MarketRegime.BULL_MARKET,
            confidence_level=0.8
        )
        
        risk_check = await risk_engine._risk_check(symbol, position_size, stats)
        
        assert isinstance(risk_check, dict)
        assert 'approved' in risk_check
        assert 'warnings' in risk_check
        assert 'risk_level' in risk_check
        assert isinstance(risk_check['warnings'], list)

    @pytest.mark.asyncio
    async def test_risk_check_excessive_position(self, risk_engine):
        """Test risk check with excessive position size"""
        symbol = "BTC/USDT"
        position_size = 0.5  # Exceeds max_position_size
        
        # Create market stats
        stats = MarketStatistics(
            mean_return=0.001, volatility=0.02, skewness=0.0, kurtosis=3.0,
            var_95=-0.03, var_99=-0.05, max_drawdown=-0.1, sharpe_ratio=0.7,
            sortino_ratio=1.0, calmar_ratio=0.3, win_rate=0.55, avg_win=0.02,
            avg_loss=-0.015, profit_factor=1.2, regime=MarketRegime.BULL_MARKET,
            confidence_level=0.8
        )
        
        risk_check = await risk_engine._risk_check(symbol, position_size, stats)
        
        assert risk_check['approved'] is False
        assert any('exceeds maximum' in warning.lower() for warning in risk_check['warnings'])

    @pytest.mark.asyncio
    async def test_update_position(self, risk_engine):
        """Test position update"""
        symbol = "BTC/USDT"
        new_position = 0.1
        price = 50000.0
        
        old_portfolio_value = risk_engine.portfolio_value
        
        await risk_engine.update_position(symbol, new_position, price)
        
        assert symbol in risk_engine.current_positions
        assert risk_engine.current_positions[symbol] == new_position
        assert risk_engine.portfolio_value != old_portfolio_value

    @pytest.mark.asyncio
    async def test_calculate_portfolio_risk(self, risk_engine):
        """Test portfolio risk calculation"""
        # Add some positions
        risk_engine.current_positions = {
            'BTC/USDT': 0.05,
            'ETH/USDT': 0.03,
            'SOL/USDT': 0.02
        }
        
        risk_metrics = await risk_engine.calculate_portfolio_risk()
        
        assert isinstance(risk_metrics, RiskMetrics)
        assert risk_metrics.portfolio_risk > 0
        assert risk_metrics.max_position_size > 0
        assert risk_metrics.var_1day > 0
        assert risk_metrics.var_5day > 0

    @pytest.mark.asyncio
    async def test_get_kelly_history(self, risk_engine):
        """Test getting Kelly calculation history"""
        symbol = "BTC/USDT"
        
        history = await risk_engine.get_kelly_history(symbol)
        
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_get_risk_events(self, risk_engine):
        """Test getting risk events"""
        events = await risk_engine.get_risk_events()
        
        assert isinstance(events, list)

    def test_update_market_data(self, risk_engine):
        """Test market data update"""
        symbol = "BTC/USDT"
        price = 50000.0
        timestamp = int(time.time() * 1000)
        
        risk_engine.update_market_data(symbol, price, timestamp)
        
        # Verify data was passed to market analyzer
        assert symbol in risk_engine.kelly_calculator.market_analyzer.price_history

    def test_get_metrics(self, risk_engine):
        """Test getting performance metrics"""
        metrics = risk_engine.get_metrics()
        
        assert isinstance(metrics, dict)
        assert 'kelly_calculations' in metrics
        assert 'current_positions' in metrics
        assert 'portfolio_value' in metrics

    @pytest.mark.asyncio
    async def test_start_stop(self, risk_engine):
        """Test starting and stopping risk engine"""
        await risk_engine.start()
        
        await risk_engine.stop()


@pytest.mark.integration
class TestRiskIntegration:
    """Integration tests for risk management module"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_risk_pipeline(self, redis_client):
        """Test end-to-end risk management pipeline"""
        # Create risk engine
        risk_engine = RiskEngine(redis_client)
        
        # Add market data
        symbol = "BTC/USDT"
        for i in range(50):
            price = 50000.0 + np.random.randn() * 100
            timestamp = int(time.time() * 1000) + i * 60000
            risk_engine.update_market_data(symbol, price, timestamp)
        
        # Calculate position size
        result = await risk_engine.calculate_position_size(symbol, 0.8, 0.7)
        
        # Update position
        await risk_engine.update_position(symbol, result['recommended_position'], 50000.0)
        
        # Calculate portfolio risk
        risk_metrics = await risk_engine.calculate_portfolio_risk()
        
        # Verify pipeline worked
        assert isinstance(result, dict)
        assert 'recommended_position' in result
        assert symbol in risk_engine.current_positions
        assert isinstance(risk_metrics, RiskMetrics)
        assert risk_metrics.portfolio_risk > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
