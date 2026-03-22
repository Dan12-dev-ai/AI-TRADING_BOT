"""
Medallion-X Dynamic Kelly Criterion Module
Advanced position sizing with adaptive risk management
Production-ready implementation with real-time parameter estimation
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
import time
from scipy import stats
from scipy.optimize import minimize
import json

import redis.asyncio as redis

from ..config.settings import config
from ..utils.logging import get_logger

logger = get_logger(__name__)

class MarketRegime(Enum):
    """Market regime classification"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

@dataclass
class MarketStatistics:
    """Market statistics for Kelly calculation"""
    mean_return: float
    volatility: float
    skewness: float
    kurtosis: float
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    regime: MarketRegime
    confidence_level: float  # Statistical confidence in estimates

@dataclass
class KellyParameters:
    """Kelly criterion parameters"""
    kelly_fraction: float  # Raw Kelly fraction
    adjusted_fraction: float  # Adjusted for safety
    leverage_multiplier: float  # Leverage adjustment
    position_size: float  # Final position size (0-1)
    confidence_interval: Tuple[float, float]  # CI for Kelly fraction
    expected_growth: float  # Expected growth rate
    risk_of_ruin: float  # Probability of ruin
    safety_factor: float  # Safety multiplier applied

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    current_position_size: float
    max_position_size: float
    portfolio_risk: float
    var_1day: float
    var_5day: float
    expected_shortfall: float
    beta: float
    correlation_risk: float
    liquidity_risk: float
    concentration_risk: float
    leverage_risk: float
    time_decay_risk: float

class MarketAnalyzer:
    """
    Advanced market analysis for Kelly criterion
    - Real-time statistical estimation
    - Regime detection
    - Risk-adjusted performance metrics
    """
    
    def __init__(self, lookback_window: int = 252):  # 1 year of daily data
        self.lookback_window = lookback_window
        self.price_history: Dict[str, List[Tuple[int, float]]] = {}
        self.return_history: Dict[str, List[float]] = {}
        
        # Statistical estimators
        self.ewma_lambda = 0.94  # Exponential weighting
        self.min_samples = 30  # Minimum samples for reliable statistics

    def update_price(self, symbol: str, price: float, timestamp: int) -> None:
        """Update price history for a symbol"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.return_history[symbol] = []
        
        # Add new price
        self.price_history[symbol].append((timestamp, price))
        
        # Maintain window size
        if len(self.price_history[symbol]) > self.lookback_window:
            self.price_history[symbol] = self.price_history[symbol][-self.lookback_window:]
        
        # Calculate returns
        if len(self.price_history[symbol]) >= 2:
            prev_price = self.price_history[symbol][-2][1]
            current_price = price
            log_return = np.log(current_price / prev_price)
            
            self.return_history[symbol].append(log_return)
            
            # Maintain return history
            if len(self.return_history[symbol]) > self.lookback_window:
                self.return_history[symbol] = self.return_history[symbol][-self.lookback_window:]

    def calculate_market_statistics(self, symbol: str) -> Optional[MarketStatistics]:
        """Calculate comprehensive market statistics"""
        if symbol not in self.return_history or len(self.return_history[symbol]) < self.min_samples:
            return None
        
        returns = np.array(self.return_history[symbol])
        
        try:
            # Basic statistics
            mean_return = np.mean(returns)
            volatility = np.std(returns)
            
            # Higher moments
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            # Risk metrics
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Drawdown calculation
            cumulative_returns = np.exp(np.cumsum(returns))
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # Performance ratios
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            excess_returns = returns - risk_free_rate
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
            
            downside_returns = returns[returns < risk_free_rate]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
            sortino_ratio = np.mean(excess_returns) / downside_std if downside_std > 0 else 0
            
            calmar_ratio = np.mean(returns) / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Trading statistics
            win_rate = len(returns[returns > 0]) / len(returns)
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            avg_win = np.mean(wins) if len(wins) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            profit_factor = abs(np.sum(wins) / np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else 0
            
            # Regime detection
            regime = self._detect_regime(returns, mean_return, volatility)
            
            # Confidence level based on sample size
            confidence_level = min(1.0, len(returns) / self.lookback_window)
            
            return MarketStatistics(
                mean_return=mean_return,
                volatility=volatility,
                skewness=skewness,
                kurtosis=kurtosis,
                var_95=var_95,
                var_99=var_99,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                regime=regime,
                confidence_level=confidence_level
            )
            
        except Exception as e:
            logger.error(f"Error calculating market statistics for {symbol}: {e}")
            return None

    def _detect_regime(self, returns: np.ndarray, mean_return: float, volatility: float) -> MarketRegime:
        """Detect market regime based on statistical properties"""
        # Normalize volatility (annualized)
        annual_vol = volatility * np.sqrt(252)
        
        # Trend detection
        if mean_return > 0.001:  # > 0.1% daily return
            if annual_vol > 0.3:  # > 30% annual volatility
                return MarketRegime.HIGH_VOLATILITY
            else:
                return MarketRegime.BULL_MARKET
        elif mean_return < -0.001:  # < -0.1% daily return
            if annual_vol > 0.3:
                return MarketRegime.HIGH_VOLATILITY
            else:
                return MarketRegime.BEAR_MARKET
        else:
            if annual_vol > 0.3:
                return MarketRegime.HIGH_VOLATILITY
            elif annual_vol < 0.15:
                return MarketRegime.LOW_VOLATILITY
            else:
                return MarketRegime.SIDEWAYS

    def calculate_correlation_matrix(self, symbols: List[str]) -> Optional[np.ndarray]:
        """Calculate correlation matrix for multiple symbols"""
        if len(symbols) < 2:
            return None
        
        # Collect returns for all symbols
        returns_data = []
        valid_symbols = []
        
        for symbol in symbols:
            if symbol in self.return_history and len(self.return_history[symbol]) >= self.min_samples:
                returns_data.append(self.return_history[symbol][-100:])  # Use last 100 returns
                valid_symbols.append(symbol)
        
        if len(returns_data) < 2:
            return None
        
        # Align returns (find common length)
        min_length = min(len(r) for r in returns_data)
        aligned_returns = [r[-min_length:] for r in returns_data]
        
        # Calculate correlation matrix
        returns_matrix = np.column_stack(aligned_returns)
        correlation_matrix = np.corrcoef(returns_matrix.T)
        
        return correlation_matrix

class DynamicKellyCalculator:
    """
    Dynamic Kelly criterion calculator with advanced risk adjustments
    - Adaptive parameter estimation
    - Multi-asset optimization
    - Risk-adjusted position sizing
    """
    
    def __init__(self, safety_factor: float = 0.25, max_leverage: float = 3.0):
        self.safety_factor = safety_factor
        self.max_leverage = max_leverage
        self.market_analyzer = MarketAnalyzer()
        
        # Kelly optimization parameters
        self.min_kelly = 0.01  # Minimum Kelly fraction
        self.max_kelly = 0.25  # Maximum Kelly fraction
        self.confidence_threshold = 0.7  # Minimum confidence for Kelly calculation

    def calculate_kelly_fraction(self, symbol: str, market_stats: MarketStatistics) -> KellyParameters:
        """Calculate dynamic Kelly fraction for a single asset"""
        try:
            # Basic Kelly formula: f = (bp - q) / b
            # Where: b = odds, p = win probability, q = lose probability
            
            # Convert market statistics to Kelly parameters
            win_prob = market_stats.win_rate
            avg_win = market_stats.avg_win
            avg_loss = abs(market_stats.avg_loss)
            
            if avg_loss == 0 or win_prob == 0:
                return self._get_default_kelly()
            
            # Calculate odds (average win/loss ratio)
            odds = avg_win / avg_loss
            
            # Basic Kelly fraction
            raw_kelly = (win_prob * odds - (1 - win_prob)) / odds
            
            # Adjust for skewness and kurtosis
            skew_adjustment = self._calculate_skew_adjustment(market_stats.skewness)
            kurtosis_adjustment = self._calculate_kurtosis_adjustment(market_stats.kurtosis)
            
            adjusted_kelly = raw_kelly * skew_adjustment * kurtosis_adjustment
            
            # Apply safety factor
            safe_kelly = adjusted_kelly * self.safety_factor
            
            # Apply regime-based adjustments
            regime_adjustment = self._get_regime_adjustment(market_stats.regime)
            final_kelly = safe_kelly * regime_adjustment
            
            # Apply confidence weighting
            confidence_weight = market_stats.confidence_level
            weighted_kelly = final_kelly * confidence_weight
            
            # Bound the Kelly fraction
            bounded_kelly = max(self.min_kelly, min(self.max_kelly, weighted_kelly))
            
            # Calculate confidence interval using bootstrap
            ci_lower, ci_upper = self._calculate_kelly_ci(symbol, bounded_kelly)
            
            # Calculate leverage multiplier
            leverage_multiplier = self._calculate_leverage_multiplier(market_stats)
            
            # Final position size
            position_size = bounded_kelly * leverage_multiplier
            
            # Calculate expected growth and risk of ruin
            expected_growth = self._calculate_expected_growth(position_size, market_stats)
            risk_of_ruin = self._calculate_risk_of_ruin(position_size, market_stats)
            
            return KellyParameters(
                kelly_fraction=raw_kelly,
                adjusted_fraction=bounded_kelly,
                leverage_multiplier=leverage_multiplier,
                position_size=position_size,
                confidence_interval=(ci_lower, ci_upper),
                expected_growth=expected_growth,
                risk_of_ruin=risk_of_ruin,
                safety_factor=self.safety_factor
            )
            
        except Exception as e:
            logger.error(f"Error calculating Kelly fraction for {symbol}: {e}")
            return self._get_default_kelly()

    def _get_default_kelly(self) -> KellyParameters:
        """Get default Kelly parameters when calculation fails"""
        return KellyParameters(
            kelly_fraction=0.05,
            adjusted_fraction=0.01,
            leverage_multiplier=1.0,
            position_size=0.01,
            confidence_interval=(0.005, 0.015),
            expected_growth=0.001,
            risk_of_ruin=0.1,
            safety_factor=self.safety_factor
        )

    def _calculate_skew_adjustment(self, skewness: float) -> float:
        """Calculate adjustment factor based on skewness"""
        # Positive skewness is favorable, negative is unfavorable
        if skewness > 0:
            return min(1.2, 1 + skewness * 0.1)  # Up to 20% increase
        else:
            return max(0.8, 1 + skewness * 0.1)  # Up to 20% decrease

    def _calculate_kurtosis_adjustment(self, kurtosis: float) -> float:
        """Calculate adjustment factor based on kurtosis"""
        # High kurtosis (fat tails) is unfavorable
        if kurtosis > 3:  # Higher than normal distribution
            return max(0.7, 1 - (kurtosis - 3) * 0.05)  # Decrease for fat tails
        else:
            return 1.0

    def _get_regime_adjustment(self, regime: MarketRegime) -> float:
        """Get regime-based adjustment factor"""
        adjustments = {
            MarketRegime.BULL_MARKET: 1.1,
            MarketRegime.BEAR_MARKET: 0.9,
            MarketRegime.SIDEWAYS: 0.8,
            MarketRegime.HIGH_VOLATILITY: 0.7,
            MarketRegime.LOW_VOLATILITY: 1.05
        }
        return adjustments.get(regime, 1.0)

    def _calculate_kelly_ci(self, symbol: str, kelly_fraction: float, n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Calculate confidence interval for Kelly fraction using bootstrap"""
        if symbol not in self.market_analyzer.return_history:
            return (kelly_fraction * 0.8, kelly_fraction * 1.2)
        
        returns = np.array(self.market_analyzer.return_history[symbol])
        if len(returns) < 30:
            return (kelly_fraction * 0.8, kelly_fraction * 1.2)
        
        bootstrap_kellys = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            sample_returns = np.random.choice(returns, size=len(returns), replace=True)
            
            # Calculate Kelly for sample
            win_rate = len(sample_returns[sample_returns > 0]) / len(sample_returns)
            wins = sample_returns[sample_returns > 0]
            losses = sample_returns[sample_returns < 0]
            
            if len(wins) > 0 and len(losses) > 0:
                avg_win = np.mean(wins)
                avg_loss = abs(np.mean(losses))
                odds = avg_win / avg_loss
                sample_kelly = (win_rate * odds - (1 - win_rate)) / odds
                bootstrap_kellys.append(sample_kelly)
        
        if bootstrap_kellys:
            ci_lower = np.percentile(bootstrap_kellys, 2.5)
            ci_upper = np.percentile(bootstrap_kellys, 97.5)
            return (max(0, ci_lower), ci_upper)
        else:
            return (kelly_fraction * 0.8, kelly_fraction * 1.2)

    def _calculate_leverage_multiplier(self, market_stats: MarketStatistics) -> float:
        """Calculate leverage multiplier based on market conditions"""
        base_leverage = 1.0
        
        # Adjust based on Sharpe ratio
        if market_stats.sharpe_ratio > 1.5:
            base_leverage *= 1.5
        elif market_stats.sharpe_ratio > 1.0:
            base_leverage *= 1.2
        elif market_stats.sharpe_ratio < 0.5:
            base_leverage *= 0.8
        
        # Adjust based on volatility
        annual_vol = market_stats.volatility * np.sqrt(252)
        if annual_vol > 0.4:  # Very high volatility
            base_leverage *= 0.7
        elif annual_vol > 0.3:  # High volatility
            base_leverage *= 0.85
        elif annual_vol < 0.15:  # Low volatility
            base_leverage *= 1.15
        
        # Adjust based on maximum drawdown
        if market_stats.max_drawdown < -0.2:  # Severe drawdown
            base_leverage *= 0.6
        elif market_stats.max_drawdown < -0.1:  # Moderate drawdown
            base_leverage *= 0.8
        
        return min(self.max_leverage, max(0.5, base_leverage))

    def _calculate_expected_growth(self, position_size: float, market_stats: MarketStatistics) -> float:
        """Calculate expected growth rate"""
        # Expected growth = position_size * mean_return - 0.5 * position_size^2 * variance
        mean_return = market_stats.mean_return
        variance = market_stats.volatility ** 2
        
        expected_growth = position_size * mean_return - 0.5 * (position_size ** 2) * variance
        return expected_growth

    def _calculate_risk_of_ruin(self, position_size: float, market_stats: MarketStatistics) -> float:
        """Calculate probability of ruin"""
        # Simplified risk of ruin calculation
        # More sophisticated methods exist for production
        
        if market_stats.mean_return <= 0:
            return 1.0  # Certain ruin with negative or zero expected return
        
        # Using Kelly's risk of ruin approximation
        sharpe_ratio = market_stats.sharpe_ratio
        if sharpe_ratio <= 0:
            return 1.0
        
        # Risk of ruin decreases with higher Sharpe ratio and smaller position size
        risk_of_ruin = np.exp(-2 * sharpe_ratio * position_size)
        return min(1.0, max(0.0, risk_of_ruin))

    def calculate_portfolio_kelly(self, symbols: List[str], market_stats_list: List[MarketStatistics]) -> Dict[str, KellyParameters]:
        """Calculate Kelly fractions for a portfolio of assets"""
        portfolio_kelly = {}
        
        # Calculate individual Kelly fractions
        for symbol, stats in zip(symbols, market_stats_list):
            kelly_params = self.calculate_kelly_fraction(symbol, stats)
            portfolio_kelly[symbol] = kelly_params
        
        # Get correlation matrix
        correlation_matrix = self.market_analyzer.calculate_correlation_matrix(symbols)
        
        if correlation_matrix is not None and len(symbols) > 1:
            # Adjust for portfolio correlation
            portfolio_kelly = self._adjust_for_correlation(portfolio_kelly, correlation_matrix)
        
        return portfolio_kelly

    def _adjust_for_correlation(self, kelly_params: Dict[str, KellyParameters], 
                               correlation_matrix: np.ndarray) -> Dict[str, KellyParameters]:
        """Adjust Kelly fractions based on portfolio correlation"""
        symbols = list(kelly_params.keys())
        
        # Calculate average correlation
        avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
        
        # Adjust position sizes based on correlation
        correlation_factor = 1.0 / (1.0 + avg_correlation)  # Reduce positions with high correlation
        
        adjusted_kelly = {}
        for symbol, params in kelly_params.items():
            adjusted_params = KellyParameters(
                kelly_fraction=params.kelly_fraction,
                adjusted_fraction=params.adjusted_fraction * correlation_factor,
                leverage_multiplier=params.leverage_multiplier,
                position_size=params.position_size * correlation_factor,
                confidence_interval=params.confidence_interval,
                expected_growth=params.expected_growth * correlation_factor,
                risk_of_ruin=params.risk_of_ruin,
                safety_factor=params.safety_factor
            )
            adjusted_kelly[symbol] = adjusted_params
        
        return adjusted_kelly


class RiskEngine:
    """
    Main Risk Management Engine for Medallion-X
    - Dynamic Kelly criterion implementation
    - Real-time risk monitoring
    - Portfolio risk management
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.kelly_calculator = DynamicKellyCalculator(
            safety_factor=config.risk.kelly_multiplier,
            max_leverage=3.0
        )
        
        # Risk limits
        self.max_portfolio_risk = config.risk.max_portfolio_risk
        self.max_position_size = config.risk.max_position_size
        
        # Current portfolio state
        self.current_positions: Dict[str, float] = {}
        self.portfolio_value = 100000.0  # Starting portfolio value
        
        # Risk metrics tracking
        self.risk_history: List[Dict[str, Any]] = []
        self.max_history_length = 1000
        
        # Performance metrics
        self.metrics = {
            'kelly_calculations': 0,
            'risk_adjustments': 0,
            'max_risk_breach': 0,
            'average_risk': 0.0,
            'risk_events': 0
        }

    async def calculate_position_size(self, symbol: str, signal_strength: float, 
                                   confidence: float) -> Dict[str, Any]:
        """Calculate optimal position size using dynamic Kelly"""
        try:
            # Get market statistics
            market_stats = self.kelly_calculator.market_analyzer.calculate_market_statistics(symbol)
            
            if market_stats is None:
                return self._get_conservative_position(symbol, signal_strength, confidence)
            
            # Calculate Kelly parameters
            kelly_params = self.kelly_calculator.calculate_kelly_fraction(symbol, market_stats)
            
            # Adjust based on signal strength and confidence
            signal_adjustment = min(1.0, signal_strength * confidence)
            adjusted_position = kelly_params.position_size * signal_adjustment
            
            # Apply risk limits
            final_position = min(adjusted_position, self.max_position_size)
            
            # Calculate portfolio impact
            portfolio_impact = final_position * (self.current_positions.get(symbol, 0) / self.portfolio_value)
            
            # Risk check
            risk_check = await self._risk_check(symbol, final_position, market_stats)
            
            result = {
                'symbol': symbol,
                'recommended_position': final_position,
                'kelly_fraction': kelly_params.kelly_fraction,
                'adjusted_fraction': kelly_params.adjusted_fraction,
                'leverage_multiplier': kelly_params.leverage_multiplier,
                'expected_growth': kelly_params.expected_growth,
                'risk_of_ruin': kelly_params.risk_of_ruin,
                'confidence_interval': kelly_params.confidence_interval,
                'signal_adjustment': signal_adjustment,
                'portfolio_impact': portfolio_impact,
                'risk_check': risk_check,
                'market_regime': market_stats.regime.value,
                'sharpe_ratio': market_stats.sharpe_ratio,
                'max_drawdown': market_stats.max_drawdown
            }
            
            # Store calculation
            await self._store_kelly_calculation(symbol, result)
            
            # Update metrics
            self.metrics['kelly_calculations'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return self._get_conservative_position(symbol, signal_strength, confidence)

    def _get_conservative_position(self, symbol: str, signal_strength: float, confidence: float) -> Dict[str, Any]:
        """Get conservative position size when Kelly calculation fails"""
        conservative_size = 0.01 * signal_strength * confidence  # Very conservative
        
        return {
            'symbol': symbol,
            'recommended_position': conservative_size,
            'kelly_fraction': 0.01,
            'adjusted_fraction': conservative_size,
            'leverage_multiplier': 1.0,
            'expected_growth': 0.001,
            'risk_of_ruin': 0.05,
            'confidence_interval': (0.005, 0.015),
            'signal_adjustment': signal_strength * confidence,
            'portfolio_impact': conservative_size / self.portfolio_value,
            'risk_check': {'approved': True, 'warnings': []},
            'market_regime': 'unknown',
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }

    async def _risk_check(self, symbol: str, position_size: float, market_stats: MarketStatistics) -> Dict[str, Any]:
        """Perform comprehensive risk check"""
        warnings = []
        approved = True
        
        # Position size check
        if position_size > self.max_position_size:
            warnings.append(f"Position size {position_size:.3f} exceeds maximum {self.max_position_size:.3f}")
            approved = False
        
        # Portfolio risk check
        current_risk = abs(self.current_positions.get(symbol, 0)) / self.portfolio_value
        new_risk = abs(position_size) / self.portfolio_value
        
        if new_risk > self.max_portfolio_risk:
            warnings.append(f"Portfolio risk {new_risk:.3f} exceeds maximum {self.max_portfolio_risk:.3f}")
            approved = False
        
        # Volatility check
        if market_stats.volatility > 0.05:  # 5% daily volatility
            warnings.append(f"High volatility detected: {market_stats.volatility:.3f}")
        
        # Drawdown check
        if market_stats.max_drawdown < -0.2:  # 20% drawdown
            warnings.append(f"Severe drawdown detected: {market_stats.max_drawdown:.3f}")
        
        # Sharpe ratio check
        if market_stats.sharpe_ratio < 0.5:
            warnings.append(f"Low Sharpe ratio: {market_stats.sharpe_ratio:.3f}")
        
        # Regime check
        if market_stats.regime == MarketRegime.HIGH_VOLATILITY:
            warnings.append("High volatility regime detected")
        
        return {
            'approved': approved,
            'warnings': warnings,
            'current_risk': current_risk,
            'new_risk': new_risk,
            'risk_level': 'HIGH' if len(warnings) > 2 else 'MEDIUM' if warnings else 'LOW'
        }

    async def update_position(self, symbol: str, new_position: float, price: float) -> None:
        """Update current position"""
        old_position = self.current_positions.get(symbol, 0.0)
        self.current_positions[symbol] = new_position
        
        # Update portfolio value (simplified)
        position_change = (new_position - old_position) * price
        self.portfolio_value += position_change
        
        # Log risk event if significant change
        if abs(position_change) > self.portfolio_value * 0.01:  # 1% change
            await self._log_risk_event(symbol, old_position, new_position, position_change)

    async def calculate_portfolio_risk(self) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            # Current position sizes
            position_sizes = np.array(list(self.current_positions.values()))
            
            # Portfolio risk metrics
            current_position_size = np.mean(np.abs(position_sizes)) if len(position_sizes) > 0 else 0.0
            max_position_size = np.max(np.abs(position_sizes)) if len(position_sizes) > 0 else 0.0
            portfolio_risk = np.sum(np.abs(position_sizes)) / self.portfolio_value
            
            # VaR calculations (simplified)
            var_1day = portfolio_risk * 0.02  # 2% daily VaR
            var_5day = portfolio_risk * 0.05  # 5% VaR
            
            # Expected shortfall
            expected_shortfall = var_1day * 1.5  # 1.5x VaR
            
            # Other risk metrics (simplified)
            beta = 1.0  # Would need market data for calculation
            correlation_risk = 0.5  # Would need correlation matrix
            liquidity_risk = 0.1  # Would need liquidity data
            concentration_risk = max_position_size / portfolio_risk if portfolio_risk > 0 else 0.0
            leverage_risk = np.mean(position_sizes) / self.portfolio_value if len(position_sizes) > 0 else 0.0
            time_decay_risk = 0.01  # Daily time decay
            
            return RiskMetrics(
                current_position_size=current_position_size,
                max_position_size=max_position_size,
                portfolio_risk=portfolio_risk,
                var_1day=var_1day,
                var_5day=var_5day,
                expected_shortfall=expected_shortfall,
                beta=beta,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                concentration_risk=concentration_risk,
                leverage_risk=leverage_risk,
                time_decay_risk=time_decay_risk
            )
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return RiskMetrics(
                current_position_size=0.0,
                max_position_size=0.0,
                portfolio_risk=0.0,
                var_1day=0.0,
                var_5day=0.0,
                expected_shortfall=0.0,
                beta=1.0,
                correlation_risk=0.0,
                liquidity_risk=0.0,
                concentration_risk=0.0,
                leverage_risk=0.0,
                time_decay_risk=0.0
            )

    async def _store_kelly_calculation(self, symbol: str, result: Dict[str, Any]) -> None:
        """Store Kelly calculation in Redis"""
        key = f"kelly:{symbol}:latest"
        
        calculation_data = {
            'symbol': symbol,
            'timestamp': int(time.time() * 1000),
            'result': result
        }
        
        await self.redis_client.setex(
            key,
            ttl=3600,  # 1 hour TTL
            value=json.dumps(calculation_data, default=str)
        )
        
        # Store in time series
        ts_key = f"kelly:ts:{symbol}"
        await self.redis_client.zadd(
            ts_key,
            {json.dumps(calculation_data, default=str): calculation_data['timestamp']}
        )
        # Keep only last 1000 calculations
        await self.redis_client.zremrangebyrank(ts_key, 0, -1001)

    async def _log_risk_event(self, symbol: str, old_position: float, new_position: float, 
                            position_change: float) -> None:
        """Log significant risk events"""
        event_data = {
            'symbol': symbol,
            'timestamp': int(time.time() * 1000),
            'old_position': old_position,
            'new_position': new_position,
            'position_change': position_change,
            'portfolio_value': self.portfolio_value,
            'risk_level': 'HIGH' if abs(position_change) > self.portfolio_value * 0.05 else 'MEDIUM'
        }
        
        key = f"risk:events:latest"
        await self.redis_client.setex(
            key,
            ttl=86400,  # 24 hours TTL
            value=json.dumps(event_data, default=str)
        )
        
        # Update metrics
        self.metrics['risk_events'] += 1

    async def get_kelly_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get Kelly calculation history for a symbol"""
        try:
            ts_key = f"kelly:ts:{symbol}"
            calculations = await self.redis_client.zrevrange(ts_key, 0, limit - 1)
            
            history = []
            for calc in calculations:
                if isinstance(calc, bytes):
                    calc = calc.decode('utf-8')
                history.append(json.loads(calc))
            
            return history
            
        except Exception as e:
            logger.error(f"Error retrieving Kelly history: {e}")
            return []

    async def get_risk_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent risk events"""
        try:
            key = f"risk:events:latest"
            event_data = await self.redis_client.get(key)
            
            if event_data:
                return [json.loads(event_data.decode('utf-8'))]
            return []
            
        except Exception as e:
            logger.error(f"Error retrieving risk events: {e}")
            return []

    def update_market_data(self, symbol: str, price: float, timestamp: int) -> None:
        """Update market data for risk calculations"""
        self.kelly_calculator.market_analyzer.update_price(symbol, price, timestamp)

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.metrics,
            'current_positions': len(self.current_positions),
            'portfolio_value': self.portfolio_value,
            'max_portfolio_risk': self.max_portfolio_risk,
            'max_position_size': self.max_position_size
        }

    async def start(self) -> None:
        """Start risk engine"""
        logger.info("Risk engine started")

    async def stop(self) -> None:
        """Stop risk engine"""
        logger.info("Risk engine stopped")
