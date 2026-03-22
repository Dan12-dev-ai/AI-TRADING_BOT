"""
Medallion-X Circuit Breaker Module
Capital protection system with automatic trading halt
Production-ready implementation with configurable loss thresholds
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
import json

import redis.asyncio as redis

from ..config.settings import config
from ..utils.logging import get_logger

logger = get_logger(__name__)

class CircuitState(Enum):
    """Circuit breaker state enumeration"""
    CLOSED = "closed"      # Normal trading allowed
    OPEN = "open"          # Trading halted
    HALF_OPEN = "half_open"  # Testing if conditions improved

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    enabled: bool = True
    max_loss_percent: float = 5.0  # Maximum loss percentage
    time_window_minutes: int = 60  # Time window to check losses
    cooldown_minutes: int = 30     # Cooldown period after halt
    min_trades: int = 10           # Minimum trades before activation
    check_interval_seconds: int = 30  # Check interval

@dataclass
class TradingMetrics:
    """Trading metrics for circuit breaker"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    max_drawdown: float
    current_drawdown: float
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    sharpe_ratio: float
    last_trade_time: int
    period_start_time: int

@dataclass
class CircuitBreakerEvent:
    """Circuit breaker event record"""
    timestamp: int
    state: CircuitState
    trigger_reason: str
    metrics: TradingMetrics
    config: CircuitBreakerConfig
    cooldown_until: Optional[int] = None

class CircuitBreaker:
    """
    Circuit Breaker for Capital Protection
    - Automatic trading halt on excessive losses
    - Configurable loss thresholds and time windows
    - Cooldown periods and recovery mechanisms
    """
    
    def __init__(self, redis_client: redis.Redis, config_override: Optional[Dict[str, Any]] = None):
        self.redis_client = redis_client
        
        # Load configuration
        if config_override:
            self.config = CircuitBreakerConfig(**config_override)
        else:
            self.config = CircuitBreakerConfig(
                enabled=getattr(config.risk, 'circuit_breaker_enabled', True),
                max_loss_percent=getattr(config.risk, 'circuit_breaker_max_loss_percent', 5.0),
                time_window_minutes=getattr(config.risk, 'circuit_breaker_time_window_minutes', 60),
                cooldown_minutes=getattr(config.risk, 'circuit_breaker_cooldown_minutes', 30),
                min_trades=getattr(config.risk, 'circuit_breaker_min_trades', 10),
                check_interval_seconds=getattr(config.risk, 'circuit_breaker_check_interval', 30)
            )
        
        # Current state
        self.current_state = CircuitState.CLOSED
        self.cooldown_until = None
        self.last_check_time = 0
        
        # Metrics tracking
        self.metrics_history: List[TradingMetrics] = []
        self.max_history_length = 1000
        
        # Event history
        self.event_history: List[CircuitBreakerEvent] = []
        self.max_event_history = 100
        
        # Performance metrics
        self.performance_metrics = {
            'trades_processed': 0,
            'circuit_breaks_triggered': 0,
            'automatic_recoveries': 0,
            'manual_interventions': 0,
            'total_downtime_minutes': 0,
            'average_recovery_time_minutes': 0
        }
        
        logger.info(f"Circuit Breaker initialized: enabled={self.config.enabled}, max_loss={self.config.max_loss_percent}%")

    async def check_trading_permission(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Check if trading is allowed
        Returns permission status and reason if denied
        """
        if not self.config.enabled:
            return {
                'allowed': True,
                'state': self.current_state.value,
                'reason': 'Circuit breaker disabled',
                'metrics': None
            }
        
        current_time = int(time.time() * 1000)
        
        # Check if we're in cooldown
        if self.cooldown_until and current_time < self.cooldown_until:
            remaining_minutes = (self.cooldown_until - current_time) / (60 * 1000)
            return {
                'allowed': False,
                'state': self.current_state.value,
                'reason': f'Circuit breaker in cooldown. {remaining_minutes:.1f} minutes remaining.',
                'metrics': await self._get_current_metrics(),
                'cooldown_until': self.cooldown_until
            }
        
        # Get current metrics
        metrics = await self._get_current_metrics()
        
        # Check if conditions are met for trading
        if self.current_state == CircuitState.OPEN:
            # Check if we can transition to half-open
            if await self._can_transition_to_half_open(metrics):
                await self._transition_to_half_open("Cooldown period ended, testing trading conditions")
                return {
                    'allowed': True,
                    'state': self.current_state.value,
                    'reason': 'Circuit breaker transitioning to half-open for testing',
                    'metrics': metrics
                }
            else:
                return {
                    'allowed': False,
                    'state': self.current_state.value,
                    'reason': 'Circuit breaker active - trading halted',
                    'metrics': metrics
                }
        
        elif self.current_state == CircuitState.HALF_OPEN:
            # In half-open state, allow limited trading
            return {
                'allowed': True,
                'state': self.current_state.value,
                'reason': 'Circuit breaker in half-open state - limited trading allowed',
                'metrics': metrics,
                'warning': 'Trading monitored closely for recovery'
            }
        
        # Normal operation - check for circuit breaker conditions
        should_trigger = await self._should_trigger_circuit_breaker(metrics)
        
        if should_trigger:
            await self._trigger_circuit_breaker(metrics)
            return {
                'allowed': False,
                'state': self.current_state.value,
                'reason': f'Circuit breaker triggered: {should_trigger}',
                'metrics': metrics
            }
        
        return {
            'allowed': True,
            'state': self.current_state.value,
            'reason': 'Normal trading conditions',
            'metrics': metrics
        }

    async def record_trade(self, symbol: str, side: str, quantity: float, 
                          entry_price: float, exit_price: Optional[float] = None,
                          pnl: Optional[float] = None, commission: float = 0.0) -> None:
        """
        Record a trade for circuit breaker monitoring
        """
        try:
            current_time = int(time.time() * 1000)
            
            # Calculate PnL if not provided
            if pnl is None and exit_price is not None:
                if side.lower() == 'buy':
                    pnl = (exit_price - entry_price) * quantity - commission
                else:
                    pnl = (entry_price - exit_price) * quantity - commission
            
            # Store trade in Redis
            trade_data = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl or 0.0,
                'commission': commission,
                'timestamp': current_time
            }
            
            # Store in time series
            ts_key = f"circuit_breaker:trades:{symbol}"
            await self.redis_client.zadd(
                ts_key,
                {json.dumps(trade_data): current_time}
            )
            
            # Keep only trades within time window
            cutoff_time = current_time - (self.config.time_window_minutes * 60 * 1000)
            await self.redis_client.zremrangebyscore(ts_key, 0, cutoff_time)
            
            # Store latest trade
            latest_key = f"circuit_breaker:latest_trade:{symbol}"
            await self.redis_client.setex(
                latest_key,
                ttl=86400,  # 24 hours
                value=json.dumps(trade_data)
            )
            
            # Update performance metrics
            self.performance_metrics['trades_processed'] += 1
            
            # Check if we need to evaluate circuit breaker
            if self.current_state == CircuitState.HALF_OPEN:
                metrics = await self._get_current_metrics()
                if await self._can_recover(metrics):
                    await self._recover_from_circuit_breaker("Recovery conditions met")
                elif await self._should_re_trigger(metrics):
                    await self._trigger_circuit_breaker(metrics, "Re-test failed - losses continue")
            
        except Exception as e:
            logger.error(f"Error recording trade for circuit breaker: {e}")

    async def manual_override(self, action: str, reason: str) -> Dict[str, Any]:
        """
        Manual override for circuit breaker
        Actions: 'open', 'close', 'force_close'
        """
        try:
            current_time = int(time.time() * 1000)
            metrics = await self._get_current_metrics()
            
            if action == 'open':
                await self._trigger_circuit_breaker(metrics, f"Manual override: {reason}")
                self.performance_metrics['manual_interventions'] += 1
                
            elif action == 'close':
                await self._force_close_circuit_breaker(f"Manual override: {reason}")
                self.performance_metrics['manual_interventions'] += 1
                
            elif action == 'force_close':
                await self._force_close_circuit_breaker(f"Force close: {reason}")
                self.performance_metrics['manual_interventions'] += 1
                
            else:
                return {'success': False, 'error': f'Unknown action: {action}'}
            
            return {
                'success': True,
                'action': action,
                'reason': reason,
                'new_state': self.current_state.value,
                'timestamp': current_time
            }
            
        except Exception as e:
            logger.error(f"Error in manual override: {e}")
            return {'success': False, 'error': str(e)}

    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker status"""
        metrics = await self._get_current_metrics()
        
        return {
            'state': self.current_state.value,
            'config': asdict(self.config),
            'current_metrics': asdict(metrics) if metrics else None,
            'cooldown_until': self.cooldown_until,
            'performance_metrics': self.performance_metrics,
            'recent_events': [asdict(event) for event in self.event_history[-5:]],
            'health_status': await self._get_health_status(metrics)
        }

    async def _get_current_metrics(self) -> Optional[TradingMetrics]:
        """Calculate current trading metrics"""
        try:
            current_time = int(time.time() * 1000)
            cutoff_time = current_time - (self.config.time_window_minutes * 60 * 1000)
            
            # Get all trades from all symbols
            trade_keys = await self.redis_client.keys("circuit_breaker:trades:*")
            
            all_trades = []
            for key in trade_keys:
                trades = await self.redis_client.zrangebyscore(key, cutoff_time, current_time)
                for trade_data in trades:
                    if isinstance(trade_data, bytes):
                        trade_data = trade_data.decode('utf-8')
                    all_trades.append(json.loads(trade_data))
            
            if not all_trades:
                return None
            
            # Calculate metrics
            total_trades = len(all_trades)
            winning_trades = len([t for t in all_trades if t['pnl'] > 0])
            losing_trades = len([t for t in all_trades if t['pnl'] < 0])
            total_pnl = sum(t['pnl'] for t in all_trades)
            
            wins = [t['pnl'] for t in all_trades if t['pnl'] > 0]
            losses = [t['pnl'] for t in all_trades if t['pnl'] < 0]
            
            average_win = np.mean(wins) if wins else 0.0
            average_loss = np.mean(losses) if losses else 0.0
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
            
            # Calculate drawdown
            cumulative_pnl = np.cumsum([t['pnl'] for t in all_trades])
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = (cumulative_pnl - running_max)
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
            current_drawdown = drawdown[-1] if len(drawdown) > 0 else 0.0
            
            # Simple Sharpe ratio (annualized)
            returns = np.array([t['pnl'] for t in all_trades])
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0
            
            return TradingMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                total_pnl=total_pnl,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                win_rate=win_rate,
                average_win=average_win,
                average_loss=average_loss,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                last_trade_time=max(t['timestamp'] for t in all_trades),
                period_start_time=cutoff_time
            )
            
        except Exception as e:
            logger.error(f"Error calculating current metrics: {e}")
            return None

    async def _should_trigger_circuit_breaker(self, metrics: Optional[TradingMetrics]) -> Optional[str]:
        """Check if circuit breaker should be triggered"""
        if not metrics or metrics.total_trades < self.config.min_trades:
            return None
        
        # Check loss percentage
        loss_percent = abs(metrics.current_drawdown) * 100
        if loss_percent >= self.config.max_loss_percent:
            return f"Loss threshold exceeded: {loss_percent:.2f}% >= {self.config.max_loss_percent}%"
        
        # Check consecutive losses
        if metrics.win_rate < 0.2 and metrics.total_trades >= 20:  # Less than 20% win rate
            return f"Win rate too low: {metrics.win_rate:.2f} < 0.20"
        
        # Check extreme drawdown
        if abs(metrics.max_drawdown) >= self.config.max_loss_percent / 100:
            return f"Maximum drawdown exceeded: {abs(metrics.max_drawdown)*100:.2f}%"
        
        return None

    async def _can_transition_to_half_open(self, metrics: Optional[TradingMetrics]) -> bool:
        """Check if we can transition to half-open state"""
        if not self.cooldown_until:
            return False
        
        current_time = int(time.time() * 1000)
        return current_time >= self.cooldown_until

    async def _can_recover(self, metrics: Optional[TradingMetrics]) -> bool:
        """Check if we can recover from circuit breaker"""
        if not metrics:
            return False
        
        # Recovery conditions
        loss_percent = abs(metrics.current_drawdown) * 100
        recent_trades_ok = metrics.total_trades >= 5 and metrics.win_rate > 0.4
        
        return loss_percent < self.config.max_loss_percent * 0.8 and recent_trades_ok

    async def _should_re_trigger(self, metrics: Optional[TradingMetrics]) -> bool:
        """Check if we should re-trigger circuit breaker in half-open state"""
        if not metrics:
            return False
        
        loss_percent = abs(metrics.current_drawdown) * 100
        return loss_percent >= self.config.max_loss_percent

    async def _trigger_circuit_breaker(self, metrics: Optional[TradingMetrics], reason: str = "Loss threshold exceeded") -> None:
        """Trigger circuit breaker"""
        current_time = int(time.time() * 1000)
        
        self.current_state = CircuitState.OPEN
        self.cooldown_until = current_time + (self.config.cooldown_minutes * 60 * 1000)
        
        # Create event
        event = CircuitBreakerEvent(
            timestamp=current_time,
            state=self.current_state,
            trigger_reason=reason,
            metrics=metrics,
            config=self.config,
            cooldown_until=self.cooldown_until
        )
        
        self.event_history.append(event)
        if len(self.event_history) > self.max_event_history:
            self.event_history.pop(0)
        
        # Store in Redis
        await self._store_circuit_breaker_event(event)
        
        # Update metrics
        self.performance_metrics['circuit_breaks_triggered'] += 1
        
        logger.critical(f"🚨 CIRCUIT BREAKER TRIGGERED: {reason}")
        logger.critical(f"⏰ Cooldown until: {datetime.fromtimestamp(self.cooldown_until/1000)}")

    async def _transition_to_half_open(self, reason: str) -> None:
        """Transition to half-open state"""
        current_time = int(time.time() * 1000)
        
        self.current_state = CircuitState.HALF_OPEN
        
        # Create event
        event = CircuitBreakerEvent(
            timestamp=current_time,
            state=self.current_state,
            trigger_reason=reason,
            metrics=await self._get_current_metrics(),
            config=self.config
        )
        
        self.event_history.append(event)
        if len(self.event_history) > self.max_event_history:
            self.event_history.pop(0)
        
        await self._store_circuit_breaker_event(event)
        
        logger.warning(f"⚡ CIRCUIT BREAKER HALF-OPEN: {reason}")

    async def _recover_from_circuit_breaker(self, reason: str) -> None:
        """Recover from circuit breaker"""
        current_time = int(time.time() * 1000)
        
        old_state = self.current_state
        self.current_state = CircuitState.CLOSED
        self.cooldown_until = None
        
        # Calculate downtime
        if self.event_history:
            last_event = self.event_history[-1]
            downtime_minutes = (current_time - last_event.timestamp) / (60 * 1000)
            self.performance_metrics['total_downtime_minutes'] += downtime_minutes
            
            # Update average recovery time
            if self.performance_metrics['circuit_breaks_triggered'] > 0:
                self.performance_metrics['average_recovery_time_minutes'] = (
                    self.performance_metrics['total_downtime_minutes'] / 
                    self.performance_metrics['circuit_breaks_triggered']
                )
        
        self.performance_metrics['automatic_recoveries'] += 1
        
        # Create event
        event = CircuitBreakerEvent(
            timestamp=current_time,
            state=self.current_state,
            trigger_reason=reason,
            metrics=await self._get_current_metrics(),
            config=self.config
        )
        
        self.event_history.append(event)
        if len(self.event_history) > self.max_event_history:
            self.event_history.pop(0)
        
        await self._store_circuit_breaker_event(event)
        
        logger.info(f"✅ CIRCUIT BREAKER RECOVERED: {reason}")

    async def _force_close_circuit_breaker(self, reason: str) -> None:
        """Force close circuit breaker"""
        current_time = int(time.time() * 1000)
        
        old_state = self.current_state
        self.current_state = CircuitState.CLOSED
        self.cooldown_until = None
        
        # Create event
        event = CircuitBreakerEvent(
            timestamp=current_time,
            state=self.current_state,
            trigger_reason=reason,
            metrics=await self._get_current_metrics(),
            config=self.config
        )
        
        self.event_history.append(event)
        if len(self.event_history) > self.max_event_history:
            self.event_history.pop(0)
        
        await self._store_circuit_breaker_event(event)
        
        logger.info(f"🔓 CIRCUIT BREAKER FORCE CLOSED: {reason}")

    async def _store_circuit_breaker_event(self, event: CircuitBreakerEvent) -> None:
        """Store circuit breaker event in Redis"""
        try:
            event_data = asdict(event)
            
            # Store latest event
            await self.redis_client.setex(
                "circuit_breaker:latest_event",
                ttl=86400 * 7,  # 7 days
                value=json.dumps(event_data, default=str)
            )
            
            # Store in time series
            await self.redis_client.zadd(
                "circuit_breaker:events",
                {json.dumps(event_data, default=str): event.timestamp}
            )
            
            # Keep only last 100 events
            await self.redis_client.zremrangebyrank("circuit_breaker:events", 0, -101)
            
        except Exception as e:
            logger.error(f"Error storing circuit breaker event: {e}")

    async def _get_health_status(self, metrics: Optional[TradingMetrics]) -> str:
        """Get health status"""
        if not self.config.enabled:
            return "DISABLED"
        
        if self.current_state == CircuitState.OPEN:
            return "CRITICAL"
        elif self.current_state == CircuitState.HALF_OPEN:
            return "WARNING"
        elif not metrics or metrics.total_trades < self.config.min_trades:
            return "INSUFFICIENT_DATA"
        elif abs(metrics.current_drawdown) > self.config.max_loss_percent * 0.8:
            return "WARNING"
        else:
            return "HEALTHY"
