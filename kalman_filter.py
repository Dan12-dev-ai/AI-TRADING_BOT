"""
Medallion-X Kalman Filter Module
Advanced noise reduction and signal smoothing for financial time series
Production-ready implementation with adaptive filtering
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from scipy.linalg import inv, cholesky
from scipy.stats import multivariate_normal
import time

import redis.asyncio as redis

from ..config.settings import config
from ..utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class KalmanState:
    """Kalman filter state representation"""
    state_vector: np.ndarray  # [price, velocity, acceleration]
    covariance_matrix: np.ndarray
    timestamp: int
    symbol: str
    exchange: str

@dataclass
class FilteredData:
    """Filtered data output structure"""
    timestamp: int
    symbol: str
    exchange: str
    
    # Original values
    original_price: float
    original_volume: float
    
    # Filtered values
    filtered_price: float
    filtered_velocity: float
    filtered_acceleration: float
    
    # Uncertainty estimates
    price_uncertainty: float
    velocity_uncertainty: float
    
    # Signal quality metrics
    signal_to_noise_ratio: float
    innovation: float
    likelihood: float

class AdaptiveKalmanFilter:
    """
    Adaptive Kalman filter for financial time series
    - 3-state model: price, velocity, acceleration
    - Adaptive process and measurement noise
    - Real-time parameter estimation
    - Multi-exchange support
    """
    
    def __init__(self, symbol: str, exchange: str):
        self.symbol = symbol
        self.exchange = exchange
        
        # State dimension: [price, velocity, acceleration]
        self.state_dim = 3
        
        # Initialize state and covariance
        self.state = np.zeros(self.state_dim)
        self.covariance = np.eye(self.state_dim) * 1000  # High initial uncertainty
        
        # State transition matrix (constant acceleration model)
        dt = 1.0  # 1 second time step
        self.F = np.array([
            [1, dt, 0.5 * dt**2],
            [0, 1, dt],
            [0, 0, 1]
        ])
        
        # Measurement matrix (we only observe price)
        self.H = np.array([[1, 0, 0]])
        
        # Process noise covariance (adaptive)
        self.Q = self._initialize_process_noise()
        
        # Measurement noise covariance (adaptive)
        self.R = np.array([[100.0]])  # Initial measurement noise
        
        # Adaptive parameters
        self.innovation_window = []
        self.max_window_size = 100
        self.learning_rate = 0.01
        
        # Performance metrics
        self.metrics = {
            'filter_updates': 0,
            'average_innovation': 0.0,
            'average_likelihood': 0.0,
            'convergence_time': 0.0,
            'start_time': time.time()
        }

    def _initialize_process_noise(self) -> np.ndarray:
        """Initialize process noise covariance matrix"""
        # Process noise for constant acceleration model
        sigma_a = 0.1  # Acceleration noise standard deviation
        dt = 1.0
        
        Q = np.array([
            [dt**4/4, dt**3/2, dt**2/2],
            [dt**3/2, dt**2, dt],
            [dt**2/2, dt, 1]
        ]) * sigma_a**2
        
        return Q

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction step of Kalman filter"""
        # Predict state
        predicted_state = self.F @ self.state
        
        # Predict covariance
        predicted_covariance = self.F @ self.covariance @ self.F.T + self.Q
        
        return predicted_state, predicted_covariance

    def update(self, measurement: float, timestamp: int) -> FilteredData:
        """Update step with new measurement"""
        start_time = time.time()
        
        # Prediction
        predicted_state, predicted_covariance = self.predict()
        
        # Innovation (measurement residual)
        innovation = measurement - self.H @ predicted_state
        
        # Innovation covariance
        innovation_covariance = self.H @ predicted_covariance @ self.H.T + self.R
        
        # Kalman gain
        kalman_gain = predicted_covariance @ self.H.T @ inv(innovation_covariance)
        
        # Update state
        self.state = predicted_state + kalman_gain.flatten() * innovation
        
        # Update covariance
        self.covariance = (np.eye(self.state_dim) - kalman_gain @ self.H) @ predicted_covariance
        
        # Calculate likelihood
        likelihood = multivariate_normal.pdf(
            innovation, 
            mean=0, 
            cov=innovation_covariance
        )
        
        # Signal to noise ratio
        signal_to_noise = abs(self.state[0]) / np.sqrt(self.covariance[0, 0]) if self.covariance[0, 0] > 0 else 0
        
        # Adaptive noise estimation
        self._adapt_parameters(innovation, innovation_covariance)
        
        # Update metrics
        self.metrics['filter_updates'] += 1
        self.metrics['average_innovation'] = (
            (self.metrics['average_innovation'] * (self.metrics['filter_updates'] - 1) + abs(innovation)) /
            self.metrics['filter_updates']
        )
        self.metrics['average_likelihood'] = (
            (self.metrics['average_likelihood'] * (self.metrics['filter_updates'] - 1) + likelihood) /
            self.metrics['filter_updates']
        )
        
        # Create filtered data output
        filtered_data = FilteredData(
            timestamp=timestamp,
            symbol=self.symbol,
            exchange=self.exchange,
            original_price=measurement,
            original_volume=0.0,  # Will be updated separately
            filtered_price=float(self.state[0]),
            filtered_velocity=float(self.state[1]),
            filtered_acceleration=float(self.state[2]),
            price_uncertainty=float(np.sqrt(self.covariance[0, 0])),
            velocity_uncertainty=float(np.sqrt(self.covariance[1, 1])),
            signal_to_noise_ratio=float(signal_to_noise),
            innovation=float(innovation),
            likelihood=float(likelihood)
        )
        
        # Update processing time
        processing_time = time.time() - start_time
        if processing_time > 0.001:  # Log if > 1ms
            logger.debug(f"Kalman filter update took {processing_time*1000:.2f}ms")
        
        return filtered_data

    def _adapt_parameters(self, innovation: float, innovation_covariance: np.ndarray) -> None:
        """Adapt process and measurement noise parameters"""
        # Store innovation for window
        self.innovation_window.append(innovation)
        
        # Maintain window size
        if len(self.innovation_window) > self.max_window_size:
            self.innovation_window.pop(0)
        
        # Update measurement noise based on innovation statistics
        if len(self.innovation_window) >= 10:
            innovation_std = np.std(self.innovation_window)
            
            # Adaptive R (measurement noise)
            target_R = innovation_std**2
            self.R = (1 - self.learning_rate) * self.R + self.learning_rate * target_R
            
            # Adaptive Q (process noise)
            if abs(innovation) > 2 * innovation_std:  # Large innovation
                # Increase process noise
                self.Q *= 1.1
            else:
                # Decrease process noise slightly
                self.Q *= 0.99

    def get_state(self) -> KalmanState:
        """Get current filter state"""
        return KalmanState(
            state_vector=self.state.copy(),
            covariance_matrix=self.covariance.copy(),
            timestamp=int(time.time() * 1000),
            symbol=self.symbol,
            exchange=self.exchange
        )

    def reset(self) -> None:
        """Reset filter to initial state"""
        self.state = np.zeros(self.state_dim)
        self.covariance = np.eye(self.state_dim) * 1000
        self.innovation_window.clear()
        self.metrics['filter_updates'] = 0
        self.metrics['start_time'] = time.time()

class KalmanFilterManager:
    """
    Manager for multiple Kalman filters across symbols and exchanges
    - Coordinates multiple filter instances
    - Handles Redis storage and retrieval
    - Provides batch processing capabilities
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.filters: Dict[str, AdaptiveKalmanFilter] = {}
        self.is_running = False
        
        # Performance metrics
        self.metrics = {
            'total_filters': 0,
            'active_filters': 0,
            'total_updates': 0,
            'average_update_time_ms': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    def get_filter_key(self, symbol: str, exchange: str) -> str:
        """Generate unique key for filter"""
        return f"{exchange}:{symbol}"

    def get_or_create_filter(self, symbol: str, exchange: str) -> AdaptiveKalmanFilter:
        """Get existing filter or create new one"""
        key = self.get_filter_key(symbol, exchange)
        
        if key not in self.filters:
            self.filters[key] = AdaptiveKalmanFilter(symbol, exchange)
            self.metrics['total_filters'] += 1
            logger.info(f"Created new Kalman filter for {key}")
        
        self.metrics['active_filters'] = len(self.filters)
        return self.filters[key]

    async def process_price_update(self, symbol: str, exchange: str, price: float, timestamp: int) -> FilteredData:
        """Process a single price update through Kalman filter"""
        start_time = time.time()
        
        try:
            # Get or create filter
            kalman_filter = self.get_or_create_filter(symbol, exchange)
            
            # Update filter with new measurement
            filtered_data = kalman_filter.update(price, timestamp)
            
            # Store in Redis
            await self._store_filtered_data(filtered_data)
            
            # Update metrics
            self.metrics['total_updates'] += 1
            update_time = (time.time() - start_time) * 1000
            self.metrics['average_update_time_ms'] = (
                (self.metrics['average_update_time_ms'] * (self.metrics['total_updates'] - 1) + update_time) /
                self.metrics['total_updates']
            )
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error processing price update for {exchange}:{symbol}: {e}")
            raise

    async def process_batch_updates(self, updates: List[Dict[str, Any]]) -> List[FilteredData]:
        """Process multiple price updates in batch"""
        results = []
        
        for update in updates:
            try:
                filtered_data = await self.process_price_update(
                    symbol=update['symbol'],
                    exchange=update['exchange'],
                    price=update['price'],
                    timestamp=update['timestamp']
                )
                results.append(filtered_data)
                
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                continue
        
        return results

    async def _store_filtered_data(self, data: FilteredData) -> None:
        """Store filtered data in Redis"""
        key = f"kalman:filtered:{data.exchange}:{data.symbol}:latest"
        
        # Convert to JSON-serializable format
        data_dict = asdict(data)
        
        # Store with TTL
        await self.redis_client.setex(
            key,
            ttl=3600,  # 1 hour TTL
            value=str(data_dict).replace("'", '"')  # Simple JSON conversion
        )
        
        # Store in time series for history
        ts_key = f"kalman:ts:{data.exchange}:{data.symbol}"
        await self.redis_client.zadd(
            ts_key,
            {str(data_dict).replace("'", '"'): data.timestamp}
        )
        # Keep only last 1000 records
        await self.redis_client.zremrangebyrank(ts_key, 0, -1001)

    async def get_latest_filtered_data(self, symbol: str, exchange: str) -> Optional[FilteredData]:
        """Get latest filtered data for a symbol"""
        try:
            key = f"kalman:filtered:{exchange}:{symbol}:latest"
            data = await self.redis_client.get(key)
            
            if data:
                self.metrics['cache_hits'] += 1
                # Parse data back to FilteredData object
                data_dict = eval(data.decode('utf-8'))
                return FilteredData(**data_dict)
            else:
                self.metrics['cache_misses'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving filtered data: {e}")
            return None

    async def get_filter_history(self, symbol: str, exchange: str, limit: int = 100) -> List[FilteredData]:
        """Get historical filtered data"""
        try:
            ts_key = f"kalman:ts:{exchange}:{symbol}"
            data_points = await self.redis_client.zrevrange(ts_key, 0, limit - 1)
            
            results = []
            for data in data_points:
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                data_dict = eval(data)
                results.append(FilteredData(**data_dict))
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving filter history: {e}")
            return []

    async def reset_filter(self, symbol: str, exchange: str) -> None:
        """Reset a specific filter"""
        key = self.get_filter_key(symbol, exchange)
        
        if key in self.filters:
            self.filters[key].reset()
            logger.info(f"Reset Kalman filter for {key}")

    async def reset_all_filters(self) -> None:
        """Reset all filters"""
        for kalman_filter in self.filters.values():
            kalman_filter.reset()
        
        logger.info("Reset all Kalman filters")

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        base_metrics = self.metrics.copy()
        
        # Add filter-specific metrics
        if self.filters:
            filter_metrics = []
            for key, kalman_filter in self.filters.items():
                filter_metrics.append({
                    'key': key,
                    'updates': kalman_filter.metrics['filter_updates'],
                    'avg_innovation': kalman_filter.metrics['average_innovation'],
                    'avg_likelihood': kalman_filter.metrics['average_likelihood']
                })
            base_metrics['filter_details'] = filter_metrics
        
        return base_metrics

    async def start(self) -> None:
        """Start Kalman filter manager"""
        self.is_running = True
        logger.info("Kalman filter manager started")

    async def stop(self) -> None:
        """Stop Kalman filter manager"""
        self.is_running = False
        logger.info("Kalman filter manager stopped")
