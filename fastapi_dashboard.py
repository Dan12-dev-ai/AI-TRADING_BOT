"""
Medallion-X FastAPI Dashboard Module
Real-time monitoring and control interface
Production-ready dashboard with WebSocket updates and comprehensive metrics
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

import redis.asyncio as redis

from ..config.settings import config
from ..utils.logging import get_logger

logger = get_logger(__name__)

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class SystemStatus(Enum):
    """System status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"

@dataclass
class SystemMetrics:
    """Comprehensive system metrics"""
    timestamp: int
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_connections: int
    total_requests: int
    error_rate: float
    average_response_time: float
    uptime_seconds: float

@dataclass
class TradingMetrics:
    """Trading performance metrics"""
    timestamp: int
    total_trades: int
    successful_trades: int
    failed_trades: int
    total_pnl: float
    daily_pnl: float
    win_rate: float
    average_trade_duration: float
    sharpe_ratio: float
    max_drawdown: float
    current_positions: int
    portfolio_value: float
    risk_exposure: float

@dataclass
class Alert:
    """System alert structure"""
    id: str
    timestamp: int
    level: AlertLevel
    source: str
    message: str
    details: Dict[str, Any]
    resolved: bool
    resolved_at: Optional[int] = None

# Pydantic models for API
class SystemMetricsResponse(BaseModel):
    timestamp: int
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_connections: int
    total_requests: int
    error_rate: float
    average_response_time: float
    uptime_seconds: float

class TradingMetricsResponse(BaseModel):
    timestamp: int
    total_trades: int
    successful_trades: int
    failed_trades: int
    total_pnl: float
    daily_pnl: float
    win_rate: float
    average_trade_duration: float
    sharpe_ratio: float
    max_drawdown: float
    current_positions: int
    portfolio_value: float
    risk_exposure: float

class AlertResponse(BaseModel):
    id: str
    timestamp: int
    level: str
    source: str
    message: str
    details: Dict[str, Any]
    resolved: bool
    resolved_at: Optional[int] = None

class SystemStatusResponse(BaseModel):
    status: str
    uptime: int
    version: str
    last_update: int
    components: Dict[str, str]

class WebSocketManager:
    """WebSocket connection manager for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_metadata[websocket] = {
            'client_id': client_id,
            'connected_at': time.time(),
            'last_ping': time.time()
        }
        logger.info(f"WebSocket client connected: {client_id}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.connection_metadata:
            client_id = self.connection_metadata[websocket]['client_id']
            del self.connection_metadata[websocket]
            logger.info(f"WebSocket client disconnected: {client_id}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific client"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        """Broadcast message to all connected clients"""
        disconnected_clients = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
                # Update last ping
                if connection in self.connection_metadata:
                    self.connection_metadata[connection]['last_ping'] = time.time()
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected_clients.append(connection)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.disconnect(client)

    async def ping_clients(self):
        """Ping all clients to check connectivity"""
        ping_message = json.dumps({
            'type': 'ping',
            'timestamp': int(time.time() * 1000)
        })
        await self.broadcast(ping_message)

    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)

class MetricsCollector:
    """Metrics collection and aggregation"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.collection_interval = 5  # seconds
        
        # Metrics storage
        self.system_metrics_history: List[SystemMetrics] = []
        self.trading_metrics_history: List[TradingMetrics] = []
        self.max_history_length = 1000
        
        # Performance tracking
        self.metrics = {
            'collections_performed': 0,
            'collection_errors': 0,
            'average_collection_time_ms': 0.0
        }

    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics"""
        start_time = time.time()
        
        try:
            import psutil
            
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # Active connections (simplified)
            active_connections = len(psutil.net_connections())
            
            # Calculate uptime (simplified)
            uptime_seconds = time.time() - psutil.boot_time()
            
            # HTTP metrics (would come from FastAPI middleware)
            total_requests = await self._get_total_requests()
            error_rate = await self._get_error_rate()
            average_response_time = await self._get_average_response_time()
            
            metrics = SystemMetrics(
                timestamp=int(time.time() * 1000),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                active_connections=active_connections,
                total_requests=total_requests,
                error_rate=error_rate,
                average_response_time=average_response_time,
                uptime_seconds=uptime_seconds
            )
            
            # Store in history
            self.system_metrics_history.append(metrics)
            if len(self.system_metrics_history) > self.max_history_length:
                self.system_metrics_history = self.system_metrics_history[-self.max_history_length:]
            
            # Store in Redis
            await self._store_system_metrics(metrics)
            
            # Update performance metrics
            collection_time = (time.time() - start_time) * 1000
            self._update_collection_metrics(collection_time)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            self.metrics['collection_errors'] += 1
            raise

    async def collect_trading_metrics(self) -> TradingMetrics:
        """Collect trading performance metrics"""
        start_time = time.time()
        
        try:
            # Get trading data from Redis
            trading_data = await self._get_trading_data()
            
            metrics = TradingMetrics(
                timestamp=int(time.time() * 1000),
                total_trades=trading_data.get('total_trades', 0),
                successful_trades=trading_data.get('successful_trades', 0),
                failed_trades=trading_data.get('failed_trades', 0),
                total_pnl=trading_data.get('total_pnl', 0.0),
                daily_pnl=trading_data.get('daily_pnl', 0.0),
                win_rate=trading_data.get('win_rate', 0.0),
                average_trade_duration=trading_data.get('average_trade_duration', 0.0),
                sharpe_ratio=trading_data.get('sharpe_ratio', 0.0),
                max_drawdown=trading_data.get('max_drawdown', 0.0),
                current_positions=trading_data.get('current_positions', 0),
                portfolio_value=trading_data.get('portfolio_value', 0.0),
                risk_exposure=trading_data.get('risk_exposure', 0.0)
            )
            
            # Store in history
            self.trading_metrics_history.append(metrics)
            if len(self.trading_metrics_history) > self.max_history_length:
                self.trading_metrics_history = self.trading_metrics_history[-self.max_history_length:]
            
            # Store in Redis
            await self._store_trading_metrics(metrics)
            
            # Update performance metrics
            collection_time = (time.time() - start_time) * 1000
            self._update_collection_metrics(collection_time)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting trading metrics: {e}")
            self.metrics['collection_errors'] += 1
            raise

    async def _get_total_requests(self) -> int:
        """Get total HTTP requests count"""
        try:
            # This would be tracked by FastAPI middleware
            data = await self.redis_client.get("metrics:http:total_requests")
            return int(data) if data else 0
        except:
            return 0

    async def _get_error_rate(self) -> float:
        """Get HTTP error rate"""
        try:
            total = await self._get_total_requests()
            errors = await self.redis_client.get("metrics:http:error_count")
            error_count = int(errors) if errors else 0
            return (error_count / total) if total > 0 else 0.0
        except:
            return 0.0

    async def _get_average_response_time(self) -> float:
        """Get average HTTP response time"""
        try:
            data = await self.redis_client.get("metrics:http:avg_response_time")
            return float(data) if data else 0.0
        except:
            return 0.0

    async def _get_trading_data(self) -> Dict[str, Any]:
        """Get trading data from Redis"""
        try:
            # Get latest trading metrics
            data = await self.redis_client.get("trading:metrics:latest")
            return json.loads(data.decode('utf-8')) if data else {}
        except:
            return {}

    async def _store_system_metrics(self, metrics: SystemMetrics) -> None:
        """Store system metrics in Redis"""
        key = "metrics:system:latest"
        await self.redis_client.setex(
            key,
            ttl=3600,  # 1 hour TTL
            value=json.dumps(asdict(metrics))
        )
        
        # Store in time series
        ts_key = "metrics:system:ts"
        await self.redis_client.zadd(
            ts_key,
            {json.dumps(asdict(metrics)): metrics.timestamp}
        )
        # Keep only last 1000 records
        await self.redis_client.zremrangebyrank(ts_key, 0, -1001)

    async def _store_trading_metrics(self, metrics: TradingMetrics) -> None:
        """Store trading metrics in Redis"""
        key = "metrics:trading:latest"
        await self.redis_client.setex(
            key,
            ttl=3600,  # 1 hour TTL
            value=json.dumps(asdict(metrics))
        )
        
        # Store in time series
        ts_key = "metrics:trading:ts"
        await self.redis_client.zadd(
            ts_key,
            {json.dumps(asdict(metrics)): metrics.timestamp}
        )
        # Keep only last 1000 records
        await self.redis_client.zremrangebyrank(ts_key, 0, -1001)

    def _update_collection_metrics(self, collection_time: float) -> None:
        """Update metrics collection performance"""
        self.metrics['collections_performed'] += 1
        
        current_avg = self.metrics['average_collection_time_ms']
        count = self.metrics['collections_performed']
        new_avg = (current_avg * (count - 1) + collection_time) / count
        self.metrics['average_collection_time_ms'] = new_avg

    async def get_metrics_history(self, metric_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical metrics"""
        try:
            if metric_type == "system":
                ts_key = "metrics:system:ts"
            elif metric_type == "trading":
                ts_key = "metrics:trading:ts"
            else:
                return []
            
            data_points = await self.redis_client.zrevrange(ts_key, 0, limit - 1)
            
            history = []
            for data in data_points:
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                history.append(json.loads(data))
            
            return history
            
        except Exception as e:
            logger.error(f"Error retrieving metrics history: {e}")
            return []

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get metrics collection performance"""
        return self.metrics.copy()

class AlertManager:
    """Alert management system"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.active_alerts: Dict[str, Alert] = {}
        
        # Alert thresholds
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'error_rate': 0.05,
            'response_time': 1000.0,  # ms
            'drawdown': 0.1,  # 10%
            'risk_exposure': 0.8  # 80%
        }

    async def check_system_alerts(self, system_metrics: SystemMetrics) -> List[Alert]:
        """Check for system-related alerts"""
        alerts = []
        
        # CPU usage alert
        if system_metrics.cpu_usage > self.thresholds['cpu_usage']:
            alert = await self._create_alert(
                AlertLevel.WARNING,
                "system",
                f"High CPU usage: {system_metrics.cpu_usage:.1f}%",
                {
                    'metric': 'cpu_usage',
                    'value': system_metrics.cpu_usage,
                    'threshold': self.thresholds['cpu_usage']
                }
            )
            alerts.append(alert)
        
        # Memory usage alert
        if system_metrics.memory_usage > self.thresholds['memory_usage']:
            alert = await self._create_alert(
                AlertLevel.WARNING,
                "system",
                f"High memory usage: {system_metrics.memory_usage:.1f}%",
                {
                    'metric': 'memory_usage',
                    'value': system_metrics.memory_usage,
                    'threshold': self.thresholds['memory_usage']
                }
            )
            alerts.append(alert)
        
        # Disk usage alert
        if system_metrics.disk_usage > self.thresholds['disk_usage']:
            alert = await self._create_alert(
                AlertLevel.ERROR,
                "system",
                f"High disk usage: {system_metrics.disk_usage:.1f}%",
                {
                    'metric': 'disk_usage',
                    'value': system_metrics.disk_usage,
                    'threshold': self.thresholds['disk_usage']
                }
            )
            alerts.append(alert)
        
        # Error rate alert
        if system_metrics.error_rate > self.thresholds['error_rate']:
            alert = await self._create_alert(
                AlertLevel.ERROR,
                "system",
                f"High error rate: {system_metrics.error_rate:.2%}",
                {
                    'metric': 'error_rate',
                    'value': system_metrics.error_rate,
                    'threshold': self.thresholds['error_rate']
                }
            )
            alerts.append(alert)
        
        # Response time alert
        if system_metrics.average_response_time > self.thresholds['response_time']:
            alert = await self._create_alert(
                AlertLevel.WARNING,
                "system",
                f"High response time: {system_metrics.average_response_time:.1f}ms",
                {
                    'metric': 'response_time',
                    'value': system_metrics.average_response_time,
                    'threshold': self.thresholds['response_time']
                }
            )
            alerts.append(alert)
        
        return alerts

    async def check_trading_alerts(self, trading_metrics: TradingMetrics) -> List[Alert]:
        """Check for trading-related alerts"""
        alerts = []
        
        # Drawdown alert
        if abs(trading_metrics.max_drawdown) > self.thresholds['drawdown']:
            alert = await self._create_alert(
                AlertLevel.ERROR,
                "trading",
                f"High drawdown: {trading_metrics.max_drawdown:.2%}",
                {
                    'metric': 'max_drawdown',
                    'value': trading_metrics.max_drawdown,
                    'threshold': self.thresholds['drawdown']
                }
            )
            alerts.append(alert)
        
        # Risk exposure alert
        if trading_metrics.risk_exposure > self.thresholds['risk_exposure']:
            alert = await self._create_alert(
                AlertLevel.WARNING,
                "trading",
                f"High risk exposure: {trading_metrics.risk_exposure:.2%}",
                {
                    'metric': 'risk_exposure',
                    'value': trading_metrics.risk_exposure,
                    'threshold': self.thresholds['risk_exposure']
                }
            )
            alerts.append(alert)
        
        return alerts

    async def _create_alert(self, level: AlertLevel, source: str, message: str, details: Dict[str, Any]) -> Alert:
        """Create and store alert"""
        alert_id = str(uuid.uuid4())
        alert = Alert(
            id=alert_id,
            timestamp=int(time.time() * 1000),
            level=level,
            source=source,
            message=message,
            details=details,
            resolved=False
        )
        
        # Store active alert
        self.active_alerts[alert_id] = alert
        
        # Store in Redis
        await self._store_alert(alert)
        
        return alert

    async def _store_alert(self, alert: Alert) -> None:
        """Store alert in Redis"""
        key = f"alerts:{alert.id}"
        
        alert_dict = asdict(alert)
        alert_dict['level'] = alert.level.value
        
        await self.redis_client.setex(
            key,
            ttl=86400 * 7,  # 7 days TTL
            value=json.dumps(alert_dict, default=str)
        )
        
        # Store in active alerts set
        if not alert.resolved:
            await self.redis_client.sadd("alerts:active", alert.id)
            await self.redis_client.expire("alerts:active", 86400)

    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.resolved_at = int(time.time() * 1000)
        
        # Update in Redis
        await self._store_alert(alert)
        await self.redis_client.srem("alerts:active", alert_id)
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        return True

    async def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        try:
            alert_ids = await self.redis_client.smembers("alerts:active")
            
            alerts = []
            for alert_id in alert_ids:
                if isinstance(alert_id, bytes):
                    alert_id = alert_id.decode('utf-8')
                
                data = await self.redis_client.get(f"alerts:{alert_id}")
                if data:
                    alert_dict = json.loads(data.decode('utf-8'))
                    alert_dict['level'] = AlertLevel(alert_dict['level'])
                    alerts.append(Alert(**alert_dict))
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error retrieving active alerts: {e}")
            return []

    async def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history"""
        try:
            # Get all alert keys
            alert_keys = await self.redis_client.keys("alerts:*")
            
            alerts = []
            for key in alert_keys[:limit]:
                data = await self.redis_client.get(key)
                if data:
                    alert_dict = json.loads(data.decode('utf-8'))
                    alert_dict['level'] = AlertLevel(alert_dict['level'])
                    alerts.append(Alert(**alert_dict))
            
            # Sort by timestamp
            alerts.sort(key=lambda x: x.timestamp, reverse=True)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error retrieving alert history: {e}")
            return []

class FastAPIDashboard:
    """
    FastAPI Dashboard for Medallion-X
    - Real-time WebSocket updates
    - Comprehensive metrics
    - Alert management
    - System control interface
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.app = FastAPI(
            title="Medallion-X Dashboard",
            description="Real-time monitoring and control for Medallion-X trading bot",
            version="1.0.0"
        )
        
        # Components
        self.websocket_manager = WebSocketManager()
        self.metrics_collector = MetricsCollector(redis_client)
        self.alert_manager = AlertManager(redis_client)
        
        # System status
        self.system_status = SystemStatus.HEALTHY
        self.start_time = time.time()
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        
        # Background tasks
        self.background_tasks = BackgroundTasks()

    def _setup_middleware(self) -> None:
        """Setup FastAPI middleware"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Request tracking middleware (simplified)
        @self.app.middleware("http")
        async def add_request_tracking(request, call_next):
            start_time = time.time()
            
            response = await call_next(request)
            
            # Track request metrics
            process_time = time.time() - start_time
            
            # Update Redis metrics (simplified)
            await self.redis_client.incr("metrics:http:total_requests")
            
            if response.status_code >= 400:
                await self.redis_client.incr("metrics:http:error_count")
            
            # Update average response time
            await self._update_response_time(process_time)
            
            response.headers["X-Process-Time"] = str(process_time)
            return response

    async def _update_response_time(self, response_time: float) -> None:
        """Update average response time metric"""
        try:
            current_data = await self.redis_client.get("metrics:http:avg_response_time")
            current_avg = float(current_data) if current_data else 0.0
            
            # Simple exponential moving average
            alpha = 0.1
            new_avg = alpha * response_time + (1 - alpha) * current_avg
            
            await self.redis_client.setex(
                "metrics:http:avg_response_time",
                ttl=3600,
                value=str(new_avg)
            )
        except Exception as e:
            logger.error(f"Error updating response time: {e}")

    def _setup_routes(self) -> None:
        """Setup API routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            """Serve dashboard HTML"""
            return self._get_dashboard_html()
        
        @self.app.get("/api/status", response_model=SystemStatusResponse)
        async def get_system_status():
            """Get system status"""
            return SystemStatusResponse(
                status=self.system_status.value,
                uptime=int(time.time() - self.start_time),
                version="1.0.0",
                last_update=int(time.time() * 1000),
                components={
                    "data_pipeline": "healthy",
                    "ai_engine": "healthy",
                    "risk_engine": "healthy",
                    "execution_engine": "healthy"
                }
            )
        
        @self.app.get("/api/metrics/system", response_model=SystemMetricsResponse)
        async def get_system_metrics():
            """Get latest system metrics"""
            try:
                metrics = await self.metrics_collector.collect_system_metrics()
                return SystemMetricsResponse(**asdict(metrics))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/metrics/trading", response_model=TradingMetricsResponse)
        async def get_trading_metrics():
            """Get latest trading metrics"""
            try:
                metrics = await self.metrics_collector.collect_trading_metrics()
                return TradingMetricsResponse(**asdict(metrics))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/metrics/history/{metric_type}")
        async def get_metrics_history(metric_type: str, limit: int = 100):
            """Get historical metrics"""
            try:
                history = await self.metrics_collector.get_metrics_history(metric_type, limit)
                return {"metric_type": metric_type, "history": history}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/alerts", response_model=List[AlertResponse])
        async def get_active_alerts():
            """Get active alerts"""
            try:
                alerts = await self.alert_manager.get_active_alerts()
                return [AlertResponse(**asdict(alert)) for alert in alerts]
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/alerts/{alert_id}/resolve")
        async def resolve_alert(alert_id: str):
            """Resolve an alert"""
            try:
                success = await self.alert_manager.resolve_alert(alert_id)
                if success:
                    return {"message": "Alert resolved successfully"}
                else:
                    raise HTTPException(status_code=404, detail="Alert not found")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/alerts/history")
        async def get_alert_history(limit: int = 100):
            """Get alert history"""
            try:
                alerts = await self.alert_manager.get_alert_history(limit)
                return [AlertResponse(**asdict(alert)) for alert in alerts]
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            client_id = str(uuid.uuid4())
            await self.websocket_manager.connect(websocket, client_id)
            
            try:
                while True:
                    # Receive message from client
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Handle different message types
                    if message.get('type') == 'subscribe':
                        # Handle subscription requests
                        await self._handle_subscription(websocket, message)
                    elif message.get('type') == 'pong':
                        # Handle pong response
                        continue
                    else:
                        # Echo back or handle other messages
                        await websocket.send_text(json.dumps({
                            'type': 'echo',
                            'data': message
                        }))
                        
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.websocket_manager.disconnect(websocket)

    async def _handle_subscription(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """Handle WebSocket subscription requests"""
        subscription_type = message.get('subscription')
        
        if subscription_type == 'metrics':
            # Send current metrics
            try:
                system_metrics = await self.metrics_collector.collect_system_metrics()
                trading_metrics = await self.metrics_collector.collect_trading_metrics()
                
                await websocket.send_text(json.dumps({
                    'type': 'metrics_update',
                    'system': asdict(system_metrics),
                    'trading': asdict(trading_metrics),
                    'timestamp': int(time.time() * 1000)
                }))
            except Exception as e:
                logger.error(f"Error sending metrics: {e}")
        
        elif subscription_type == 'alerts':
            # Send active alerts
            try:
                alerts = await self.alert_manager.get_active_alerts()
                await websocket.send_text(json.dumps({
                    'type': 'alerts_update',
                    'alerts': [asdict(alert) for alert in alerts],
                    'timestamp': int(time.time() * 1000)
                }))
            except Exception as e:
                logger.error(f"Error sending alerts: {e}")

    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Medallion-X Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
                .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .metric-value { font-size: 2em; font-weight: bold; color: #333; }
                .metric-label { color: #666; margin-top: 5px; }
                .status-healthy { color: #4CAF50; }
                .status-warning { color: #FF9800; }
                .status-error { color: #F44336; }
                .alerts { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .alert { padding: 10px; margin: 5px 0; border-radius: 5px; }
                .alert-warning { background: #FFF3CD; border-left: 4px solid #FF9800; }
                .alert-error { background: #F8D7DA; border-left: 4px solid #F44336; }
                .connection-status { position: fixed; top: 20px; right: 20px; padding: 10px; border-radius: 5px; }
                .connected { background: #4CAF50; color: white; }
                .disconnected { background: #F44336; color: white; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 Medallion-X Trading Bot Dashboard</h1>
                    <p>Real-time monitoring and control interface</p>
                </div>
                
                <div class="connection-status" id="connectionStatus">
                    <span id="connectionText">Connecting...</span>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value" id="cpuUsage">--</div>
                        <div class="metric-label">CPU Usage</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="memoryUsage">--</div>
                        <div class="metric-label">Memory Usage</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="totalTrades">--</div>
                        <div class="metric-label">Total Trades</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="totalPnL">--</div>
                        <div class="metric-label">Total P&L</div>
                    </div>
                </div>
                
                <div class="alerts">
                    <h2>🚨 Active Alerts</h2>
                    <div id="alertsList">
                        <p>No active alerts</p>
                    </div>
                </div>
            </div>
            
            <script>
                const ws = new WebSocket('ws://localhost:8000/ws');
                const connectionStatus = document.getElementById('connectionStatus');
                const connectionText = document.getElementById('connectionText');
                
                ws.onopen = function() {
                    connectionStatus.className = 'connection-status connected';
                    connectionText.textContent = 'Connected';
                    
                    // Subscribe to metrics and alerts
                    ws.send(JSON.stringify({type: 'subscribe', subscription: 'metrics'}));
                    ws.send(JSON.stringify({type: 'subscribe', subscription: 'alerts'}));
                };
                
                ws.onclose = function() {
                    connectionStatus.className = 'connection-status disconnected';
                    connectionText.textContent = 'Disconnected';
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'metrics_update') {
                        updateMetrics(data.system, data.trading);
                    } else if (data.type === 'alerts_update') {
                        updateAlerts(data.alerts);
                    } else if (data.type === 'ping') {
                        ws.send(JSON.stringify({type: 'pong'}));
                    }
                };
                
                function updateMetrics(system, trading) {
                    document.getElementById('cpuUsage').textContent = system.cpu_usage.toFixed(1) + '%';
                    document.getElementById('memoryUsage').textContent = system.memory_usage.toFixed(1) + '%';
                    document.getElementById('totalTrades').textContent = trading.total_trades;
                    document.getElementById('totalPnL').textContent = '$' + trading.total_pnl.toFixed(2);
                }
                
                function updateAlerts(alerts) {
                    const alertsList = document.getElementById('alertsList');
                    
                    if (alerts.length === 0) {
                        alertsList.innerHTML = '<p>No active alerts</p>';
                        return;
                    }
                    
                    alertsList.innerHTML = alerts.map(alert => 
                        `<div class="alert alert-${alert.level}">
                            <strong>${alert.source.toUpperCase()}:</strong> ${alert.message}
                            <br><small>${new Date(alert.timestamp).toLocaleString()}</small>
                        </div>`
                    ).join('');
                }
                
                // Auto-reconnect
                setInterval(function() {
                    if (ws.readyState === WebSocket.CLOSED) {
                        location.reload();
                    }
                }, 5000);
            </script>
        </body>
        </html>
        """

    async def start_metrics_collection(self) -> None:
        """Start background metrics collection"""
        while True:
            try:
                # Collect system metrics
                system_metrics = await self.metrics_collector.collect_system_metrics()
                
                # Collect trading metrics
                trading_metrics = await self.metrics_collector.collect_trading_metrics()
                
                # Check for alerts
                system_alerts = await self.alert_manager.check_system_alerts(system_metrics)
                trading_alerts = await self.alert_manager.check_trading_alerts(trading_metrics)
                
                # Broadcast updates to WebSocket clients
                if system_alerts or trading_alerts:
                    all_alerts = system_alerts + trading_alerts
                    await self._broadcast_alerts(all_alerts)
                
                # Broadcast metrics update
                await self._broadcast_metrics_update(system_metrics, trading_metrics)
                
                # Update system status
                await self._update_system_status(system_metrics, trading_metrics)
                
                # Wait before next collection
                await asyncio.sleep(self.metrics_collector.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(10)

    async def _broadcast_metrics_update(self, system_metrics: SystemMetrics, trading_metrics: TradingMetrics) -> None:
        """Broadcast metrics update to WebSocket clients"""
        message = json.dumps({
            'type': 'metrics_update',
            'system': asdict(system_metrics),
            'trading': asdict(trading_metrics),
            'timestamp': int(time.time() * 1000)
        })
        await self.websocket_manager.broadcast(message)

    async def _broadcast_alerts(self, alerts: List[Alert]) -> None:
        """Broadcast alerts to WebSocket clients"""
        message = json.dumps({
            'type': 'alerts_update',
            'alerts': [asdict(alert) for alert in alerts],
            'timestamp': int(time.time() * 1000)
        })
        await self.websocket_manager.broadcast(message)

    async def _update_system_status(self, system_metrics: SystemMetrics, trading_metrics: TradingMetrics) -> None:
        """Update overall system status"""
        # Simple status calculation based on metrics
        if (system_metrics.cpu_usage > 90 or 
            system_metrics.memory_usage > 95 or 
            system_metrics.error_rate > 0.1):
            self.system_status = SystemStatus.UNHEALTHY
        elif (system_metrics.cpu_usage > 80 or 
              system_metrics.memory_usage > 85 or 
              system_metrics.error_rate > 0.05):
            self.system_status = SystemStatus.DEGRADED
        else:
            self.system_status = SystemStatus.HEALTHY

    async def start(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the FastAPI dashboard"""
        # Start background metrics collection
        asyncio.create_task(self.start_metrics_collection())
        
        # Start WebSocket ping task
        asyncio.create_task(self._websocket_ping_task())
        
        logger.info(f"Starting FastAPI dashboard on {host}:{port}")
        
        # Run FastAPI server
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def _websocket_ping_task(self) -> None:
        """Periodic WebSocket ping task"""
        while True:
            try:
                await self.websocket_manager.ping_clients()
                await asyncio.sleep(30)  # Ping every 30 seconds
            except Exception as e:
                logger.error(f"Error in WebSocket ping task: {e}")
                await asyncio.sleep(30)

    def get_app(self) -> FastAPI:
        """Get FastAPI app instance"""
        return self.app
