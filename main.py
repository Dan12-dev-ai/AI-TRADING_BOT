"""
Medallion-X Main Entry Point
Production-ready trading bot with comprehensive initialization
"""

import asyncio
import os
import sys
import signal
from pathlib import Path
from typing import Optional
import argparse
from contextlib import asynccontextmanager

import redis.asyncio as redis
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from medallion_x.config.settings import config, Environment
from medallion_x.data_pipeline import MarketDataIngestion, NewsIngestion, OnChainIngestion
from medallion_x.utils.logging import get_logger, setup_logger

logger = get_logger(__name__)

class MedallionX:
    """
    Main Medallion-X trading bot application
    Coordinates all system components and manages lifecycle
    """
    
    def __init__(self):
        self.data_pipeline = None
        self.ai_engine = None
        self.risk_engine = None
        self.execution_engine = None
        self.monitoring = None
        self.running = False
        
    async def initialize(self):
        """Initialize all bot components"""
        logger.info("🚀 Initializing DEDAN Trading Bot...")
        
        try:
            # Initialize data pipeline
            from .data_pipeline import MarketDataIngestion, NewsIngestion, OnChainIngestion
            
            self.data_pipeline = {
                'market': MarketDataIngestion(),
                'news': NewsIngestion(),
                'onchain': OnChainIngestion()
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory_human', '0B'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }
        except Exception as e:
            logger.error(f"Error getting Redis metrics: {e}")
            return {}

    async def _check_system_health(self, metrics: dict) -> None:
        """Check system health and alert on issues"""
        try:
            # Check market data connections
            market_metrics = metrics.get('market_data', {})
            active_connections = market_metrics.get('active_connections', 0)
            total_connections = market_metrics.get('total_connections', 0)
            
            if total_connections > 0 and active_connections < total_connections * 0.8:
                logger.warning(f"⚠️ Market data connection issues: {active_connections}/{total_connections} active")
            
            # Check Redis health
            redis_metrics = metrics.get('redis', {})
            if redis_metrics.get('connected_clients', 0) == 0:
                logger.error("❌ Redis connection lost!")
            
            # Check error rates
            for component, component_metrics in metrics.items():
                if isinstance(component_metrics, dict):
                    errors = component_metrics.get('errors_count', 0)
                    if errors > 10:  # Alert if > 10 errors
                        logger.warning(f"⚠️ High error rate in {component}: {errors} errors")
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")

    async def shutdown(self) -> None:
        """Graceful shutdown of all components"""
        logger.info("🛑 Shutting down Medallion-X...")
        
        self.is_running = False
        
        try:
            # Shutdown data pipeline components
            if self.market_data_ingestion:
                await self.market_data_ingestion.stop()
                logger.info("✅ Market data ingestion stopped")
            
            if self.news_ingestion:
                await self.news_ingestion.stop()
                logger.info("✅ News ingestion stopped")
            
            if self.onchain_ingestion:
                await self.onchain_ingestion.stop()
                logger.info("✅ On-chain ingestion stopped")
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
                logger.info("✅ Redis connection closed")
            
            logger.info("✅ Medallion-X shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Medallion-X: Ultimate AI Trading Bot"
    )
    
    parser.add_argument(
        "--env",
        choices=["development", "testnet", "production"],
        default="testnet",
        help="Environment to run in (default: testnet)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in simulation mode without real trades"
    )
    
    return parser.parse_args()


async def main() -> None:
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    setup_logger(
        name="medallion_x",
        log_level=args.log_level,
        log_file="logs/medallion_x.log",
        structured=True
    )
    
    logger.info("🎯 Medallion-X Trading Bot Starting...")
    logger.info(f"📍 Environment: {args.env}")
    logger.info(f"📝 Log Level: {args.log_level}")
    
    # Create and run the bot
    try:
        environment = Environment(args.env)
        bot = MedallionX(env=environment)
        
        # Initialize and start
        await bot.initialize()
        await bot.start()
        
    except KeyboardInterrupt:
        logger.info("👋 Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())
