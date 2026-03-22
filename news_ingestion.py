"""
Medallion-X News Data Ingestion Module
Multi-source financial news aggregation with sentiment analysis
Production-ready, async, real-time news processing
"""

import asyncio
import logging
import aiohttp
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from urllib.parse import urlencode

import redis.asyncio as redis
from bs4 import BeautifulSoup
import newspaper
from textblob import TextBlob
import numpy as np

from ..config.settings import config
from ..utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class NewsArticle:
    """News article data structure"""
    id: str
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    sentiment_score: float  # -1 to 1
    sentiment_label: str   # positive, negative, neutral
    relevance_score: float  # 0 to 1
    keywords: List[str]
    symbols_mentioned: List[str]

class NewsIngestion:
    """
    Multi-source news data ingestion engine
    - Real-time news aggregation from multiple APIs
    - Sentiment analysis and relevance scoring
    - Symbol mention detection
    - Redis caching with TTL
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.session = None
        self.is_running = False
        self.processed_articles = set()
        
        # News source configurations
        self.news_sources = {
            'newsapi': {
                'url': 'https://newsapi.org/v2/everything',
                'key': config.news_apis['newsapi'],
                'rate_limit': 1000  # requests per hour
            },
            'alpha_vantage': {
                'url': 'https://www.alphavantage.co/query',
                'key': config.news_apis['alpha_vantage'],
                'rate_limit': 5  # requests per minute
            },
            'cryptocompare': {
                'url': 'https://min-api.cryptocompare.com/data/v2/news/',
                'key': config.news_apis['cryptocompare'],
                'rate_limit': 100  # requests per minute
            }
        }
        
        # Trading symbols to monitor
        self.trading_symbols = config.symbols
        
        # Symbol keyword mapping
        self.symbol_keywords = self._create_symbol_keywords()
        
        # Metrics
        self.metrics = {
            'articles_processed': 0,
            'sentiment_analyzed': 0,
            'relevant_articles': 0,
            'errors_count': 0
        }

    def _create_symbol_keywords(self) -> Dict[str, List[str]]:
        """Create keyword mappings for symbol detection"""
        return {
            'BTC/USDT': ['bitcoin', 'btc', 'bitcoin price', 'btc price', 'bitcoin trading'],
            'ETH/USDT': ['ethereum', 'eth', 'ethereum price', 'eth price', 'ethereum trading'],
            'SOL/USDT': ['solana', 'sol', 'solana price', 'sol price', 'solana trading']
        }

    async def start(self) -> None:
        """Start news ingestion process"""
        self.is_running = True
        self.session = aiohttp.ClientSession()
        
        logger.info("Starting news data ingestion...")
        
        # Create tasks for each news source
        tasks = []
        for source_name, source_config in self.news_sources.items():
            if source_config['key']:
                task = asyncio.create_task(
                    self._run_news_source(source_name, source_config)
                )
                tasks.append(task)
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in news ingestion: {e}")

    async def _run_news_source(self, source_name: str, source_config: Dict[str, Any]) -> None:
        """Run continuous news fetching from a specific source"""
        while self.is_running:
            try:
                # Fetch news based on source type
                if source_name == 'newsapi':
                    articles = await self._fetch_newsapi_news(source_config)
                elif source_name == 'alpha_vantage':
                    articles = await self._fetch_alpha_vantage_news(source_config)
                elif source_name == 'cryptocompare':
                    articles = await self._fetch_cryptocompare_news(source_config)
                else:
                    articles = []
                
                # Process articles
                for article in articles:
                    await self._process_article(article, source_name)
                
                # Rate limiting
                if source_name == 'alpha_vantage':
                    await asyncio.sleep(12)  # 5 requests per minute
                else:
                    await asyncio.sleep(60)  # 1 request per minute for others
                
            except Exception as e:
                logger.error(f"Error fetching from {source_name}: {e}")
                self.metrics['errors_count'] += 1
                await asyncio.sleep(60)  # Wait before retry

    async def _fetch_newsapi_news(self, source_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch news from NewsAPI"""
        # Create search queries for each symbol
        all_articles = []
        
        for symbol in self.trading_symbols:
            keywords = self.symbol_keywords.get(symbol, [])
            
            for keyword in keywords[:3]:  # Limit to avoid rate limits
                params = {
                    'q': keyword,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'from': (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(),
                    'to': datetime.now(timezone.utc).isoformat(),
                    'pageSize': 20,
                    'apiKey': source_config['key']
                }
                
                try:
                    async with self.session.get(source_config['url'], params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            articles = data.get('articles', [])
                            all_articles.extend(articles)
                        else:
                            logger.warning(f"NewsAPI error: {response.status}")
                            
                except Exception as e:
                    logger.error(f"NewsAPI fetch error: {e}")
        
        return all_articles

    async def _fetch_alpha_vantage_news(self, source_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch news from Alpha Vantage"""
        params = {
            'function': 'NEWS_SENTIMENT',
            'apikey': source_config['key']
        }
        
        try:
            async with self.session.get(source_config['url'], params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('feed', [])
                else:
                    logger.warning(f"Alpha Vantage error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Alpha Vantage fetch error: {e}")
            return []

    async def _fetch_cryptocompare_news(self, source_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch news from CryptoCompare"""
        params = {
            'api_key': source_config['key'],
            'lang': 'EN'
        }
        
        try:
            async with self.session.get(source_config['url'], params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('Data', [])
                else:
                    logger.warning(f"CryptoCompare error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"CryptoCompare fetch error: {e}")
            return []

    async def _process_article(self, article_data: Dict[str, Any], source: str) -> None:
        """Process a single news article"""
        try:
            # Extract article content based on source format
            if source == 'newsapi':
                article = self._parse_newsapi_article(article_data)
            elif source == 'alpha_vantage':
                article = self._parse_alpha_vantage_article(article_data)
            elif source == 'cryptocompare':
                article = self._parse_cryptocompare_article(article_data)
            else:
                return
            
            # Skip if already processed
            if article.id in self.processed_articles:
                return
            
            # Perform sentiment analysis
            sentiment = self._analyze_sentiment(article.title + " " + article.content)
            article.sentiment_score = sentiment['score']
            article.sentiment_label = sentiment['label']
            
            # Calculate relevance score
            relevance = self._calculate_relevance(article)
            article.relevance_score = relevance['score']
            article.symbols_mentioned = relevance['symbols']
            
            # Extract keywords
            article.keywords = self._extract_keywords(article.title + " " + article.content)
            
            # Store in Redis if relevant
            if article.relevance_score > 0.3:  # Relevance threshold
                await self._store_article(article)
                self.processed_articles.add(article.id)
                self.metrics['relevant_articles'] += 1
            
            self.metrics['articles_processed'] += 1
            self.metrics['sentiment_analyzed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing article: {e}")
            self.metrics['errors_count'] += 1

    def _parse_newsapi_article(self, data: Dict[str, Any]) -> NewsArticle:
        """Parse NewsAPI article format"""
        return NewsArticle(
            id=data.get('url', ''),
            title=data.get('title', ''),
            content=data.get('content', data.get('description', '')),
            source=data.get('source', {}).get('name', 'NewsAPI'),
            url=data.get('url', ''),
            published_at=datetime.fromisoformat(
                data.get('publishedAt', '').replace('Z', '+00:00')
            ) if data.get('publishedAt') else datetime.now(timezone.utc),
            sentiment_score=0.0,
            sentiment_label='neutral',
            relevance_score=0.0,
            keywords=[],
            symbols_mentioned=[]
        )

    def _parse_alpha_vantage_article(self, data: Dict[str, Any]) -> NewsArticle:
        """Parse Alpha Vantage article format"""
        return NewsArticle(
            id=data.get('id', ''),
            title=data.get('title', ''),
            content=data.get('summary', ''),
            source='Alpha Vantage',
            url=data.get('url', ''),
            published_at=datetime.fromisoformat(
                data.get('time_published', '').replace('T', ' ').replace('Z', '+00:00')
            ) if data.get('time_published') else datetime.now(timezone.utc),
            sentiment_score=0.0,
            sentiment_label='neutral',
            relevance_score=0.0,
            keywords=[],
            symbols_mentioned=[]
        )

    def _parse_cryptocompare_article(self, data: Dict[str, Any]) -> NewsArticle:
        """Parse CryptoCompare article format"""
        return NewsArticle(
            id=data.get('id', ''),
            title=data.get('title', ''),
            content=data.get('body', ''),
            source=data.get('source_info', {}).get('name', 'CryptoCompare'),
            url=data.get('url', ''),
            published_at=datetime.fromtimestamp(
                data.get('published_on', 0)
            ),
            sentiment_score=0.0,
            sentiment_label='neutral',
            relevance_score=0.0,
            keywords=[],
            symbols_mentioned=[]
        )

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Perform sentiment analysis on text"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # Normalize to -1 to 1 range
            score = max(-1.0, min(1.0, polarity))
            
            # Determine label
            if score > 0.1:
                label = 'positive'
            elif score < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            return {
                'score': score,
                'label': label
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {'score': 0.0, 'label': 'neutral'}

    def _calculate_relevance(self, article: NewsArticle) -> Dict[str, Any]:
        """Calculate relevance score and detect symbol mentions"""
        text = (article.title + " " + article.content).lower()
        symbols_mentioned = []
        relevance_score = 0.0
        
        for symbol, keywords in self.symbol_keywords.items():
            symbol_relevance = 0.0
            for keyword in keywords:
                if keyword.lower() in text:
                    symbol_relevance += 1.0
            
            if symbol_relevance > 0:
                symbols_mentioned.append(symbol)
                relevance_score += symbol_relevance
        
        # Normalize relevance score
        if relevance_score > 0:
            relevance_score = min(1.0, relevance_score / 3.0)  # Max 3 keywords per symbol
        
        return {
            'score': relevance_score,
            'symbols': symbols_mentioned
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        try:
            blob = TextBlob(text)
            # Get noun phrases as keywords
            noun_phrases = blob.noun_phrases
            
            # Filter and clean keywords
            keywords = []
            for phrase in noun_phrases:
                phrase = phrase.strip().lower()
                if len(phrase) > 3 and phrase not in keywords:
                    keywords.append(phrase)
            
            return keywords[:10]  # Return top 10 keywords
            
        except Exception as e:
            logger.error(f"Keyword extraction error: {e}")
            return []

    async def _store_article(self, article: NewsArticle) -> None:
        """Store article in Redis"""
        # Convert datetime to string for JSON serialization
        article_dict = asdict(article)
        article_dict['published_at'] = article.published_at.isoformat()
        
        # Store latest articles
        await self.redis_client.setex(
            f"news:latest:{article.id}",
            ttl=3600,  # 1 hour TTL
            value=json.dumps(article_dict)
        )
        
        # Store in time series for each symbol
        for symbol in article.symbols_mentioned:
            await self.redis_client.zadd(
                f"news:symbol:{symbol}",
                {json.dumps(article_dict): int(article.published_at.timestamp())}
            )
            # Keep only last 100 articles per symbol
            await self.redis_client.zremrangebyrank(f"news:symbol:{symbol}", 0, -101)

    async def get_latest_news(self, symbol: Optional[str] = None, limit: int = 10) -> List[NewsArticle]:
        """Get latest news, optionally filtered by symbol"""
        try:
            if symbol:
                # Get news for specific symbol
                news_data = await self.redis_client.zrevrange(
                    f"news:symbol:{symbol}", 0, limit - 1
                )
            else:
                # Get all latest news
                keys = await self.redis_client.keys("news:latest:*")
                news_data = []
                for key in keys[:limit]:
                    data = await self.redis_client.get(key)
                    if data:
                        news_data.append(data)
            
            articles = []
            for data in news_data:
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                article_dict = json.loads(data)
                article_dict['published_at'] = datetime.fromisoformat(article_dict['published_at'])
                articles.append(NewsArticle(**article_dict))
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching latest news: {e}")
            return []

    async def stop(self) -> None:
        """Stop news ingestion gracefully"""
        self.is_running = False
        
        if self.session:
            await self.session.close()
        
        logger.info("News ingestion stopped")

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.metrics,
            'processed_articles_count': len(self.processed_articles),
            'active_sources': len([s for s in self.news_sources.values() if s['key']])
        }
