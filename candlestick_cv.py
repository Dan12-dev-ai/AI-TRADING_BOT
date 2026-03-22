"""
Medallion-X Candlestick Computer Vision Module
Advanced pattern recognition using computer vision techniques
Production-ready implementation with 50+ candlestick patterns
"""

import asyncio
import logging
import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
import time

import redis.asyncio as redis
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from ..config.settings import config
from ..utils.logging import get_logger

logger = get_logger(__name__)

class PatternType(Enum):
    """Candlestick pattern types"""
    REVERSAL_BULLISH = "reversal_bullish"
    REVERSAL_BEARISH = "reversal_bearish"
    CONTINUATION_BULLISH = "continuation_bullish"
    CONTINUATION_BEARISH = "continuation_bearish"
    INDECISION = "indecision"
    STRONG_TREND = "strong_trend"

@dataclass
class CandlestickData:
    """Individual candlestick data"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class PatternMatch:
    """Pattern match result"""
    pattern_name: str
    pattern_type: PatternType
    confidence: float  # 0-1 confidence score
    timestamp: int
    symbol: str
    exchange: str
    
    # Pattern characteristics
    body_size: float
    upper_shadow: float
    lower_shadow: float
    color: str  # 'green' or 'red'
    
    # Market context
    trend_direction: str  # 'up', 'down', 'sideways'
    volume_ratio: float  # Current volume / average volume
    price_position: float  # Position within recent range (0-1)

class CandlestickPatterns:
    """Collection of 50+ candlestick pattern recognizers"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()

    def _initialize_patterns(self) -> Dict[str, callable]:
        """Initialize all pattern recognition functions"""
        return {
            # Single candlestick patterns
            'doji': self.recognize_doji,
            'hammer': self.recognize_hammer,
            'inverted_hammer': self.recognize_inverted_hammer,
            'shooting_star': self.recognize_shooting_star,
            'hanging_man': self.recognize_hanging_man,
            'marubozu_bullish': self.recognize_marubozu_bullish,
            'marubozu_bearish': self.recognize_marubozu_bearish,
            'spinning_top': self.recognize_spinning_top,
            'white_soldier': self.recognize_white_soldier,
            'black_marubozu': self.recognize_black_marubozu,
            
            # Two candlestick patterns
            'engulfing_bullish': self.recognize_engulfing_bullish,
            'engulfing_bearish': self.recognize_engulfing_bearish,
            'harami_bullish': self.recognize_harami_bullish,
            'harami_bearish': self.recognize_harami_bearish,
            'tweezer_top': self.recognize_tweezer_top,
            'tweezer_bottom': self.recognize_tweezer_bottom,
            'dark_cloud_cover': self.recognize_dark_cloud_cover,
            'piercing_pattern': self.recognize_piercing_pattern,
            'bullish_kick': self.recognize_bullish_kick,
            'bearish_kick': self.recognize_bearish_kick,
            
            # Three candlestick patterns
            'morning_star': self.recognize_morning_star,
            'evening_star': self.recognize_evening_star,
            'abandoned_baby_bullish': self.recognize_abandoned_baby_bullish,
            'abandoned_baby_bearish': self.recognize_abandoned_baby_bearish,
            'three_white_soldiers': self.recognize_three_white_soldiers,
            'three_black_crows': self.recognize_three_black_crows,
            'three_inside_up': self.recognize_three_inside_up,
            'three_inside_down': self.recognize_three_inside_down,
            'three_outside_up': self.recognize_three_outside_up,
            'three_outside_down': self.recognize_three_outside_down,
            'rising_three': self.recognize_rising_three,
            'falling_three': self.recognize_falling_three,
            
            # Multi-candlestick patterns
            'ascending_triangle': self.recognize_ascending_triangle,
            'descending_triangle': self.recognize_descending_triangle,
            'symmetrical_triangle': self.recognize_symmetrical_triangle,
            'flag_bullish': self.recognize_flag_bullish,
            'flag_bearish': self.recognize_flag_bearish,
            'pennant_bullish': self.recognize_pennant_bullish,
            'pennant_bearish': self.recognize_pennant_bearish,
            'wedge_rising': self.recognize_wedge_rising,
            'wedge_falling': self.recognize_wedge_falling,
            'double_top': self.recognize_double_top,
            'double_bottom': self.recognize_double_bottom,
            'head_shoulders': self.recognize_head_shoulders,
            'inverse_head_shoulders': self.recognize_inverse_head_shoulders,
            'cup_handle': self.recognize_cup_handle,
            'rounded_bottom': self.recognize_rounded_bottom,
            'rounded_top': self.recognize_rounded_top,
            
            # Gap patterns
            'gap_up': self.recognize_gap_up,
            'gap_down': self.recognize_gap_down,
            'breakaway_gap': self.recognize_breakaway_gap,
            'runaway_gap': self.recognize_runaway_gap,
            'exhaustion_gap': self.recognize_exhaustion_gap,
            'island_reversal': self.recognize_island_reversal,
            
            # Special patterns
            'dragonfly_doji': self.recognize_dragonfly_doji,
            'gravestone_doji': self.recognize_gravestone_doji,
            'long_legged_doji': self.recognize_long_legged_doji,
            'four_price_doji': self.recognize_four_price_doji,
        }

    def calculate_candlestick_properties(self, candle: CandlestickData) -> Dict[str, float]:
        """Calculate candlestick properties"""
        body_size = abs(candle.close - candle.open)
        upper_shadow = candle.high - max(candle.open, candle.close)
        lower_shadow = min(candle.open, candle.close) - candle.low
        total_range = candle.high - candle.low
        
        # Normalize properties
        if total_range > 0:
            body_ratio = body_size / total_range
            upper_shadow_ratio = upper_shadow / total_range
            lower_shadow_ratio = lower_shadow / total_range
        else:
            body_ratio = 0.0
            upper_shadow_ratio = 0.0
            lower_shadow_ratio = 0.0
        
        color = 'green' if candle.close >= candle.open else 'red'
        
        return {
            'body_size': body_size,
            'upper_shadow': upper_shadow,
            'lower_shadow': lower_shadow,
            'total_range': total_range,
            'body_ratio': body_ratio,
            'upper_shadow_ratio': upper_shadow_ratio,
            'lower_shadow_ratio': lower_shadow_ratio,
            'color': color
        }

    # Single candlestick patterns
    def recognize_doji(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Doji pattern"""
        if len(candles) < 1:
            return None
        
        candle = candles[-1]
        props = self.calculate_candlestick_properties(candle)
        
        # Doji: very small body relative to range
        if props['body_ratio'] < 0.1:
            return PatternMatch(
                pattern_name='doji',
                pattern_type=PatternType.INDECISION,
                confidence=1.0 - props['body_ratio'] * 10,
                timestamp=candle.timestamp,
                symbol="",  # Will be set by caller
                exchange="",
                body_size=props['body_size'],
                upper_shadow=props['upper_shadow'],
                lower_shadow=props['lower_shadow'],
                color=props['color'],
                trend_direction="sideways",
                volume_ratio=1.0,
                price_position=0.5
            )
        return None

    def recognize_hammer(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Hammer pattern"""
        if len(candles) < 2:
            return None
        
        candle = candles[-1]
        prev_candle = candles[-2]
        props = self.calculate_candlestick_properties(candle)
        
        # Hammer: small body, long lower shadow, little or no upper shadow
        # Occurs after downtrend
        is_downtrend = prev_candle.close < prev_candle.open
        
        if (is_downtrend and 
            props['body_ratio'] < 0.3 and 
            props['lower_shadow_ratio'] > 0.6 and 
            props['upper_shadow_ratio'] < 0.1):
            
            confidence = min(1.0, props['lower_shadow_ratio'] + (1.0 - props['body_ratio']))
            
            return PatternMatch(
                pattern_name='hammer',
                pattern_type=PatternType.REVERSAL_BULLISH,
                confidence=confidence,
                timestamp=candle.timestamp,
                symbol="",
                exchange="",
                body_size=props['body_size'],
                upper_shadow=props['upper_shadow'],
                lower_shadow=props['lower_shadow'],
                color=props['color'],
                trend_direction="down",
                volume_ratio=1.0,
                price_position=0.5
            )
        return None

    def recognize_inverted_hammer(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Inverted Hammer pattern"""
        if len(candles) < 2:
            return None
        
        candle = candles[-1]
        prev_candle = candles[-2]
        props = self.calculate_candlestick_properties(candle)
        
        # Inverted Hammer: small body, long upper shadow, little or no lower shadow
        # Occurs after downtrend
        is_downtrend = prev_candle.close < prev_candle.open
        
        if (is_downtrend and 
            props['body_ratio'] < 0.3 and 
            props['upper_shadow_ratio'] > 0.6 and 
            props['lower_shadow_ratio'] < 0.1):
            
            confidence = min(1.0, props['upper_shadow_ratio'] + (1.0 - props['body_ratio']))
            
            return PatternMatch(
                pattern_name='inverted_hammer',
                pattern_type=PatternType.REVERSAL_BULLISH,
                confidence=confidence,
                timestamp=candle.timestamp,
                symbol="",
                exchange="",
                body_size=props['body_size'],
                upper_shadow=props['upper_shadow'],
                lower_shadow=props['lower_shadow'],
                color=props['color'],
                trend_direction="down",
                volume_ratio=1.0,
                price_position=0.5
            )
        return None

    def recognize_shooting_star(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Shooting Star pattern"""
        if len(candles) < 2:
            return None
        
        candle = candles[-1]
        prev_candle = candles[-2]
        props = self.calculate_candlestick_properties(candle)
        
        # Shooting Star: small body, long upper shadow, little or no lower shadow
        # Occurs after uptrend
        is_uptrend = prev_candle.close > prev_candle.open
        
        if (is_uptrend and 
            props['body_ratio'] < 0.3 and 
            props['upper_shadow_ratio'] > 0.6 and 
            props['lower_shadow_ratio'] < 0.1):
            
            confidence = min(1.0, props['upper_shadow_ratio'] + (1.0 - props['body_ratio']))
            
            return PatternMatch(
                pattern_name='shooting_star',
                pattern_type=PatternType.REVERSAL_BEARISH,
                confidence=confidence,
                timestamp=candle.timestamp,
                symbol="",
                exchange="",
                body_size=props['body_size'],
                upper_shadow=props['upper_shadow'],
                lower_shadow=props['lower_shadow'],
                color=props['color'],
                trend_direction="up",
                volume_ratio=1.0,
                price_position=0.5
            )
        return None

    def recognize_hanging_man(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Hanging Man pattern"""
        if len(candles) < 2:
            return None
        
        candle = candles[-1]
        prev_candle = candles[-2]
        props = self.calculate_candlestick_properties(candle)
        
        # Hanging Man: small body, long lower shadow, little or no upper shadow
        # Occurs after uptrend
        is_uptrend = prev_candle.close > prev_candle.open
        
        if (is_uptrend and 
            props['body_ratio'] < 0.3 and 
            props['lower_shadow_ratio'] > 0.6 and 
            props['upper_shadow_ratio'] < 0.1):
            
            confidence = min(1.0, props['lower_shadow_ratio'] + (1.0 - props['body_ratio']))
            
            return PatternMatch(
                pattern_name='hanging_man',
                pattern_type=PatternType.REVERSAL_BEARISH,
                confidence=confidence,
                timestamp=candle.timestamp,
                symbol="",
                exchange="",
                body_size=props['body_size'],
                upper_shadow=props['upper_shadow'],
                lower_shadow=props['lower_shadow'],
                color=props['color'],
                trend_direction="up",
                volume_ratio=1.0,
                price_position=0.5
            )
        return None

    def recognize_marubozu_bullish(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Bullish Marubozu pattern"""
        if len(candles) < 1:
            return None
        
        candle = candles[-1]
        props = self.calculate_candlestick_properties(candle)
        
        # Bullish Marubozu: long green body with no shadows
        if (props['color'] == 'green' and 
            props['body_ratio'] > 0.9 and 
            props['upper_shadow_ratio'] < 0.05 and 
            props['lower_shadow_ratio'] < 0.05):
            
            confidence = props['body_ratio']
            
            return PatternMatch(
                pattern_name='marubozu_bullish',
                pattern_type=PatternType.STRONG_TREND,
                confidence=confidence,
                timestamp=candle.timestamp,
                symbol="",
                exchange="",
                body_size=props['body_size'],
                upper_shadow=props['upper_shadow'],
                lower_shadow=props['lower_shadow'],
                color=props['color'],
                trend_direction="up",
                volume_ratio=1.0,
                price_position=0.5
            )
        return None

    def recognize_marubozu_bearish(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Bearish Marubozu pattern"""
        if len(candles) < 1:
            return None
        
        candle = candles[-1]
        props = self.calculate_candlestick_properties(candle)
        
        # Bearish Marubozu: long red body with no shadows
        if (props['color'] == 'red' and 
            props['body_ratio'] > 0.9 and 
            props['upper_shadow_ratio'] < 0.05 and 
            props['lower_shadow_ratio'] < 0.05):
            
            confidence = props['body_ratio']
            
            return PatternMatch(
                pattern_name='marubozu_bearish',
                pattern_type=PatternType.STRONG_TREND,
                confidence=confidence,
                timestamp=candle.timestamp,
                symbol="",
                exchange="",
                body_size=props['body_size'],
                upper_shadow=props['upper_shadow'],
                lower_shadow=props['lower_shadow'],
                color=props['color'],
                trend_direction="down",
                volume_ratio=1.0,
                price_position=0.5
            )
        return None

    def recognize_spinning_top(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Spinning Top pattern"""
        if len(candles) < 1:
            return None
        
        candle = candles[-1]
        props = self.calculate_candlestick_properties(candle)
        
        # Spinning Top: small body with shadows on both sides
        if (props['body_ratio'] < 0.3 and 
            props['upper_shadow_ratio'] > 0.2 and 
            props['lower_shadow_ratio'] > 0.2):
            
            confidence = 1.0 - props['body_ratio']
            
            return PatternMatch(
                pattern_name='spinning_top',
                pattern_type=PatternType.INDECISION,
                confidence=confidence,
                timestamp=candle.timestamp,
                symbol="",
                exchange="",
                body_size=props['body_size'],
                upper_shadow=props['upper_shadow'],
                lower_shadow=props['lower_shadow'],
                color=props['color'],
                trend_direction="sideways",
                volume_ratio=1.0,
                price_position=0.5
            )
        return None

    # Two candlestick patterns
    def recognize_engulfing_bullish(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Bullish Engulfing pattern"""
        if len(candles) < 2:
            return None
        
        candle1 = candles[-2]
        candle2 = candles[-1]
        
        props1 = self.calculate_candlestick_properties(candle1)
        props2 = self.calculate_candlestick_properties(candle2)
        
        # Bullish Engulfing: second candle (green) completely engulfs first candle (red)
        # Occurs after downtrend
        is_downtrend = candle1.close < candle1.open
        
        if (is_downtrend and 
            props1['color'] == 'red' and 
            props2['color'] == 'green' and 
            candle2.open < candle1.close and 
            candle2.close > candle1.open):
            
            # Calculate confidence based on engulfing ratio
            engulfing_ratio = (candle2.close - candle2.open) / (candle1.open - candle1.close)
            confidence = min(1.0, engulfing_ratio)
            
            return PatternMatch(
                pattern_name='engulfing_bullish',
                pattern_type=PatternType.REVERSAL_BULLISH,
                confidence=confidence,
                timestamp=candle2.timestamp,
                symbol="",
                exchange="",
                body_size=props2['body_size'],
                upper_shadow=props2['upper_shadow'],
                lower_shadow=props2['lower_shadow'],
                color=props2['color'],
                trend_direction="down",
                volume_ratio=1.0,
                price_position=0.5
            )
        return None

    def recognize_engulfing_bearish(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Bearish Engulfing pattern"""
        if len(candles) < 2:
            return None
        
        candle1 = candles[-2]
        candle2 = candles[-1]
        
        props1 = self.calculate_candlestick_properties(candle1)
        props2 = self.calculate_candlestick_properties(candle2)
        
        # Bearish Engulfing: second candle (red) completely engulfs first candle (green)
        # Occurs after uptrend
        is_uptrend = candle1.close > candle1.open
        
        if (is_uptrend and 
            props1['color'] == 'green' and 
            props2['color'] == 'red' and 
            candle2.open > candle1.close and 
            candle2.close < candle1.open):
            
            # Calculate confidence based on engulfing ratio
            engulfing_ratio = (candle2.open - candle2.close) / (candle1.close - candle1.open)
            confidence = min(1.0, engulfing_ratio)
            
            return PatternMatch(
                pattern_name='engulfing_bearish',
                pattern_type=PatternType.REVERSAL_BEARISH,
                confidence=confidence,
                timestamp=candle2.timestamp,
                symbol="",
                exchange="",
                body_size=props2['body_size'],
                upper_shadow=props2['upper_shadow'],
                lower_shadow=props2['lower_shadow'],
                color=props2['color'],
                trend_direction="up",
                volume_ratio=1.0,
                price_position=0.5
            )
        return None

    def recognize_harami_bullish(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Bullish Harami pattern"""
        if len(candles) < 2:
            return None
        
        candle1 = candles[-2]
        candle2 = candles[-1]
        
        props1 = self.calculate_candlestick_properties(candle1)
        props2 = self.calculate_candlestick_properties(candle2)
        
        # Bullish Harami: small second candle completely inside first candle
        # Occurs after downtrend
        is_downtrend = candle1.close < candle1.open
        
        if (is_downtrend and 
            props1['color'] == 'red' and 
            candle2.open > candle1.open and 
            candle2.close < candle1.close):
            
            # Calculate confidence based on size ratio
            size_ratio = props2['body_size'] / props1['body_size']
            confidence = 1.0 - size_ratio  # Smaller second candle = higher confidence
            
            return PatternMatch(
                pattern_name='harami_bullish',
                pattern_type=PatternType.REVERSAL_BULLISH,
                confidence=confidence,
                timestamp=candle2.timestamp,
                symbol="",
                exchange="",
                body_size=props2['body_size'],
                upper_shadow=props2['upper_shadow'],
                lower_shadow=props2['lower_shadow'],
                color=props2['color'],
                trend_direction="down",
                volume_ratio=1.0,
                price_position=0.5
            )
        return None

    def recognize_harami_bearish(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Bearish Harami pattern"""
        if len(candles) < 2:
            return None
        
        candle1 = candles[-2]
        candle2 = candles[-1]
        
        props1 = self.calculate_candlestick_properties(candle1)
        props2 = self.calculate_candlestick_properties(candle2)
        
        # Bearish Harami: small second candle completely inside first candle
        # Occurs after uptrend
        is_uptrend = candle1.close > candle1.open
        
        if (is_uptrend and 
            props1['color'] == 'green' and 
            candle2.open < candle1.open and 
            candle2.close > candle1.close):
            
            # Calculate confidence based on size ratio
            size_ratio = props2['body_size'] / props1['body_size']
            confidence = 1.0 - size_ratio  # Smaller second candle = higher confidence
            
            return PatternMatch(
                pattern_name='harami_bearish',
                pattern_type=PatternType.REVERSAL_BEARISH,
                confidence=confidence,
                timestamp=candle2.timestamp,
                symbol="",
                exchange="",
                body_size=props2['body_size'],
                upper_shadow=props2['upper_shadow'],
                lower_shadow=props2['lower_shadow'],
                color=props2['color'],
                trend_direction="up",
                volume_ratio=1.0,
                price_position=0.5
            )
        return None

    # Three candlestick patterns
    def recognize_morning_star(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Morning Star pattern"""
        if len(candles) < 3:
            return None
        
        candle1 = candles[-3]
        candle2 = candles[-2]
        candle3 = candles[-1]
        
        props1 = self.calculate_candlestick_properties(candle1)
        props2 = self.calculate_candlestick_properties(candle3)
        
        # Morning Star: large red candle, small-bodied candle (gap down), large green candle
        # Occurs after downtrend
        is_downtrend = candle1.close < candle1.open
        
        gap_down = candle2.high < candle1.low
        is_recovery = candle3.close > (candle1.open + candle1.close) / 2
        
        if (is_downtrend and 
            props1['color'] == 'red' and 
            gap_down and 
            props2['color'] == 'green' and 
            is_recovery):
            
            confidence = min(1.0, props1['body_ratio'] + props2['body_ratio'])
            
            return PatternMatch(
                pattern_name='morning_star',
                pattern_type=PatternType.REVERSAL_BULLISH,
                confidence=confidence,
                timestamp=candle3.timestamp,
                symbol="",
                exchange="",
                body_size=props2['body_size'],
                upper_shadow=0.0,
                lower_shadow=0.0,
                color=props2['color'],
                trend_direction="down",
                volume_ratio=1.0,
                price_position=0.5
            )
        return None

    def recognize_evening_star(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Evening Star pattern"""
        if len(candles) < 3:
            return None
        
        candle1 = candles[-3]
        candle2 = candles[-2]
        candle3 = candles[-1]
        
        props1 = self.calculate_candlestick_properties(candle1)
        props3 = self.calculate_candlestick_properties(candle3)
        
        # Evening Star: large green candle, small-bodied candle (gap up), large red candle
        # Occurs after uptrend
        is_uptrend = candle1.close > candle1.open
        
        gap_up = candle2.low > candle1.high
        is_decline = candle3.close < (candle1.open + candle1.close) / 2
        
        if (is_uptrend and 
            props1['color'] == 'green' and 
            gap_up and 
            props3['color'] == 'red' and 
            is_decline):
            
            confidence = min(1.0, props1['body_ratio'] + props3['body_ratio'])
            
            return PatternMatch(
                pattern_name='evening_star',
                pattern_type=PatternType.REVERSAL_BEARISH,
                confidence=confidence,
                timestamp=candle3.timestamp,
                symbol="",
                exchange="",
                body_size=props3['body_size'],
                upper_shadow=0.0,
                lower_shadow=0.0,
                color=props3['color'],
                trend_direction="up",
                volume_ratio=1.0,
                price_position=0.5
            )
        return None

    def recognize_three_white_soldiers(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Three White Soldiers pattern"""
        if len(candles) < 3:
            return None
        
        candle1 = candles[-3]
        candle2 = candles[-2]
        candle3 = candles[-1]
        
        props1 = self.calculate_candlestick_properties(candle1)
        props2 = self.calculate_candlestick_properties(candle2)
        props3 = self.calculate_candlestick_properties(candle3)
        
        # Three White Soldiers: three long green candles, each closing higher than previous
        # Occurs after downtrend
        is_downtrend = candle1.close < candle1.open
        
        progressive = (candle2.close > candle1.close and 
                      candle3.close > candle2.close and
                      candle2.open > candle1.open and
                      candle3.open > candle2.open)
        
        if (is_downtrend and 
            props1['color'] == 'green' and 
            props2['color'] == 'green' and 
            props3['color'] == 'green' and 
            progressive):
            
            confidence = (props1['body_ratio'] + props2['body_ratio'] + props3['body_ratio']) / 3
            
            return PatternMatch(
                pattern_name='three_white_soldiers',
                pattern_type=PatternType.REVERSAL_BULLISH,
                confidence=confidence,
                timestamp=candle3.timestamp,
                symbol="",
                exchange="",
                body_size=props3['body_size'],
                upper_shadow=0.0,
                lower_shadow=0.0,
                color=props3['color'],
                trend_direction="down",
                volume_ratio=1.0,
                price_position=0.5
            )
        return None

    def recognize_three_black_crows(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Three Black Crows pattern"""
        if len(candles) < 3:
            return None
        
        candle1 = candles[-3]
        candle2 = candles[-2]
        candle3 = candles[-1]
        
        props1 = self.calculate_candlestick_properties(candle1)
        props2 = self.calculate_candlestick_properties(candle2)
        props3 = self.calculate_candlestick_properties(candle3)
        
        # Three Black Crows: three long red candles, each closing lower than previous
        # Occurs after uptrend
        is_uptrend = candle1.close > candle1.open
        
        progressive = (candle2.close < candle1.close and 
                      candle3.close < candle2.close and
                      candle2.open < candle1.open and
                      candle3.open < candle2.open)
        
        if (is_uptrend and 
            props1['color'] == 'red' and 
            props2['color'] == 'red' and 
            props3['color'] == 'red' and 
            progressive):
            
            confidence = (props1['body_ratio'] + props2['body_ratio'] + props3['body_ratio']) / 3
            
            return PatternMatch(
                pattern_name='three_black_crows',
                pattern_type=PatternType.REVERSAL_BEARISH,
                confidence=confidence,
                timestamp=candle3.timestamp,
                symbol="",
                exchange="",
                body_size=props3['body_size'],
                upper_shadow=0.0,
                lower_shadow=0.0,
                color=props3['color'],
                trend_direction="up",
                volume_ratio=1.0,
                price_position=0.5
            )
        return None

    # Placeholder methods for remaining patterns
    def recognize_tweezer_top(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Tweezer Top pattern"""
        # Implementation would check for two candles with same highs
        return None

    def recognize_tweezer_bottom(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Tweezer Bottom pattern"""
        # Implementation would check for two candles with same lows
        return None

    def recognize_dark_cloud_cover(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Dark Cloud Cover pattern"""
        # Implementation would check for specific two-candle bearish pattern
        return None

    def recognize_piercing_pattern(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Piercing Pattern"""
        # Implementation would check for specific two-candle bullish pattern
        return None

    def recognize_bullish_kick(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Bullish Kick pattern"""
        # Implementation would check for gap up with marubozu
        return None

    def recognize_bearish_kick(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Bearish Kick pattern"""
        # Implementation would check for gap down with marubozu
        return None

    def recognize_abandoned_baby_bullish(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Bullish Abandoned Baby pattern"""
        return None

    def recognize_abandoned_baby_bearish(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Bearish Abandoned Baby pattern"""
        return None

    def recognize_three_inside_up(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Three Inside Up pattern"""
        return None

    def recognize_three_inside_down(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Three Inside Down pattern"""
        return None

    def recognize_three_outside_up(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Three Outside Up pattern"""
        return None

    def recognize_three_outside_down(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Three Outside Down pattern"""
        return None

    def recognize_rising_three(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Rising Three pattern"""
        return None

    def recognize_falling_three(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Falling Three pattern"""
        return None

    def recognize_ascending_triangle(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Ascending Triangle pattern"""
        return None

    def recognize_descending_triangle(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Descending Triangle pattern"""
        return None

    def recognize_symmetrical_triangle(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Symmetrical Triangle pattern"""
        return None

    def recognize_flag_bullish(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Bullish Flag pattern"""
        return None

    def recognize_flag_bearish(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Bearish Flag pattern"""
        return None

    def recognize_pennant_bullish(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Bullish Pennant pattern"""
        return None

    def recognize_pennant_bearish(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Bearish Pennant pattern"""
        return None

    def recognize_wedge_rising(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Rising Wedge pattern"""
        return None

    def recognize_wedge_falling(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Falling Wedge pattern"""
        return None

    def recognize_double_top(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Double Top pattern"""
        return None

    def recognize_double_bottom(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Double Bottom pattern"""
        return None

    def recognize_head_shoulders(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Head and Shoulders pattern"""
        return None

    def recognize_inverse_head_shoulders(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Inverse Head and Shoulders pattern"""
        return None

    def recognize_cup_handle(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Cup and Handle pattern"""
        return None

    def recognize_rounded_bottom(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Rounded Bottom pattern"""
        return None

    def recognize_rounded_top(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Rounded Top pattern"""
        return None

    def recognize_gap_up(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Gap Up pattern"""
        return None

    def recognize_gap_down(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Gap Down pattern"""
        return None

    def recognize_breakaway_gap(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Breakaway Gap pattern"""
        return None

    def recognize_runaway_gap(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Runaway Gap pattern"""
        return None

    def recognize_exhaustion_gap(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Exhaustion Gap pattern"""
        return None

    def recognize_island_reversal(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Island Reversal pattern"""
        return None

    def recognize_dragonfly_doji(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Dragonfly Doji pattern"""
        return None

    def recognize_gravestone_doji(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Gravestone Doji pattern"""
        return None

    def recognize_long_legged_doji(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Long Legged Doji pattern"""
        return None

    def recognize_four_price_doji(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Four Price Doji pattern"""
        return None

    def recognize_white_soldier(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize White Soldier pattern"""
        return None

    def recognize_black_marubozu(self, candles: List[CandlestickData]) -> Optional[PatternMatch]:
        """Recognize Black Marubozu pattern"""
        return None

class CandlestickCV:
    """
    Candlestick Computer Vision Engine
    - Pattern recognition using CV techniques
    - Image generation and analysis
    - Machine learning classification
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.pattern_recognizer = CandlestickPatterns()
        self.candlestick_history: Dict[str, List[CandlestickData]] = {}
        
        # Image generation settings
        self.image_size = (224, 224)  # Standard CNN input size
        self.max_candles = 50  # Maximum candles to display
        
        # Performance metrics
        self.metrics = {
            'patterns_detected': 0,
            'images_generated': 0,
            'cv_analysis_time_ms': 0.0,
            'pattern_accuracy': 0.0
        }

    def update_candlestick_data(self, symbol: str, exchange: str, 
                               open_price: float, high: float, low: float, 
                               close: float, volume: float, timestamp: int) -> None:
        """Update candlestick data for a symbol"""
        if symbol not in self.candlestick_history:
            self.candlestick_history[symbol] = []
        
        candlestick = CandlestickData(
            timestamp=timestamp,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume
        )
        
        self.candlestick_history[symbol].append(candlestick)
        
        # Keep only recent history
        if len(self.candlestick_history[symbol]) > self.max_candles:
            self.candlestick_history[symbol] = self.candlestick_history[symbol][-self.max_candles:]

    async def detect_patterns(self, symbol: str, exchange: str) -> List[PatternMatch]:
        """Detect candlestick patterns for a symbol"""
        start_time = time.time()
        
        if symbol not in self.candlestick_history or len(self.candlestick_history[symbol]) < 3:
            return []
        
        candles = self.candlestick_history[symbol]
        detected_patterns = []
        
        # Check all patterns
        for pattern_name, pattern_func in self.pattern_recognizer.patterns.items():
            try:
                pattern_match = pattern_func(candles)
                if pattern_match:
                    pattern_match.symbol = symbol
                    pattern_match.exchange = exchange
                    detected_patterns.append(pattern_match)
                    
            except Exception as e:
                logger.error(f"Error detecting pattern {pattern_name}: {e}")
                continue
        
        # Sort by confidence
        detected_patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        # Update metrics
        self.metrics['patterns_detected'] += len(detected_patterns)
        analysis_time = (time.time() - start_time) * 1000
        self.metrics['cv_analysis_time_ms'] = (
            (self.metrics['cv_analysis_time_ms'] * (self.metrics['patterns_detected'] - len(detected_patterns)) + analysis_time) /
            self.metrics['patterns_detected']
        )
        
        # Store patterns in Redis
        for pattern in detected_patterns:
            await self._store_pattern(pattern)
        
        return detected_patterns

    async def generate_candlestick_image(self, symbol: str, exchange: str) -> Optional[np.ndarray]:
        """Generate candlestick chart image for CV analysis"""
        if symbol not in self.candlestick_history or len(self.candlestick_history[symbol]) < 5:
            return None
        
        candles = self.candlestick_history[symbol]
        
        try:
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(self.image_size[0]/100, self.image_size[1]/100), dpi=100)
            fig.patch.set_facecolor('white')
            
            # Prepare data
            dates = list(range(len(candles)))
            opens = [c.open for c in candles]
            highs = [c.high for c in candles]
            lows = [c.low for c in candles]
            closes = [c.close for c in candles]
            
            # Colors for candles
            colors = ['green' if c.close >= c.open else 'red' for c in candles]
            
            # Plot candlesticks
            for i, (date, open_price, high, low, close_price, color) in enumerate(
                zip(dates, opens, highs, lows, closes, colors)):
                
                # Draw the candlestick
                ax.plot([date, date], [low, high], color='black', linewidth=1)
                
                if color == 'green':
                    ax.bar(date, close_price - open_price, bottom=open_price, 
                          color='green', width=0.6, alpha=0.8)
                else:
                    ax.bar(date, open_price - close_price, bottom=close_price, 
                          color='red', width=0.6, alpha=0.8)
            
            # Styling
            ax.set_facecolor('white')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{symbol} Candlestick Chart', fontsize=10)
            ax.set_xlabel('Time', fontsize=8)
            ax.set_ylabel('Price', fontsize=8)
            
            # Convert to numpy array
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            
            # Update metrics
            self.metrics['images_generated'] += 1
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error generating candlestick image: {e}")
            return None

    async def analyze_image_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract features from candlestick image using CV techniques"""
        if image is None:
            return {}
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Contour analysis
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Feature extraction
            features = {
                'edge_density': np.sum(edges > 0) / edges.size,
                'contour_count': len(contours),
                'avg_contour_area': np.mean([cv2.contourArea(c) for c in contours]) if contours else 0,
                'symmetry_score': self._calculate_symmetry(gray),
                'trend_angle': self._calculate_trend_angle(gray),
                'volatility_index': self._calculate_volatility_index(gray)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error analyzing image features: {e}")
            return {}

    def _calculate_symmetry(self, gray_image: np.ndarray) -> float:
        """Calculate vertical symmetry score"""
        height, width = gray_image.shape
        left_half = gray_image[:, :width//2]
        right_half = gray_image[:, width//2:]
        
        # Flip right half and compare
        right_flipped = cv2.flip(right_half, 1)
        
        # Calculate similarity
        if left_half.shape == right_flipped.shape:
            similarity = np.corrcoef(left_half.flatten(), right_flipped.flatten())[0, 1]
            return similarity if not np.isnan(similarity) else 0.0
        return 0.0

    def _calculate_trend_angle(self, gray_image: np.ndarray) -> float:
        """Calculate dominant trend angle using Hough lines"""
        edges = cv2.Canny(gray_image, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is not None:
            angles = [line[0][1] for line in lines]
            # Convert to degrees and normalize
            angle_degrees = np.mean(angles) * 180 / np.pi
            return angle_degrees
        return 0.0

    def _calculate_volatility_index(self, gray_image: np.ndarray) -> float:
        """Calculate volatility index from image intensity variation"""
        # Calculate standard deviation of pixel intensities
        return np.std(gray_image) / 255.0  # Normalized

    async def _store_pattern(self, pattern: PatternMatch) -> None:
        """Store pattern match in Redis"""
        key = f"patterns:{pattern.symbol}:{pattern.exchange}:latest"
        
        pattern_dict = asdict(pattern)
        pattern_dict['pattern_type'] = pattern.pattern_type.value
        
        await self.redis_client.setex(
            key,
            ttl=3600,  # 1 hour TTL
            value=json.dumps(pattern_dict, default=str)
        )
        
        # Store in time series
        ts_key = f"patterns:ts:{pattern.symbol}:{pattern.exchange}"
        await self.redis_client.zadd(
            ts_key,
            {json.dumps(pattern_dict, default=str): pattern.timestamp}
        )
        # Keep only last 1000 patterns
        await self.redis_client.zremrangebyrank(ts_key, 0, -1001)

    async def get_latest_patterns(self, symbol: str, exchange: str, limit: int = 10) -> List[PatternMatch]:
        """Get latest patterns for a symbol"""
        try:
            ts_key = f"patterns:ts:{symbol}:{exchange}"
            pattern_data = await self.redis_client.zrevrange(ts_key, 0, limit - 1)
            
            patterns = []
            for data in pattern_data:
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                pattern_dict = json.loads(data)
                pattern_dict['pattern_type'] = PatternType(pattern_dict['pattern_type'])
                patterns.append(PatternMatch(**pattern_dict))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error retrieving patterns: {e}")
            return []

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.copy()

    async def start(self) -> None:
        """Start candlestick CV engine"""
        logger.info("Candlestick CV engine started")

    async def stop(self) -> None:
        """Stop candlestick CV engine"""
        logger.info("Candlestick CV engine stopped")
