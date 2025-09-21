# config.py - Configuration for Deriv AI Trading Bot
import os
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class TradingConfig:
    # Deriv API Configuration
    DERIV_APP_ID: str = "1089"  # Public app ID for testing
    DERIV_WS_URL: str = "wss://ws.binaryws.com/websockets/v3"
    DERIV_TOKEN: str = field(default_factory=lambda: os.getenv("DERIV_TOKEN", ""))
    
    # Trading Parameters
    SYMBOLS: List[str] = field(default_factory=lambda: ["R_100", "R_50", "R_75", "R_25"])
    INITIAL_STAKE: float = 2.0
    MAX_STAKE: float = 100.0
    MIN_STAKE: float = 0.35
    
    # Risk Management
    MAX_DAILY_LOSS: float = 500.0
    MAX_CONSECUTIVE_LOSSES: int = 5
    DAILY_PROFIT_TARGET: float = 200.0
    MAX_POSITION_SIZE_PCT: float = 0.02  # 2% of balance per trade
    
    # AI/ML Parameters
    LOOKBACK_WINDOW: int = 500  # Number of ticks to analyze
    MIN_CONFIDENCE: float = 0.65  # Minimum prediction confidence
    FEATURE_UPDATE_INTERVAL: int = 10  # Update features every N ticks
    
    # Pattern Detection
    STREAK_THRESHOLD: int = 3
    VOLATILITY_THRESHOLD: float = 0.02
    DIGIT_FREQUENCY_WINDOW: int = 100
    N_GRAM_N: int = 3  # For pattern detection
    HISTORY_WINDOW: int = 200  # Pattern history window
    
    # Contract Types
    AVAILABLE_CONTRACTS: List[str] = field(default_factory=lambda: [
        "DIGITEVEN", "DIGITODD", 
        "DIGITOVER", "DIGITUNDER",
        "CALL", "PUT"
    ])
    
    # Database
    DB_PATH: str = "trading_data.db"
    BACKUP_INTERVAL: int = 3600  # Backup every hour

# Global config instance
config = TradingConfig()
