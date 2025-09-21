# shared_config.py - Configuration for Multi-Bot Trading System
import os
from dataclasses import dataclass
from typing import List, Dict

# Load .env file if it exists
def load_env_file():
    """Load environment variables from .env file"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Load .env file at startup
load_env_file()

@dataclass
class MultiBotConfig:
    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = None
    
    # Signal Exchange
    SIGNAL_CHANNEL: str = "deriv_signals"
    SIGNAL_EXPIRY: int = 1  # seconds - signals expire after 1 second
    MAX_SIGNAL_AGE: float = 0.6  # seconds - reject signals older than 0.6s
    
    # Deriv API
    DERIV_APP_ID: str = "1089"
    DERIV_WS_URL: str = "wss://ws.binaryws.com/websockets/v3"
    
    # Bot Tokens (set via environment variables)
    PROBE_A_TOKEN: str = os.getenv("PROBE_A_TOKEN", "")  # Demo token
    PROBE_B_TOKEN: str = os.getenv("PROBE_B_TOKEN", "")  # Demo token  
    PROBE_C_TOKEN: str = os.getenv("PROBE_C_TOKEN", "")  # Demo token
    COORDINATOR_TOKEN: str = os.getenv("COORDINATOR_TOKEN", "")  # Real token
    
    # Trading Symbols
    SYMBOLS: List[str] = None
    PRIMARY_SYMBOL: str = "R_100"
    
    def __post_init__(self):
        if self.SYMBOLS is None:
            self.SYMBOLS = ["R_100", "R_50", "R_75", "R_25"]
    
    # Probe Bot Configuration
    PROBE_A_STRATEGY: str = "digit_parity"  # EVEN/ODD
    PROBE_B_STRATEGY: str = "digit_over_under"  # OVER/UNDER (opposite of A)
    PROBE_C_STRATEGY: str = "momentum"  # RISE/FALL
    
    # Contract Configuration
    CONTRACT_DURATION: int = 1  # ticks
    CONTRACT_DURATION_UNIT: str = "t"
    DIGIT_BARRIER: str = "5"  # for over/under contracts
    
    # Coordinator Decision Thresholds
    MIN_PROBABILITY: float = 0.30  # minimum confidence to trade (lowered for realistic market conditions)
    MIN_EV_THRESHOLD: float = 0.01  # minimum expected value (lowered to enable more trading)
    SIGNAL_AGREEMENT_THRESHOLD: float = 0.1  # confidence difference for agreement
    
    # Risk Management
    INITIAL_BALANCE: float = 1000.0  # Default starting balance for enhanced system
    MAX_STAKE_PCT: float = 0.02  # 2% of bankroll per trade
    MIN_STAKE: float = 0.35
    MAX_STAKE: float = 100.0
    DAILY_LOSS_LIMIT_PCT: float = 0.05  # 5% of bankroll
    MAX_CONSECUTIVE_LOSSES: int = 7
    COOLDOWN_MINUTES: int = 30  # cooldown after max losses
    
    # Probe Stakes (small amounts for intelligence gathering)
    PROBE_STAKE: float = 1.0
    
    # Database
    DB_PATH: str = "multi_bot_trading.db"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "multi_bot_system.log"
    
    # Performance Tracking
    PERFORMANCE_WINDOW: int = 100  # trades to consider for probe performance
    PROBE_SWITCH_THRESHOLD: float = 0.1  # switch if performance diff > 10%
    
    # Backtesting
    BACKTEST_MIN_TRADES: int = 50000  # minimum trades before going live
    BACKTEST_MIN_WIN_RATE: float = 0.52  # minimum win rate to proceed

# Global config instance
config = MultiBotConfig()

# Bot identifiers
BOT_IDS = {
    "PROBE_A": "probe_a_parity",
    "PROBE_B": "probe_b_opposite", 
    "PROBE_C": "probe_c_momentum",
    "COORDINATOR": "coordinator_real"
}

# Signal types
SIGNAL_TYPES = {
    "TRADE_SIGNAL": "trade_signal",
    "TRADE_RESULT": "trade_result", 
    "PERFORMANCE_UPDATE": "performance_update",
    "SYSTEM_STATUS": "system_status"
}
