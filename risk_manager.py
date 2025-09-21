# risk_manager.py - Advanced risk management system
import time
import sqlite3
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from config import config

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    current_balance: float
    daily_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    consecutive_losses: int
    consecutive_wins: int
    max_drawdown: float
    current_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

class RiskManager:
    def __init__(self, initial_balance: float = 1000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.daily_start_balance = initial_balance
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance
        self.last_reset_time = time.time()
        self.trades_today = 0
        self.init_risk_tables()
        
    def init_risk_tables(self):
        """Initialize risk tracking tables"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            # Risk events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT,
                    description TEXT,
                    balance_before REAL,
                    balance_after REAL,
                    timestamp INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Daily risk summary
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_risk_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    starting_balance REAL,
                    ending_balance REAL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    max_consecutive_losses INTEGER,
                    max_drawdown REAL,
                    profit_factor REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing risk tables: {e}")
    
    def calculate_position_size(self, confidence: float, symbol: str) -> float:
        """Calculate optimal position size based on Kelly Criterion and risk limits"""
        # Base stake from config
        base_stake = config.INITIAL_STAKE
        
        # Kelly Criterion adjustment (simplified)
        # Assume payout ratio of 0.8 (typical for Deriv)
        payout_ratio = 0.8
        win_prob = confidence
        
        # Kelly fraction: f = (bp - q) / b
        # where b = payout ratio, p = win probability, q = loss probability
        kelly_fraction = (payout_ratio * win_prob - (1 - win_prob)) / payout_ratio
        
        # Conservative Kelly (use 25% of full Kelly)
        conservative_kelly = max(0, kelly_fraction * 0.25)
        
        # Calculate stake based on Kelly and current balance
        kelly_stake = self.current_balance * conservative_kelly
        
        # Apply confidence scaling
        confidence_multiplier = min(2.0, confidence / 0.5)  # Scale between 0.5 and 2.0
        adjusted_stake = base_stake * confidence_multiplier
        
        # Use the smaller of Kelly stake and confidence-adjusted stake
        proposed_stake = min(kelly_stake, adjusted_stake)
        
        # Apply hard limits
        max_position_size = self.current_balance * config.MAX_POSITION_SIZE_PCT
        proposed_stake = min(proposed_stake, max_position_size)
        proposed_stake = max(proposed_stake, config.MIN_STAKE)
        proposed_stake = min(proposed_stake, config.MAX_STAKE)
        
        # Reduce stake after consecutive losses
        if self.consecutive_losses > 0:
            loss_reduction = 0.8 ** self.consecutive_losses  # Exponential reduction
            proposed_stake *= loss_reduction
        
        return round(proposed_stake, 2)
    
    def check_trading_allowed(self) -> Tuple[bool, str]:
        """Check if trading is allowed based on risk rules"""
        current_time = time.time()
        
        # Check daily reset
        if current_time - self.last_reset_time > 24 * 3600:
            self.reset_daily_limits()
        
        # Check daily loss limit
        daily_loss = self.daily_start_balance - self.current_balance
        if daily_loss >= config.MAX_DAILY_LOSS:
            return False, f"Daily loss limit reached: ${daily_loss:.2f}"
        
        # Check consecutive losses
        if self.consecutive_losses >= config.MAX_CONSECUTIVE_LOSSES:
            return False, f"Max consecutive losses reached: {self.consecutive_losses}"
        
        # Check minimum balance
        if self.current_balance < config.MIN_STAKE * 10:
            return False, f"Insufficient balance: ${self.current_balance:.2f}"
        
        # Check if daily profit target reached
        daily_profit = self.current_balance - self.daily_start_balance
        if daily_profit >= config.DAILY_PROFIT_TARGET:
            return False, f"Daily profit target reached: ${daily_profit:.2f}"
        
        # Check maximum trades per day (optional limit)
        max_daily_trades = 100  # Configurable
        if self.trades_today >= max_daily_trades:
            return False, f"Maximum daily trades reached: {self.trades_today}"
        
        return True, "Trading allowed"
    
    def update_balance(self, trade_result: float, contract_type: str, stake: float) -> Dict:
        """Update balance and risk metrics after a trade"""
        old_balance = self.current_balance
        self.current_balance += trade_result
        self.trades_today += 1
        
        # Update peak balance and drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
            
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Update consecutive wins/losses
        if trade_result > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        # Log risk event if significant
        if abs(trade_result) > stake * 2 or self.consecutive_losses >= 3:
            self.log_risk_event(
                "significant_trade" if abs(trade_result) > stake * 2 else "consecutive_losses",
                f"Trade result: ${trade_result:.2f}, Consecutive losses: {self.consecutive_losses}",
                old_balance,
                self.current_balance
            )
        
        return {
            'old_balance': old_balance,
            'new_balance': self.current_balance,
            'trade_result': trade_result,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_wins': self.consecutive_wins,
            'current_drawdown': current_drawdown,
            'max_drawdown': self.max_drawdown
        }
    
    def log_risk_event(self, event_type: str, description: str, 
                      balance_before: float, balance_after: float):
        """Log significant risk events"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO risk_events 
                (event_type, description, balance_before, balance_after, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (event_type, description, balance_before, balance_after, int(time.time())))
            
            conn.commit()
            conn.close()
            
            logger.warning(f"Risk event: {event_type} - {description}")
            
        except Exception as e:
            logger.error(f"Error logging risk event: {e}")
    
    def reset_daily_limits(self):
        """Reset daily trading limits"""
        # Save daily summary
        self.save_daily_summary()
        
        # Reset daily counters
        self.daily_start_balance = self.current_balance
        self.trades_today = 0
        self.last_reset_time = time.time()
        
        # Reset consecutive counters if it's a new day
        # (Optional: you might want to keep these across days)
        # self.consecutive_losses = 0
        # self.consecutive_wins = 0
        
        logger.info(f"Daily limits reset. Starting balance: ${self.current_balance:.2f}")
    
    def save_daily_summary(self):
        """Save daily risk summary to database"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            # Get today's date
            from datetime import datetime
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Calculate daily metrics
            daily_trades = self.get_daily_trade_count()
            winning_trades, losing_trades = self.get_daily_win_loss_count()
            
            cursor.execute("""
                INSERT OR REPLACE INTO daily_risk_summary 
                (date, starting_balance, ending_balance, total_trades, 
                 winning_trades, losing_trades, max_consecutive_losses, 
                 max_drawdown, profit_factor)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                today, self.daily_start_balance, self.current_balance,
                daily_trades, winning_trades, losing_trades,
                self.consecutive_losses, self.max_drawdown,
                self.calculate_profit_factor()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving daily summary: {e}")
    
    def get_daily_trade_count(self) -> int:
        """Get number of trades today"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            today_start = int(time.time()) - (int(time.time()) % (24 * 3600))
            
            cursor.execute("""
                SELECT COUNT(*) FROM trades 
                WHERE entry_time >= ?
            """, (today_start,))
            
            count = cursor.fetchone()[0]
            conn.close()
            return count
            
        except Exception as e:
            logger.error(f"Error getting daily trade count: {e}")
            return 0
    
    def get_daily_win_loss_count(self) -> Tuple[int, int]:
        """Get daily win/loss counts"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            today_start = int(time.time()) - (int(time.time()) % (24 * 3600))
            
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN win = 1 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN win = 0 THEN 1 ELSE 0 END) as losses
                FROM trades 
                WHERE entry_time >= ?
            """, (today_start,))
            
            result = cursor.fetchone()
            conn.close()
            
            return (result[0] or 0, result[1] or 0)
            
        except Exception as e:
            logger.error(f"Error getting win/loss count: {e}")
            return (0, 0)
    
    def calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            today_start = int(time.time()) - (int(time.time()) % (24 * 3600))
            
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN profit_loss > 0 THEN profit_loss ELSE 0 END) as gross_profit,
                    SUM(CASE WHEN profit_loss < 0 THEN ABS(profit_loss) ELSE 0 END) as gross_loss
                FROM trades 
                WHERE entry_time >= ?
            """, (today_start,))
            
            result = cursor.fetchone()
            conn.close()
            
            gross_profit = result[0] or 0
            gross_loss = result[1] or 0
            
            return gross_profit / gross_loss if gross_loss > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating profit factor: {e}")
            return 0
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics"""
        daily_pnl = self.current_balance - self.daily_start_balance
        daily_trades = self.get_daily_trade_count()
        winning_trades, losing_trades = self.get_daily_win_loss_count()
        
        win_rate = winning_trades / max(1, daily_trades)
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        
        return RiskMetrics(
            current_balance=self.current_balance,
            daily_pnl=daily_pnl,
            total_trades=daily_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            consecutive_losses=self.consecutive_losses,
            consecutive_wins=self.consecutive_wins,
            max_drawdown=self.max_drawdown,
            current_drawdown=current_drawdown,
            win_rate=win_rate,
            avg_win=0.0,  # Calculate if needed
            avg_loss=0.0,  # Calculate if needed
            profit_factor=self.calculate_profit_factor()
        )
    
    def emergency_stop(self, reason: str):
        """Emergency stop all trading"""
        self.log_risk_event(
            "emergency_stop",
            f"Emergency stop triggered: {reason}",
            self.current_balance,
            self.current_balance
        )
        
        logger.critical(f"EMERGENCY STOP: {reason}")
        
        # You could add additional emergency procedures here
        # like sending notifications, closing positions, etc.

if __name__ == "__main__":
    # Test risk manager
    rm = RiskManager(1000.0)
    
    # Simulate some trades
    print("Initial metrics:", rm.get_risk_metrics())
    
    # Simulate a winning trade
    rm.update_balance(10.0, "DIGITEVEN", 5.0)
    print("After win:", rm.get_risk_metrics())
    
    # Simulate a losing trade
    rm.update_balance(-5.0, "DIGITODD", 5.0)
    print("After loss:", rm.get_risk_metrics())
