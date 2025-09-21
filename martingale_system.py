# martingale_system.py - Capped Confidence-Weighted Martingale Recovery System
import numpy as np
import pandas as pd
import sqlite3
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import json
from config import config

logger = logging.getLogger(__name__)

@dataclass
class MartingaleSequence:
    """Represents an active martingale recovery sequence"""
    sequence_id: str
    start_time: float
    symbol: str
    contract_type: str
    base_stake: float
    current_level: int
    max_levels: int
    total_spent: float
    total_risk_budget: float
    payout_ratio: float
    confidence_threshold: float
    last_trade_time: float
    trades: List[Dict]
    is_active: bool

@dataclass
class MartingaleConfig:
    """Configuration for martingale system"""
    max_levels: int = 4  # Maximum recovery levels
    confidence_threshold: float = 0.70  # Minimum confidence to start/continue
    min_ev_threshold: float = 0.02  # Minimum expected value
    risk_budget_pct: float = 0.03  # 3% of bankroll per sequence
    payout_adjustment: bool = True  # Adjust multiplier based on actual payout
    timeout_seconds: int = 300  # 5 minutes max per sequence
    cooldown_seconds: int = 600  # 10 minutes between sequences
    max_daily_sequences: int = 5  # Maximum sequences per day

class MartingaleRecoverySystem:
    """Advanced martingale recovery system with safety controls"""
    
    def __init__(self, config: MartingaleConfig = None):
        self.config = config or MartingaleConfig()
        self.active_sequences: Dict[str, MartingaleSequence] = {}
        self.completed_sequences = deque(maxlen=100)
        self.last_sequence_time = 0
        self.daily_sequences = 0
        self.daily_reset_time = 0
        self.current_balance = 1000.0  # Will be updated from actual balance
        self.init_database()
        
    def init_database(self):
        """Initialize martingale tracking database"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            # Martingale sequences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS martingale_sequences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sequence_id TEXT UNIQUE,
                    symbol TEXT,
                    contract_type TEXT,
                    start_time INTEGER,
                    end_time INTEGER,
                    base_stake REAL,
                    max_levels INTEGER,
                    levels_used INTEGER,
                    total_spent REAL,
                    total_recovered REAL,
                    final_profit REAL,
                    success INTEGER,
                    reason_stopped TEXT,
                    risk_budget REAL,
                    confidence_threshold REAL,
                    trades_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Martingale trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS martingale_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sequence_id TEXT,
                    level INTEGER,
                    timestamp INTEGER,
                    stake REAL,
                    confidence REAL,
                    probability REAL,
                    expected_value REAL,
                    actual_payout REAL,
                    profit_loss REAL,
                    win INTEGER,
                    cumulative_loss REAL,
                    recovery_target REAL,
                    contract_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (sequence_id) REFERENCES martingale_sequences(sequence_id)
                )
            """)
            
            # Martingale performance metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS martingale_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    sequences_attempted INTEGER,
                    sequences_successful INTEGER,
                    total_risk_budget REAL,
                    total_recovered REAL,
                    net_profit REAL,
                    success_rate REAL,
                    avg_levels_used REAL,
                    max_drawdown REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing martingale database: {e}")
    
    def should_start_sequence(self, symbol: str, contract_type: str, 
                            probability: float, confidence: float,
                            expected_value: float) -> Tuple[bool, str]:
        """Determine if we should start a new martingale sequence"""
        current_time = time.time()
        
        # Check daily reset
        if current_time - self.daily_reset_time > 24 * 3600:
            self.daily_sequences = 0
            self.daily_reset_time = current_time
        
        # Check cooldown period
        if current_time - self.last_sequence_time < self.config.cooldown_seconds:
            return False, "cooldown_active"
        
        # Check daily limit
        if self.daily_sequences >= self.config.max_daily_sequences:
            return False, "daily_limit_reached"
        
        # Check confidence threshold
        if confidence < self.config.confidence_threshold:
            return False, f"confidence_too_low_{confidence:.3f}"
        
        # Check expected value threshold
        if expected_value < self.config.min_ev_threshold:
            return False, f"ev_too_low_{expected_value:.4f}"
        
        # Check if we have an active sequence for this symbol/type
        sequence_key = f"{symbol}_{contract_type}"
        for seq_id, sequence in self.active_sequences.items():
            if f"{sequence.symbol}_{sequence.contract_type}" == sequence_key:
                return False, "active_sequence_exists"
        
        # Check available balance
        risk_budget = self.current_balance * self.config.risk_budget_pct
        if risk_budget < config.MIN_STAKE * 3:  # Need at least 3 trades worth
            return False, "insufficient_balance"
        
        return True, "approved"
    
    def start_sequence(self, symbol: str, contract_type: str, base_stake: float,
                      probability: float, confidence: float, payout_ratio: float) -> str:
        """Start a new martingale recovery sequence"""
        try:
            current_time = time.time()
            sequence_id = f"mg_{int(current_time)}_{symbol}_{contract_type}"
            
            # Calculate risk budget
            risk_budget = self.current_balance * self.config.risk_budget_pct
            
            # Create sequence
            sequence = MartingaleSequence(
                sequence_id=sequence_id,
                start_time=current_time,
                symbol=symbol,
                contract_type=contract_type,
                base_stake=base_stake,
                current_level=0,
                max_levels=self.config.max_levels,
                total_spent=0.0,
                total_risk_budget=risk_budget,
                payout_ratio=payout_ratio,
                confidence_threshold=self.config.confidence_threshold,
                last_trade_time=current_time,
                trades=[],
                is_active=True
            )
            
            # Store sequence
            self.active_sequences[sequence_id] = sequence
            self.last_sequence_time = current_time
            self.daily_sequences += 1
            
            # Log to database
            self._log_sequence_start(sequence)
            
            logger.info(f"Started martingale sequence: {sequence_id}")
            return sequence_id
            
        except Exception as e:
            logger.error(f"Error starting martingale sequence: {e}")
            return None
    
    def calculate_next_stake(self, sequence_id: str, current_confidence: float,
                           current_payout: float) -> Tuple[float, bool, str]:
        """Calculate the next stake in the sequence"""
        try:
            sequence = self.active_sequences.get(sequence_id)
            if not sequence or not sequence.is_active:
                return 0.0, False, "sequence_not_active"
            
            # Check timeout
            if time.time() - sequence.start_time > self.config.timeout_seconds:
                self._stop_sequence(sequence_id, "timeout")
                return 0.0, False, "sequence_timeout"
            
            # Check confidence threshold
            if current_confidence < sequence.confidence_threshold:
                self._stop_sequence(sequence_id, "confidence_dropped")
                return 0.0, False, "confidence_too_low"
            
            # Check if we've reached max levels
            if sequence.current_level >= sequence.max_levels:
                self._stop_sequence(sequence_id, "max_levels_reached")
                return 0.0, False, "max_levels_reached"
            
            # Calculate required stake for recovery
            if self.config.payout_adjustment:
                # Payout-adjusted multiplier
                net_payout_ratio = (current_payout - 1.0) if current_payout > 1.0 else 0.8
                if net_payout_ratio <= 0:
                    self._stop_sequence(sequence_id, "invalid_payout")
                    return 0.0, False, "invalid_payout"
                
                # Calculate stake needed to recover all losses plus base profit
                recovery_target = sequence.total_spent + sequence.base_stake
                required_stake = recovery_target / net_payout_ratio
                
            else:
                # Simple doubling strategy
                if sequence.current_level == 0:
                    required_stake = sequence.base_stake
                else:
                    required_stake = sequence.base_stake * (2 ** sequence.current_level)
            
            # Apply safety caps
            max_allowed_stake = min(
                sequence.total_risk_budget - sequence.total_spent,
                self.current_balance * 0.1,  # Never risk more than 10% of balance in one trade
                config.MAX_STAKE
            )
            
            final_stake = min(required_stake, max_allowed_stake)
            
            # Check if stake is viable
            if final_stake < config.MIN_STAKE:
                self._stop_sequence(sequence_id, "stake_too_small")
                return 0.0, False, "stake_too_small"
            
            # Check if this would exceed risk budget
            if sequence.total_spent + final_stake > sequence.total_risk_budget:
                self._stop_sequence(sequence_id, "risk_budget_exceeded")
                return 0.0, False, "risk_budget_exceeded"
            
            return final_stake, True, "approved"
            
        except Exception as e:
            logger.error(f"Error calculating next stake: {e}")
            return 0.0, False, f"error_{str(e)}"
    
    def record_trade(self, sequence_id: str, stake: float, confidence: float,
                    probability: float, expected_value: float, 
                    actual_payout: float, profit_loss: float, win: bool,
                    contract_id: str = None) -> bool:
        """Record a trade in the martingale sequence"""
        try:
            sequence = self.active_sequences.get(sequence_id)
            if not sequence:
                return False
            
            current_time = time.time()
            
            # Update sequence
            sequence.total_spent += stake if not win else 0  # Only count losses
            sequence.current_level += 1
            sequence.last_trade_time = current_time
            
            # Create trade record
            trade_record = {
                'level': sequence.current_level,
                'timestamp': current_time,
                'stake': stake,
                'confidence': confidence,
                'probability': probability,
                'expected_value': expected_value,
                'actual_payout': actual_payout,
                'profit_loss': profit_loss,
                'win': win,
                'cumulative_loss': sequence.total_spent,
                'contract_id': contract_id
            }
            
            sequence.trades.append(trade_record)
            
            # Log to database
            self._log_trade(sequence_id, trade_record)
            
            # Check if sequence should end
            if win:
                # Success! Calculate total recovery
                total_recovery = sum(trade['profit_loss'] for trade in sequence.trades)
                self._complete_sequence(sequence_id, True, total_recovery)
                logger.info(f"Martingale sequence {sequence_id} completed successfully: ${total_recovery:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording martingale trade: {e}")
            return False
    
    def _stop_sequence(self, sequence_id: str, reason: str):
        """Stop an active sequence"""
        try:
            sequence = self.active_sequences.get(sequence_id)
            if not sequence:
                return
            
            sequence.is_active = False
            total_loss = -sequence.total_spent  # Negative because it's a loss
            
            self._complete_sequence(sequence_id, False, total_loss, reason)
            logger.warning(f"Martingale sequence {sequence_id} stopped: {reason} (loss: ${total_loss:.2f})")
            
        except Exception as e:
            logger.error(f"Error stopping sequence: {e}")
    
    def _complete_sequence(self, sequence_id: str, success: bool, 
                          final_profit: float, reason: str = None):
        """Complete a martingale sequence and move to history"""
        try:
            sequence = self.active_sequences.get(sequence_id)
            if not sequence:
                return
            
            # Update database
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE martingale_sequences 
                SET end_time = ?, levels_used = ?, total_recovered = ?, 
                    final_profit = ?, success = ?, reason_stopped = ?,
                    trades_data = ?
                WHERE sequence_id = ?
            """, (
                int(time.time()), sequence.current_level, 
                final_profit if success else 0, final_profit,
                int(success), reason or ("success" if success else "failure"),
                json.dumps(sequence.trades), sequence_id
            ))
            
            conn.commit()
            conn.close()
            
            # Move to completed sequences
            self.completed_sequences.append({
                'sequence_id': sequence_id,
                'success': success,
                'final_profit': final_profit,
                'levels_used': sequence.current_level,
                'total_spent': sequence.total_spent,
                'completion_time': time.time()
            })
            
            # Remove from active sequences
            del self.active_sequences[sequence_id]
            
        except Exception as e:
            logger.error(f"Error completing sequence: {e}")
    
    def _log_sequence_start(self, sequence: MartingaleSequence):
        """Log sequence start to database"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO martingale_sequences 
                (sequence_id, symbol, contract_type, start_time, base_stake,
                 max_levels, risk_budget, confidence_threshold)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sequence.sequence_id, sequence.symbol, sequence.contract_type,
                int(sequence.start_time), sequence.base_stake, sequence.max_levels,
                sequence.total_risk_budget, sequence.confidence_threshold
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging sequence start: {e}")
    
    def _log_trade(self, sequence_id: str, trade_record: Dict):
        """Log individual trade to database"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO martingale_trades 
                (sequence_id, level, timestamp, stake, confidence, probability,
                 expected_value, actual_payout, profit_loss, win, cumulative_loss,
                 recovery_target, contract_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sequence_id, trade_record['level'], int(trade_record['timestamp']),
                trade_record['stake'], trade_record['confidence'], 
                trade_record['probability'], trade_record['expected_value'],
                trade_record['actual_payout'], trade_record['profit_loss'],
                int(trade_record['win']), trade_record['cumulative_loss'],
                trade_record['cumulative_loss'] + trade_record['stake'],
                trade_record.get('contract_id')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
    
    def get_active_sequences(self) -> Dict[str, MartingaleSequence]:
        """Get all active sequences"""
        return self.active_sequences.copy()
    
    def get_sequence_status(self, sequence_id: str) -> Dict[str, Any]:
        """Get detailed status of a sequence"""
        sequence = self.active_sequences.get(sequence_id)
        if not sequence:
            return {'error': 'sequence_not_found'}
        
        return {
            'sequence_id': sequence.sequence_id,
            'symbol': sequence.symbol,
            'contract_type': sequence.contract_type,
            'current_level': sequence.current_level,
            'max_levels': sequence.max_levels,
            'total_spent': sequence.total_spent,
            'risk_budget': sequence.total_risk_budget,
            'remaining_budget': sequence.total_risk_budget - sequence.total_spent,
            'is_active': sequence.is_active,
            'trades_count': len(sequence.trades),
            'time_elapsed': time.time() - sequence.start_time,
            'last_trade': sequence.trades[-1] if sequence.trades else None
        }
    
    def get_performance_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get martingale system performance statistics"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            
            cutoff_time = int(time.time()) - (days * 24 * 3600)
            
            # Get sequence statistics
            df_sequences = pd.read_sql_query("""
                SELECT * FROM martingale_sequences 
                WHERE start_time > ? AND end_time IS NOT NULL
                ORDER BY start_time DESC
            """, conn, params=(cutoff_time,))
            
            # Get trade statistics
            df_trades = pd.read_sql_query("""
                SELECT mt.*, ms.success as sequence_success
                FROM martingale_trades mt
                JOIN martingale_sequences ms ON mt.sequence_id = ms.sequence_id
                WHERE mt.timestamp > ?
                ORDER BY mt.timestamp DESC
            """, conn, params=(cutoff_time,))
            
            conn.close()
            
            if len(df_sequences) == 0:
                return {'error': 'no_data', 'period_days': days}
            
            # Calculate performance metrics
            total_sequences = len(df_sequences)
            successful_sequences = len(df_sequences[df_sequences['success'] == 1])
            success_rate = successful_sequences / total_sequences
            
            total_risk_budget = df_sequences['risk_budget'].sum()
            total_profit = df_sequences['final_profit'].sum()
            net_return = total_profit / total_risk_budget if total_risk_budget > 0 else 0
            
            avg_levels_used = df_sequences['levels_used'].mean()
            max_levels_used = df_sequences['levels_used'].max()
            
            # Trade-level statistics
            if len(df_trades) > 0:
                total_trades = len(df_trades)
                winning_trades = len(df_trades[df_trades['win'] == 1])
                trade_win_rate = winning_trades / total_trades
                avg_stake = df_trades['stake'].mean()
                
                # Risk metrics
                max_single_loss = df_trades['profit_loss'].min()
                max_cumulative_loss = df_trades['cumulative_loss'].max()
            else:
                total_trades = trade_win_rate = avg_stake = 0
                max_single_loss = max_cumulative_loss = 0
            
            return {
                'period_days': days,
                'total_sequences': total_sequences,
                'successful_sequences': successful_sequences,
                'success_rate': success_rate,
                'total_risk_budget': total_risk_budget,
                'total_profit': total_profit,
                'net_return': net_return,
                'avg_levels_used': avg_levels_used,
                'max_levels_used': max_levels_used,
                'total_trades': total_trades,
                'trade_win_rate': trade_win_rate,
                'avg_stake': avg_stake,
                'max_single_loss': max_single_loss,
                'max_cumulative_loss': max_cumulative_loss,
                'active_sequences': len(self.active_sequences)
            }
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {'error': str(e)}
    
    def cleanup_expired_sequences(self):
        """Clean up expired or stuck sequences"""
        current_time = time.time()
        expired_sequences = []
        
        for sequence_id, sequence in self.active_sequences.items():
            if (current_time - sequence.last_trade_time > self.config.timeout_seconds or
                not sequence.is_active):
                expired_sequences.append(sequence_id)
        
        for sequence_id in expired_sequences:
            self._stop_sequence(sequence_id, "expired")
        
        if expired_sequences:
            logger.info(f"Cleaned up {len(expired_sequences)} expired sequences")
    
    def update_balance(self, new_balance: float):
        """Update current balance for risk calculations"""
        self.current_balance = new_balance

# Simulation function for testing
def simulate_martingale_sequences(num_simulations: int = 10000, 
                                win_probability: float = 0.7,
                                payout_ratio: float = 1.8) -> Dict[str, Any]:
    """Monte Carlo simulation of martingale sequences"""
    results = []
    config = MartingaleConfig()
    
    for _ in range(num_simulations):
        total_spent = 0
        base_stake = 2.0
        level = 0
        
        while level < config.max_levels:
            # Calculate stake (payout-adjusted)
            if level == 0:
                stake = base_stake
            else:
                recovery_target = total_spent + base_stake
                stake = recovery_target / (payout_ratio - 1.0)
            
            total_spent += stake
            
            # Simulate outcome
            win = np.random.random() < win_probability
            
            if win:
                # Calculate profit
                payout = stake * payout_ratio
                net_profit = payout - total_spent
                results.append({
                    'success': True,
                    'levels_used': level + 1,
                    'total_spent': total_spent,
                    'final_profit': net_profit
                })
                break
            
            level += 1
        
        else:
            # Failed to recover
            results.append({
                'success': False,
                'levels_used': config.max_levels,
                'total_spent': total_spent,
                'final_profit': -total_spent
            })
    
    # Calculate statistics
    df = pd.DataFrame(results)
    
    success_rate = df['success'].mean()
    avg_profit = df['final_profit'].mean()
    avg_levels = df['levels_used'].mean()
    max_loss = df['final_profit'].min()
    
    return {
        'simulations': num_simulations,
        'win_probability': win_probability,
        'payout_ratio': payout_ratio,
        'success_rate': success_rate,
        'avg_profit': avg_profit,
        'avg_levels_used': avg_levels,
        'max_loss': max_loss,
        'results': results[:100]  # Return first 100 for analysis
    }

# Global martingale system instance
martingale_system = MartingaleRecoverySystem()

if __name__ == "__main__":
    import pandas as pd
    
    # Test martingale system
    system = MartingaleRecoverySystem()
    system.update_balance(1000.0)
    
    print("Testing Martingale Recovery System...")
    
    # Test sequence creation
    should_start, reason = system.should_start_sequence(
        "R_100", "DIGITEVEN", 0.72, 0.75, 0.05
    )
    
    print(f"Should start sequence: {should_start} ({reason})")
    
    if should_start:
        sequence_id = system.start_sequence(
            "R_100", "DIGITEVEN", 2.0, 0.72, 0.75, 1.8
        )
        
        print(f"Started sequence: {sequence_id}")
        
        # Simulate some trades
        for level in range(3):
            stake, approved, reason = system.calculate_next_stake(
                sequence_id, 0.75, 1.8
            )
            
            print(f"Level {level}: Stake ${stake:.2f}, Approved: {approved} ({reason})")
            
            if approved:
                # Simulate trade outcome
                win = np.random.random() < 0.72
                profit = stake * 0.8 if win else -stake
                
                system.record_trade(
                    sequence_id, stake, 0.75, 0.72, 0.05,
                    1.8 if win else 0, profit, win
                )
                
                print(f"Trade result: {'WIN' if win else 'LOSS'} ${profit:.2f}")
                
                if win:
                    break
            else:
                break
        
        # Get sequence status
        status = system.get_sequence_status(sequence_id)
        print(f"Sequence status: {status}")
    
    # Run simulation
    print("\nRunning Monte Carlo simulation...")
    sim_results = simulate_martingale_sequences(1000, 0.7, 1.8)
    
    print(f"Simulation Results:")
    print(f"Success Rate: {sim_results['success_rate']:.2%}")
    print(f"Average Profit: ${sim_results['avg_profit']:.2f}")
    print(f"Average Levels: {sim_results['avg_levels_used']:.1f}")
    print(f"Maximum Loss: ${sim_results['max_loss']:.2f}")
    
    # Get performance stats
    perf_stats = system.get_performance_stats()
    print(f"Performance Stats: {perf_stats}")
