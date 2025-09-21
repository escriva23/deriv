# signal_manager.py - Redis-based signal exchange system
import redis
import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from shared_config import config, SIGNAL_TYPES, BOT_IDS

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    bot_id: str
    signal_type: str
    timestamp: float
    symbol: str
    contract_type: str
    probability: float
    confidence: float
    expected_payout: float
    stake: float
    barrier: Optional[str] = None
    reasoning: Optional[str] = None
    features: Optional[Dict] = None

@dataclass
class TradeResult:
    bot_id: str
    signal_id: str
    timestamp: float
    contract_id: str
    symbol: str
    contract_type: str
    stake: float
    profit: float
    win: bool
    entry_spot: float
    exit_spot: float
    execution_time: float

@dataclass
class PerformanceUpdate:
    bot_id: str
    timestamp: float
    total_trades: int
    winning_trades: int
    win_rate: float
    total_profit: float
    avg_profit_per_trade: float
    recent_performance: float  # last N trades win rate

class SignalManager:
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                password=config.REDIS_PASSWORD,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def publish_signal(self, signal: TradingSignal) -> bool:
        """Publish a trading signal to Redis"""
        try:
            signal_data = {
                **asdict(signal),
                'signal_id': f"{signal.bot_id}_{int(signal.timestamp * 1000)}"
            }
            
            message = json.dumps(signal_data)
            
            # Publish to main channel
            self.redis_client.publish(config.SIGNAL_CHANNEL, message)
            
            # Store in Redis with expiry
            key = f"signal:{signal_data['signal_id']}"
            self.redis_client.setex(key, config.SIGNAL_EXPIRY, message)
            
            logger.debug(f"Published signal: {signal.bot_id} -> {signal.contract_type} ({signal.probability:.3f})")
            return True
            
        except Exception as e:
            logger.error(f"Error publishing signal: {e}")
            return False
    
    def publish_trade_result(self, result: TradeResult) -> bool:
        """Publish trade result"""
        try:
            result_data = {
                **asdict(result),
                'result_id': f"{result.bot_id}_{int(result.timestamp * 1000)}"
            }
            
            message = json.dumps(result_data)
            
            # Publish to results channel
            self.redis_client.publish(f"{config.SIGNAL_CHANNEL}_results", message)
            
            # Store in Redis
            key = f"result:{result_data['result_id']}"
            self.redis_client.setex(key, 3600, message)  # Keep results for 1 hour
            
            logger.debug(f"Published result: {result.bot_id} -> {'WIN' if result.win else 'LOSS'} (${result.profit:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Error publishing trade result: {e}")
            return False
    
    def publish_performance_update(self, performance: PerformanceUpdate) -> bool:
        """Publish performance update"""
        try:
            perf_data = asdict(performance)
            message = json.dumps(perf_data)
            
            # Publish to performance channel
            self.redis_client.publish(f"{config.SIGNAL_CHANNEL}_performance", message)
            
            # Store latest performance
            key = f"performance:{performance.bot_id}"
            self.redis_client.set(key, message)
            
            logger.debug(f"Published performance: {performance.bot_id} -> {performance.win_rate:.3f} win rate")
            return True
            
        except Exception as e:
            logger.error(f"Error publishing performance: {e}")
            return False
    
    def subscribe_to_signals(self, callback_func):
        """Subscribe to trading signals"""
        try:
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe(config.SIGNAL_CHANNEL)
            
            logger.info("Subscribed to trading signals")
            
            for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        signal_data = json.loads(message['data'])
                        
                        # Check signal age
                        signal_age = time.time() - signal_data['timestamp']
                        if signal_age > config.MAX_SIGNAL_AGE:
                            logger.debug(f"Ignoring stale signal (age: {signal_age:.3f}s)")
                            continue
                        
                        # Convert back to TradingSignal object
                        signal = TradingSignal(**{k: v for k, v in signal_data.items() 
                                                if k in TradingSignal.__annotations__})
                        
                        callback_func(signal)
                        
                    except Exception as e:
                        logger.error(f"Error processing signal: {e}")
                        
        except Exception as e:
            logger.error(f"Error in signal subscription: {e}")
    
    def subscribe_to_results(self, callback_func):
        """Subscribe to trade results"""
        try:
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe(f"{config.SIGNAL_CHANNEL}_results")
            
            logger.info("Subscribed to trade results")
            
            for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        result_data = json.loads(message['data'])
                        result = TradeResult(**{k: v for k, v in result_data.items() 
                                              if k in TradeResult.__annotations__})
                        callback_func(result)
                        
                    except Exception as e:
                        logger.error(f"Error processing result: {e}")
                        
        except Exception as e:
            logger.error(f"Error in results subscription: {e}")
    
    def subscribe_to_performance(self, callback_func):
        """Subscribe to performance updates"""
        try:
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe(f"{config.SIGNAL_CHANNEL}_performance")
            
            logger.info("Subscribed to performance updates")
            
            for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        perf_data = json.loads(message['data'])
                        performance = PerformanceUpdate(**perf_data)
                        callback_func(performance)
                        
                    except Exception as e:
                        logger.error(f"Error processing performance: {e}")
                        
        except Exception as e:
            logger.error(f"Error in performance subscription: {e}")
    
    def get_recent_signals(self, max_age: float = 1.0) -> List[TradingSignal]:
        """Get recent signals from Redis"""
        try:
            signals = []
            current_time = time.time()
            
            # Get all signal keys
            signal_keys = self.redis_client.keys("signal:*")
            
            for key in signal_keys:
                signal_data = self.redis_client.get(key)
                if signal_data:
                    data = json.loads(signal_data)
                    signal_age = current_time - data['timestamp']
                    
                    if signal_age <= max_age:
                        signal = TradingSignal(**{k: v for k, v in data.items() 
                                                if k in TradingSignal.__annotations__})
                        signals.append(signal)
            
            return sorted(signals, key=lambda x: x.timestamp, reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting recent signals: {e}")
            return []
    
    def get_bot_performance(self, bot_id: str) -> Optional[PerformanceUpdate]:
        """Get latest performance for a bot"""
        try:
            key = f"performance:{bot_id}"
            perf_data = self.redis_client.get(key)
            
            if perf_data:
                data = json.loads(perf_data)
                return PerformanceUpdate(**data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting bot performance: {e}")
            return None
    
    def get_all_bot_performances(self) -> Dict[str, PerformanceUpdate]:
        """Get performance for all bots"""
        try:
            performances = {}
            
            for bot_id in BOT_IDS.values():
                perf = self.get_bot_performance(bot_id)
                if perf:
                    performances[bot_id] = perf
            
            return performances
            
        except Exception as e:
            logger.error(f"Error getting all performances: {e}")
            return {}
    
    def cleanup_old_data(self):
        """Clean up old signals and results"""
        try:
            current_time = time.time()
            
            # Clean old signals (older than 5 minutes)
            signal_keys = self.redis_client.keys("signal:*")
            for key in signal_keys:
                signal_data = self.redis_client.get(key)
                if signal_data:
                    data = json.loads(signal_data)
                    if current_time - data['timestamp'] > 300:  # 5 minutes
                        self.redis_client.delete(key)
            
            # Clean old results (older than 24 hours)
            result_keys = self.redis_client.keys("result:*")
            for key in result_keys:
                result_data = self.redis_client.get(key)
                if result_data:
                    data = json.loads(result_data)
                    if current_time - data['timestamp'] > 86400:  # 24 hours
                        self.redis_client.delete(key)
            
            logger.debug("Cleaned up old Redis data")
            
        except Exception as e:
            logger.error(f"Error cleaning up Redis data: {e}")
    
    def close(self):
        """Close Redis connection"""
        try:
            self.redis_client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")

# Global signal manager instance
signal_manager = SignalManager()
