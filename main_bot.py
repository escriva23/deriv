# main_bot.py - Main AI Trading Bot Orchestrator
import os
import time
import logging
import threading
import signal
import sys
from datetime import datetime
from typing import Dict, Optional

from config import config
from data_collector import TickDataCollector
from ai_predictor import AIPredictor
from risk_manager import RiskManager
from trading_executor import TradingExecutor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DerivAITradingBot:
    def __init__(self, demo_mode: bool = True):
        self.demo_mode = demo_mode
        self.running = False
        self.paused = False
        
        # Initialize components
        self.data_collector = TickDataCollector()
        self.ai_predictor = AIPredictor()
        self.risk_manager = RiskManager(initial_balance=1000.0 if demo_mode else 10000.0)
        
        # Trading executor (requires token)
        token = os.getenv("DERIV_TOKEN")
        if not token:
            logger.error("DERIV_TOKEN environment variable not set!")
            raise ValueError("Please set your Deriv API token")
        
        self.executor = TradingExecutor(token)
        
        # Threading
        self.collector_thread = None
        self.trading_thread = None
        self.monitor_thread = None
        
        # Statistics
        self.session_stats = {
            'start_time': time.time(),
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def initialize(self) -> bool:
        """Initialize all components"""
        logger.info("Initializing Deriv AI Trading Bot...")
        
        # Connect to Deriv
        if not self.executor.connect():
            logger.error("Failed to connect to Deriv API")
            return False
        
        # Check account balance
        balance = self.executor.get_account_balance()
        if balance is None:
            logger.error("Failed to get account balance")
            return False
        
        logger.info(f"Account balance: ${balance:.2f}")
        self.risk_manager.current_balance = balance
        self.risk_manager.daily_start_balance = balance
        
        # Load or train AI models
        logger.info("Loading AI models...")
        models_loaded = 0
        for symbol in config.SYMBOLS:
            for pred_type in ['digit_parity', 'digit_over_under']:
                try:
                    # Try to load existing model
                    import joblib
                    key = f"{symbol}_{pred_type}"
                    model_file = f"model_{key}_xgboost.pkl"
                    scaler_file = f"scaler_{key}.pkl"
                    
                    if os.path.exists(model_file) and os.path.exists(scaler_file):
                        self.ai_predictor.models[key] = joblib.load(model_file)
                        self.ai_predictor.scalers[key] = joblib.load(scaler_file)
                        models_loaded += 1
                        logger.info(f"Loaded model: {key}")
                    else:
                        logger.info(f"No existing model for {key}, will train after data collection")
                except Exception as e:
                    logger.error(f"Error loading model {key}: {e}")
        
        logger.info(f"Loaded {models_loaded} existing models")
        
        logger.info("✅ Bot initialized successfully!")
        return True
    
    def start_data_collection(self):
        """Start background data collection"""
        logger.info("Starting data collection...")
        self.collector_thread = threading.Thread(
            target=self.data_collector.start_collection,
            daemon=True
        )
        self.collector_thread.start()
        
        # Wait a bit for initial data
        time.sleep(5)
    
    def train_models_if_needed(self):
        """Train AI models if we have enough data"""
        logger.info("Checking if models need training...")
        
        for symbol in config.SYMBOLS:
            try:
                # Check if we have enough data
                recent_features = self.data_collector.get_recent_features(symbol, 1000)
                
                if len(recent_features) >= 500:
                    for pred_type in ['digit_parity', 'digit_over_under']:
                        key = f"{symbol}_{pred_type}"
                        
                        if key not in self.ai_predictor.models:
                            logger.info(f"Training model: {key}")
                            success = self.ai_predictor.train_models(symbol, pred_type)
                            if success:
                                logger.info(f"✅ Model trained: {key}")
                            else:
                                logger.warning(f"❌ Failed to train: {key}")
                else:
                    logger.info(f"Not enough data for {symbol}: {len(recent_features)} samples")
                    
            except Exception as e:
                logger.error(f"Error training models for {symbol}: {e}")
    
    def make_trading_decision(self, symbol: str) -> Optional[Dict]:
        """Make a trading decision for a symbol"""
        try:
            # Get recent features
            recent_features = self.data_collector.get_recent_features(symbol, 1)
            
            if len(recent_features) == 0:
                return None
            
            # Convert to dict format expected by predictor
            latest_features = recent_features.iloc[-1].to_dict()
            
            # Get AI predictions
            predictions = self.ai_predictor.get_ensemble_prediction(symbol, latest_features)
            
            # Check if we should trade
            should_trade, contract_type, confidence = self.ai_predictor.should_trade(predictions)
            
            if not should_trade:
                return None
            
            # Check risk management
            allowed, reason = self.risk_manager.check_trading_allowed()
            if not allowed:
                logger.info(f"Trading not allowed: {reason}")
                return None
            
            # Calculate position size
            stake = self.risk_manager.calculate_position_size(confidence, symbol)
            
            return {
                'symbol': symbol,
                'contract_type': contract_type,
                'stake': stake,
                'confidence': confidence,
                'predictions': predictions,
                'features': latest_features
            }
            
        except Exception as e:
            logger.error(f"Error making trading decision for {symbol}: {e}")
            return None
    
    def execute_trading_decision(self, decision: Dict) -> Optional[Dict]:
        """Execute a trading decision"""
        try:
            symbol = decision['symbol']
            contract_type = decision['contract_type']
            stake = decision['stake']
            confidence = decision['confidence']
            
            logger.info(f"🎯 TRADING: {contract_type} {symbol} ${stake:.2f} (confidence: {confidence:.3f})")
            
            # Determine barrier for digit contracts
            barrier = None
            if contract_type in ['DIGITOVER', 'DIGITUNDER']:
                barrier = "5"  # Default barrier
            
            # Execute trade
            result = self.executor.execute_and_wait(
                contract_type=contract_type,
                symbol=symbol,
                stake=stake,
                confidence=confidence,
                barrier=barrier
            )
            
            if result and result.get('success'):
                # Log trade to database
                self.log_trade(decision, result)
                
                # Update risk management
                profit = result.get('profit', 0)
                risk_update = self.risk_manager.update_balance(profit, contract_type, stake)
                
                # Update session stats
                self.session_stats['total_trades'] += 1
                self.session_stats['total_profit'] += profit
                
                if profit > 0:
                    self.session_stats['winning_trades'] += 1
                    logger.info(f"✅ WIN: ${profit:.2f} | Balance: ${risk_update['new_balance']:.2f}")
                else:
                    self.session_stats['losing_trades'] += 1
                    logger.info(f"❌ LOSS: ${profit:.2f} | Balance: ${risk_update['new_balance']:.2f}")
                
                # Update AI model performance
                if 'exit_spot' in result and 'entry_spot' in result:
                    self.update_model_performance(decision, result)
                
                return result
            else:
                logger.error(f"Trade execution failed: {result}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None
    
    def log_trade(self, decision: Dict, result: Dict):
        """Log trade to database"""
        try:
            import sqlite3
            import json
            
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trades 
                (contract_id, symbol, contract_type, stake, prediction_confidence,
                 entry_time, exit_time, profit_loss, win, features_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.get('contract_id', ''),
                decision['symbol'],
                decision['contract_type'],
                decision['stake'],
                decision['confidence'],
                result.get('timestamp', int(time.time())),
                result.get('exit_time', int(time.time())),
                result.get('profit', 0),
                1 if result.get('profit', 0) > 0 else 0,
                json.dumps(decision['features'])
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
    
    def update_model_performance(self, decision: Dict, result: Dict):
        """Update AI model performance tracking"""
        try:
            # Determine actual outcome based on contract type and result
            contract_type = decision['contract_type']
            profit = result.get('profit', 0)
            
            # For now, just track if the trade was profitable
            actual_outcome = 1 if profit > 0 else 0
            
            # Get the prediction that was used
            predictions = decision['predictions']
            for pred_type, pred_data in predictions.items():
                predicted_outcome = pred_data['prediction']
                
                self.ai_predictor.update_model_performance(
                    decision['symbol'],
                    pred_type,
                    predicted_outcome,
                    actual_outcome,
                    profit
                )
                
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
    
    def trading_loop(self):
        """Main trading loop"""
        logger.info("Starting trading loop...")
        
        while self.running:
            try:
                if self.paused:
                    time.sleep(1)
                    continue
                
                # Check each symbol for trading opportunities
                for symbol in config.SYMBOLS:
                    if not self.running:
                        break
                    
                    # Make trading decision
                    decision = self.make_trading_decision(symbol)
                    
                    if decision:
                        # Execute the trade
                        result = self.execute_trading_decision(decision)
                        
                        if result:
                            # Wait a bit before next trade
                            time.sleep(2)
                
                # Wait before next iteration
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(5)
    
    def monitor_loop(self):
        """Monitor bot performance and health"""
        logger.info("Starting monitoring loop...")
        
        while self.running:
            try:
                # Print status every 60 seconds
                time.sleep(60)
                
                if not self.running:
                    break
                
                # Get current metrics
                risk_metrics = self.risk_manager.get_risk_metrics()
                
                # Print status
                logger.info("=" * 60)
                logger.info("📊 BOT STATUS")
                logger.info(f"Balance: ${risk_metrics.current_balance:.2f}")
                logger.info(f"Daily P&L: ${risk_metrics.daily_pnl:.2f}")
                logger.info(f"Total Trades: {risk_metrics.total_trades}")
                logger.info(f"Win Rate: {risk_metrics.win_rate:.1%}")
                logger.info(f"Consecutive Losses: {risk_metrics.consecutive_losses}")
                logger.info(f"Max Drawdown: {risk_metrics.max_drawdown:.1%}")
                
                # Check for emergency conditions
                if risk_metrics.consecutive_losses >= config.MAX_CONSECUTIVE_LOSSES:
                    logger.warning("⚠️ Max consecutive losses reached!")
                    self.pause()
                
                if risk_metrics.daily_pnl <= -config.MAX_DAILY_LOSS:
                    logger.warning("⚠️ Daily loss limit reached!")
                    self.pause()
                
                logger.info("=" * 60)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(60)
    
    def start(self):
        """Start the trading bot"""
        if not self.initialize():
            logger.error("Failed to initialize bot")
            return False
        
        self.running = True
        
        # Start data collection
        self.start_data_collection()
        
        # Wait for initial data and train models if needed
        logger.info("Waiting for initial data...")
        time.sleep(30)  # Wait 30 seconds for data collection
        
        self.train_models_if_needed()
        
        # Start trading and monitoring threads
        self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        
        self.trading_thread.start()
        self.monitor_thread.start()
        
        logger.info("🚀 Deriv AI Trading Bot started successfully!")
        logger.info(f"Mode: {'DEMO' if self.demo_mode else 'LIVE'}")
        logger.info("Press Ctrl+C to stop")
        
        return True
    
    def pause(self):
        """Pause trading (but keep data collection)"""
        self.paused = True
        logger.info("⏸️ Trading paused")
    
    def resume(self):
        """Resume trading"""
        self.paused = False
        logger.info("▶️ Trading resumed")
    
    def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping trading bot...")
        
        self.running = False
        self.paused = False
        
        # Stop data collection
        if self.data_collector:
            self.data_collector.stop_collection()
        
        # Cleanup executor
        if self.executor:
            self.executor.cleanup()
        
        # Wait for threads to finish
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=5)
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("🛑 Trading bot stopped")
    
    def get_status(self) -> Dict:
        """Get current bot status"""
        risk_metrics = self.risk_manager.get_risk_metrics()
        
        return {
            'running': self.running,
            'paused': self.paused,
            'demo_mode': self.demo_mode,
            'balance': risk_metrics.current_balance,
            'daily_pnl': risk_metrics.daily_pnl,
            'total_trades': risk_metrics.total_trades,
            'win_rate': risk_metrics.win_rate,
            'consecutive_losses': risk_metrics.consecutive_losses,
            'session_stats': self.session_stats
        }

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deriv AI Trading Bot')
    parser.add_argument('--live', action='store_true', help='Run in live mode (default: demo)')
    parser.add_argument('--collect-only', action='store_true', help='Only collect data, no trading')
    
    args = parser.parse_args()
    
    # Create and start bot
    bot = DerivAITradingBot(demo_mode=not args.live)
    
    if args.collect_only:
        logger.info("Data collection mode - no trading will occur")
        bot.start_data_collection()
        try:
            while True:
                time.sleep(60)
                logger.info("Data collection running...")
        except KeyboardInterrupt:
            logger.info("Stopping data collection...")
            bot.data_collector.stop_collection()
    else:
        try:
            if bot.start():
                # Keep main thread alive
                while bot.running:
                    time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            bot.stop()

if __name__ == "__main__":
    main()
