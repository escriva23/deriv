# enhanced_coordinator.py - Enhanced Coordinator with Pattern-Aware Intelligence
import time
import json
import logging
import threading
import sqlite3
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
from websocket import create_connection, WebSocketConnectionClosedException

# Original imports
from signal_manager import signal_manager, TradingSignal, TradeResult, PerformanceUpdate
from shared_config import config, BOT_IDS, SIGNAL_TYPES
from risk_manager import RiskManager

# Enhanced imports
from pattern_calibration import ProbabilityCalibrator
from pattern_detectors import AdvancedPatternEngine
from meta_controller import MetaController
from martingale_system import MartingaleRecoverySystem
from online_learning import AdaptiveLearningSystem
from enhanced_ai_predictor import EnhancedAIPredictor

logger = logging.getLogger(__name__)

class EnhancedTradingCoordinator:
    def __init__(self):
        self.bot_id = BOT_IDS["COORDINATOR"]
        self.token = config.COORDINATOR_TOKEN
        self.ws = None
        self.running = False
        
        # Original signal processing
        self.recent_signals = deque(maxlen=100)
        self.probe_performances = {}
        self.signal_history = defaultdict(list)
        
        # Enhanced AI components
        self.calibrator = ProbabilityCalibrator()
        self.pattern_engine = AdvancedPatternEngine()
        self.meta_controller = MetaController()
        self.martingale_system = MartingaleRecoverySystem()
        self.online_learner = AdaptiveLearningSystem()
        self.enhanced_predictor = EnhancedAIPredictor()
        
        # Risk management (enhanced)
        self.risk_manager = RiskManager(config.INITIAL_BALANCE)
        self.current_balance = 0.0  # Will be initialized from API
        self.daily_start_balance = 0.0  # Will be initialized from API
        
        # Performance tracking (enhanced)
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.recent_trades = deque(maxlen=50)
        
        # Decision making (enhanced)
        self.last_trade_time = 0
        self.min_signal_interval = 5.0  # Minimum seconds between decisions
        self.current_tick_data = None
        
        logger.info(f"Enhanced Coordinator initialized: {self.bot_id}")
        logger.info("Enhanced features: Pattern detection, Calibration, Meta-controller, Martingale")
    
    def connect_deriv(self) -> bool:
        """Connect to Deriv WebSocket API"""
        try:
            url = f"{config.DERIV_WS_URL}?app_id={config.DERIV_APP_ID}"
            self.ws = create_connection(url)
            
            # Authorize
            auth_msg = {"authorize": self.token}
            self.ws.send(json.dumps(auth_msg))
            
            response = json.loads(self.ws.recv())
            
            if "authorize" in response and response["authorize"].get("loginid"):
                logger.info(f"Enhanced Coordinator authorized: {response['authorize']['loginid']}")
                return True
            else:
                logger.error(f"Enhanced Coordinator authorization failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Enhanced Coordinator connection failed: {e}")
            return False
    
    def subscribe_to_ticks(self):
        """Subscribe to tick stream for pattern analysis"""
        try:
            subscribe_msg = {
                "ticks": config.PRIMARY_SYMBOL,
                "subscribe": 1
            }
            self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Enhanced Coordinator subscribed to {config.PRIMARY_SYMBOL} ticks")
            
        except Exception as e:
            logger.error(f"Enhanced Coordinator tick subscription failed: {e}")
    
    def run(self):
        """Main coordinator loop with enhanced intelligence"""
        try:
            if not self.connect_deriv():
                logger.error("Enhanced Coordinator failed to connect")
                return
            
            self.subscribe_to_ticks()
            self.running = True
            
            # Start signal processing thread
            signal_thread = threading.Thread(target=self.process_signals)
            signal_thread.daemon = True
            signal_thread.start()
            
            logger.info("Enhanced Coordinator started - processing signals and ticks")
            
            while self.running:
                try:
                    # Receive WebSocket messages
                    message = self.ws.recv()
                    data = json.loads(message)
                    
                    # Handle tick data for pattern analysis
                    if "tick" in data:
                        self.process_tick_data(data["tick"])
                    
                    # Handle other responses
                    elif "buy" in data:
                        self.handle_trade_result(data)
                    
                    time.sleep(0.1)  # Small delay for CPU efficiency
                    
                except WebSocketConnectionClosedException:
                    logger.warning("Enhanced Coordinator WebSocket connection closed")
                    break
                except Exception as e:
                    logger.error(f"Enhanced Coordinator error: {e}")
                    time.sleep(1)
            
        except KeyboardInterrupt:
            logger.info("Enhanced Coordinator stopped by user")
        finally:
            self.cleanup()
    
    def process_tick_data(self, tick_data: Dict):
        """Process incoming tick data for pattern analysis"""
        try:
            self.current_tick_data = tick_data
            quote = float(tick_data["quote"])
            timestamp = tick_data["epoch"]
            
            # Update pattern engine with new tick
            pattern_features = self.pattern_engine.update_patterns(
                quote, 
                int(str(quote).replace('.', '')[-1]),  # last digit
                config.PRIMARY_SYMBOL
            )
            
            # Update online learning models
            if len(self.recent_trades) > 0:
                # Use recent trade outcomes for model updates
                recent_trade = self.recent_trades[-1]
                self.online_learner.add_sample(
                    f"{config.PRIMARY_SYMBOL}_{recent_trade.get('contract_type', 'UNKNOWN')}",
                    pattern_features,
                    1 if recent_trade.get('profit', 0) > 0 else 0,
                    recent_trade.get('probability', 0.5),
                    recent_trade.get('confidence', 0.5)
                )
            
            # Check for drift and update meta-controller
            drift_detected, method, strength = self.meta_controller.drift_detector.detect_drift()
            if drift_detected:
                logger.warning(f"Market drift detected via {method} (strength: {strength:.2f})")
                self.online_learner.handle_drift_detection(method, strength)
            
            logger.debug(f"Processed tick: {quote}, patterns: {len(pattern_features)} features")
            
        except Exception as e:
            logger.error(f"Error processing tick data: {e}")
    
    def process_signals(self):
        """Process signals from probe bots with enhanced intelligence"""
        while self.running:
            try:
                # Get recent signals
                signals = signal_manager.get_recent_signals(
                    max_age=30.0  # 30 seconds
                )
                
                if signals and time.time() - self.last_trade_time >= self.min_signal_interval:
                    decision = self.make_enhanced_trading_decision(signals)
                    
                    if decision:
                        success = self.execute_enhanced_trade(decision)
                        if success:
                            self.last_trade_time = time.time()
                
                time.sleep(2)  # Check for new signals every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in enhanced signal processing: {e}")
                time.sleep(5)
    
    def make_enhanced_trading_decision(self, signals: List[TradingSignal]) -> Optional[Dict]:
        """Enhanced trading decision with pattern-aware intelligence"""
        try:
            if not self.current_tick_data:
                logger.debug("No current tick data for enhanced decision")
                return None
            
            # Group signals by contract type
            signal_groups = defaultdict(list)
            for signal in signals:
                signal_groups[signal.contract_type].append(signal)
            
            best_decision = None
            best_ev = -float('inf')
            
            for contract_type, type_signals in signal_groups.items():
                if len(type_signals) < 1:
                    continue
                
                # Enhanced consensus calculation with calibration
                enhanced_decision = self.calculate_enhanced_consensus(
                    type_signals, contract_type
                )
                
                if enhanced_decision and enhanced_decision['expected_value'] > best_ev:
                    best_ev = enhanced_decision['expected_value']
                    best_decision = enhanced_decision
            
            return best_decision
            
        except Exception as e:
            logger.error(f"Error in enhanced trading decision: {e}")
            return None
    
    def calculate_enhanced_consensus(self, signals: List[TradingSignal], 
                                   contract_type: str) -> Optional[Dict]:
        """Calculate enhanced consensus using pattern-aware features"""
        try:
            # Get current market features
            quote = float(self.current_tick_data["quote"])
            last_digit = int(str(quote).replace('.', '')[-1])
            
            # Get pattern features
            pattern_features = self.pattern_engine.get_current_features(config.PRIMARY_SYMBOL)
            
            # Calculate probe consensus (original method)
            total_weight = 0
            weighted_probability = 0
            weighted_confidence = 0
            weighted_payout = 0
            
            for signal in signals:
                # Weight by probe performance
                probe_perf = self.probe_performances.get(signal.bot_id, {})
                recent_perf = probe_perf.get('recent_performance', 0.5)
                weight = max(0.1, recent_perf)
                
                total_weight += weight
                weighted_probability += signal.probability * weight
                weighted_confidence += signal.confidence * weight
                weighted_payout += signal.expected_payout * weight
            
            if total_weight == 0:
                return None
            
            # Raw consensus metrics
            raw_probability = weighted_probability / total_weight
            raw_confidence = weighted_confidence / total_weight
            avg_payout = weighted_payout / total_weight
            
            # ENHANCED: Calibrate the probability
            model_key = f"{config.PRIMARY_SYMBOL}_{contract_type}"
            calibrated_probability = self.calibrator.calibrate_probability(
                model_key, raw_probability
            )
            
            # ENHANCED: Get pattern score
            pattern_score = pattern_features.get('pattern_confidence', 0.0)
            
            # ENHANCED: Get AI predictor enhancement
            enhanced_prediction = self.enhanced_predictor.predict_enhanced(
                config.PRIMARY_SYMBOL,
                {
                    'quote': quote,
                    'last_digit': last_digit,
                    'contract_type': contract_type,
                    **pattern_features
                }
            )
            
            if enhanced_prediction:
                ai_probability = enhanced_prediction[1]  # confidence
                ai_pattern_score = enhanced_prediction[2].get('pattern_score', 0.0)
                
                # Blend calibrated probe consensus with AI prediction
                final_probability = (
                    0.6 * calibrated_probability +  # 60% probe consensus (calibrated)
                    0.3 * ai_probability +          # 30% AI prediction
                    0.1 * pattern_score             # 10% pure pattern score
                )
                
                combined_pattern_score = max(pattern_score, ai_pattern_score)
            else:
                final_probability = calibrated_probability
                combined_pattern_score = pattern_score
            
            # ENHANCED: Meta-controller decision
            payout_net = (avg_payout - config.INITIAL_STAKE) / config.INITIAL_STAKE
            
            # Get model agreement score
            model_agreement = self.calculate_model_agreement(signals, enhanced_prediction)
            
            should_trade, adjusted_ev, decision_reason = self.meta_controller.make_decision(
                final_probability,
                combined_pattern_score,
                payout_net,
                config.INITIAL_STAKE,
                n_effective=min(100, len(self.recent_trades) * 10),
                model_agreement_score=model_agreement
            )
            
            if not should_trade:
                logger.info(f"Meta-controller rejected trade: {decision_reason}")
                return None
            
            # ENHANCED: Martingale stake calculation
            martingale_stake, stake_reason = self.martingale_system.get_next_stake(
                final_probability,
                payout_net,
                self.current_balance
            )
            
            if martingale_stake <= 0:
                logger.info(f"Martingale system rejected trade: {stake_reason}")
                return None
            
            # Final risk management check
            allowed, risk_reason = self.risk_manager.check_trading_allowed()
            if not allowed:
                logger.info(f"Risk manager rejected trade: {risk_reason}")
                return None
            
            final_stake = min(martingale_stake, self.risk_manager.calculate_position_size(final_probability))
            
            return {
                'contract_type': contract_type,
                'probability': final_probability,
                'calibrated_probability': calibrated_probability,
                'pattern_score': combined_pattern_score,
                'stake': final_stake,
                'expected_payout': avg_payout,
                'expected_value': adjusted_ev,
                'raw_probability': raw_probability,
                'model_agreement': model_agreement,
                'decision_reason': decision_reason,
                'stake_reason': stake_reason,
                'features': pattern_features,
                'signals_count': len(signals)
            }
            
        except Exception as e:
            logger.error(f"Error calculating enhanced consensus: {e}")
            return None
    
    def calculate_model_agreement(self, signals: List[TradingSignal], 
                                ai_prediction: Optional[Tuple]) -> float:
        """Calculate agreement between different models"""
        try:
            if not signals or not ai_prediction:
                return 0.5
            
            # Get probabilities from different sources
            probe_probs = [s.probability for s in signals]
            ai_prob = ai_prediction[1]
            
            # Calculate variance in predictions
            all_probs = probe_probs + [ai_prob]
            variance = np.var(all_probs)
            
            # Convert variance to agreement score (lower variance = higher agreement)
            max_variance = 0.25  # Maximum expected variance
            agreement = max(0.0, 1.0 - (variance / max_variance))
            
            return min(1.0, agreement)
            
        except Exception as e:
            logger.error(f"Error calculating model agreement: {e}")
            return 0.5
    
    def execute_enhanced_trade(self, decision: Dict) -> bool:
        """Execute trade with enhanced logging and tracking"""
        try:
            # Get proposal
            proposal_msg = {
                "proposal": 1,
                "amount": decision['stake'],
                "basis": "stake",
                "contract_type": decision['contract_type'],
                "currency": "USD",
                "symbol": config.PRIMARY_SYMBOL
            }
            
            if decision['contract_type'] in ["DIGITOVER", "DIGITUNDER", "DIGITEVEN", "DIGITODD"]:
                proposal_msg["barrier"] = config.DIGIT_BARRIER
            
            self.ws.send(json.dumps(proposal_msg))
            
            # Wait for proposal response
            for _ in range(10):
                try:
                    response = json.loads(self.ws.recv())
                    if "proposal" in response:
                        proposal = response["proposal"]
                        break
                except:
                    time.sleep(0.1)
            else:
                logger.error("No proposal response received")
                return False
            
            # Execute buy
            buy_msg = {
                "buy": proposal["id"],
                "price": proposal["ask_price"]
            }
            
            self.ws.send(json.dumps(buy_msg))
            
            # Enhanced trade logging
            trade_info = {
                'timestamp': time.time(),
                'contract_type': decision['contract_type'],
                'stake': decision['stake'],
                'probability': decision['probability'],
                'calibrated_probability': decision['calibrated_probability'],
                'pattern_score': decision['pattern_score'],
                'expected_value': decision['expected_value'],
                'model_agreement': decision['model_agreement'],
                'proposal_id': proposal["id"],
                'features': decision['features'],
                'decision_metadata': {
                    'decision_reason': decision['decision_reason'],
                    'stake_reason': decision['stake_reason'],
                    'signals_count': decision['signals_count']
                }
            }
            
            # Store in database for analysis
            self.store_enhanced_trade(trade_info)
            
            logger.info(f"Enhanced trade executed: {decision['contract_type']} "
                       f"${decision['stake']:.2f} P={decision['probability']:.3f} "
                       f"EV=${decision['expected_value']:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing enhanced trade: {e}")
            return False
    
    def store_enhanced_trade(self, trade_info: Dict):
        """Store enhanced trade information in database"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trades (
                    entry_time, symbol, contract_type, stake, 
                    raw_confidence, calibrated_probability, pattern_score,
                    ev_calculated, features, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_info['timestamp'],
                config.PRIMARY_SYMBOL,
                trade_info['contract_type'],
                trade_info['stake'],
                trade_info['probability'],
                trade_info['calibrated_probability'],
                trade_info['pattern_score'],
                trade_info['expected_value'],
                json.dumps(trade_info['features']),
                json.dumps(trade_info['decision_metadata'])
            ))
            
            conn.commit()
            conn.close()
            
            # Also collect calibration data
            self.calibrator.collect_calibration_data(
                f"{config.PRIMARY_SYMBOL}_{trade_info['contract_type']}",
                trade_info['probability'],
                None,  # Will be updated when trade result is known
                config.PRIMARY_SYMBOL,
                trade_info['contract_type']
            )
            
        except Exception as e:
            logger.error(f"Error storing enhanced trade: {e}")
    
    def handle_trade_result(self, result_data: Dict):
        """Handle trade result with enhanced learning"""
        try:
            # Extract result information
            if "buy" in result_data:
                buy_info = result_data["buy"]
                contract_id = buy_info.get("contract_id")
                
                # Wait for contract settlement
                # This would normally be handled by a separate contract monitoring thread
                # For now, we'll update the recent trades and learning systems
                
                profit = buy_info.get("payout", 0) - buy_info.get("buy_price", 0)
                win = profit > 0
                
                # Update recent trades
                trade_result = {
                    'timestamp': time.time(),
                    'contract_id': contract_id,
                    'profit': profit,
                    'win': win
                }
                self.recent_trades.append(trade_result)
                
                # Update martingale system
                self.martingale_system.record_trade_outcome(
                    buy_info.get("buy_price", 0), profit
                )
                
                # Update performance tracking
                self.total_trades += 1
                if win:
                    self.winning_trades += 1
                self.total_profit += profit
                
                logger.info(f"Trade result: {'WIN' if win else 'LOSS'} ${profit:.2f}")
                
        except Exception as e:
            logger.error(f"Error handling trade result: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.running = False
            if self.ws:
                self.ws.close()
            logger.info("Enhanced Coordinator cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def main():
    """Main function to run enhanced coordinator"""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    coordinator = EnhancedTradingCoordinator()
    
    try:
        coordinator.run()
    except KeyboardInterrupt:
        logger.info("Enhanced Coordinator stopped by user")
    except Exception as e:
        logger.error(f"Enhanced Coordinator crashed: {e}")

if __name__ == "__main__":
    main()