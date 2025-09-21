# enhanced_probe_a.py - Enhanced Probe A with Pattern-Aware Intelligence
import time
import json
import logging
import threading
import numpy as np
from typing import Dict, Optional, Tuple
from websocket import create_connection, WebSocketConnectionClosedException

# Original imports
from signal_manager import signal_manager, TradingSignal, TradeResult, PerformanceUpdate
from shared_config import config, BOT_IDS, SIGNAL_TYPES

# Enhanced imports
from pattern_calibration import ProbabilityCalibrator
from pattern_detectors import AdvancedPatternEngine
from enhanced_ai_predictor import EnhancedAIPredictor
from online_learning import AdaptiveLearningSystem

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class EnhancedProbeA:
    def __init__(self):
        self.bot_id = BOT_IDS["PROBE_A"] + "_ENHANCED"  # Distinguish from original
        self.token = config.PROBE_A_TOKEN
        self.ws = None
        self.running = False
        
        # Strategy-specific parameters (enhanced)
        self.strategy = "enhanced_digit_parity"
        self.frequency_window = 100  # Larger window for better pattern detection
        
        # Enhanced AI components
        self.calibrator = ProbabilityCalibrator()
        self.pattern_engine = AdvancedPatternEngine()
        self.enhanced_predictor = EnhancedAIPredictor()
        self.online_learner = AdaptiveLearningSystem()
        
        # Performance tracking (enhanced)
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.recent_trades = []
        self.calibration_samples = []
        
        # Market state tracking
        self.current_tick = None
        self.last_signal_time = 0
        self.tick_history = []
        self.min_signal_interval = 3.0  # Faster than original for demo accounts
        
        logger.info(f"Enhanced Probe A initialized: {self.bot_id}")
        logger.info("Enhanced features: Pattern detection, Calibration, AI prediction")
    
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
                logger.info(f"Enhanced Probe A authorized: {response['authorize']['loginid']}")
                return True
            else:
                logger.error(f"Enhanced Probe A authorization failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Enhanced Probe A connection failed: {e}")
            return False
    
    def subscribe_to_ticks(self):
        """Subscribe to tick stream"""
        try:
            subscribe_msg = {
                "ticks": config.PRIMARY_SYMBOL,
                "subscribe": 1
            }
            self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Enhanced Probe A subscribed to {config.PRIMARY_SYMBOL} ticks")
            
        except Exception as e:
            logger.error(f"Enhanced Probe A tick subscription failed: {e}")
    
    def run(self):
        """Main probe loop with enhanced intelligence"""
        try:
            if not self.connect_deriv():
                logger.error("Enhanced Probe A failed to connect")
                return
            
            self.subscribe_to_ticks()
            self.running = True
            
            logger.info("Enhanced Probe A started - analyzing patterns and generating signals")
            
            while self.running:
                try:
                    # Receive WebSocket messages
                    message = self.ws.recv()
                    data = json.loads(message)
                    
                    # Handle tick data
                    if "tick" in data:
                        self.process_enhanced_tick(data["tick"])
                    
                    # Handle trade results
                    elif "buy" in data:
                        self.handle_trade_result(data)
                    
                    time.sleep(0.05)  # High-frequency processing for demo account
                    
                except WebSocketConnectionClosedException:
                    logger.warning("Enhanced Probe A WebSocket connection closed")
                    break
                except Exception as e:
                    logger.error(f"Enhanced Probe A error: {e}")
                    time.sleep(1)
            
        except KeyboardInterrupt:
            logger.info("Enhanced Probe A stopped by user")
        finally:
            self.cleanup()
    
    def process_enhanced_tick(self, tick_data: Dict):
        """Process tick with enhanced pattern analysis"""
        try:
            self.current_tick = tick_data
            quote = float(tick_data["quote"])
            timestamp = tick_data["epoch"]
            
            # Store tick history
            self.tick_history.append({
                'quote': quote,
                'timestamp': timestamp,
                'last_digit': int(str(quote).replace('.', '')[-1])
            })
            
            # Keep recent history
            if len(self.tick_history) > self.frequency_window:
                self.tick_history = self.tick_history[-self.frequency_window:]
            
            # Update pattern engine
            last_digit = int(str(quote).replace('.', '')[-1])
            pattern_features = self.pattern_engine.update_patterns(
                quote, last_digit, config.PRIMARY_SYMBOL
            )
            
            # Generate enhanced analysis
            if (len(self.tick_history) >= 20 and  # Minimum history
                time.time() - self.last_signal_time >= self.min_signal_interval):
                
                analysis = self.analyze_enhanced_patterns(quote, pattern_features)
                
                if analysis and analysis['should_signal']:
                    self.generate_enhanced_signal(analysis)
                    self.last_signal_time = time.time()
            
        except Exception as e:
            logger.error(f"Error processing enhanced tick: {e}")
    
    def analyze_enhanced_patterns(self, quote: float, pattern_features: Dict) -> Optional[Dict]:
        """Enhanced pattern analysis combining multiple approaches"""
        try:
            last_digit = int(str(quote).replace('.', '')[-1])
            
            # 1. Original digit parity analysis (enhanced)
            parity_analysis = self.analyze_digit_parity_enhanced(quote)
            if not parity_analysis:
                return None
            
            # 2. Pattern engine signals
            pattern_confidence = pattern_features.get('pattern_confidence', 0.0)
            even_signal = pattern_features.get('pattern_even_signal', 0.0)
            odd_signal = pattern_features.get('pattern_odd_signal', 0.0)
            
            # 3. Enhanced AI prediction
            features_dict = {
                'quote': quote,
                'last_digit': last_digit,
                'parity': 'even' if last_digit % 2 == 0 else 'odd',
                **pattern_features,
                **parity_analysis
            }
            
            ai_prediction = self.enhanced_predictor.predict_enhanced(
                config.PRIMARY_SYMBOL, features_dict
            )
            
            # 4. Combine all signals
            combined_analysis = self.combine_enhanced_signals(
                parity_analysis, pattern_features, ai_prediction
            )
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"Error in enhanced pattern analysis: {e}")
            return None
    
    def analyze_digit_parity_enhanced(self, quote: float) -> Optional[Dict]:
        """Enhanced digit parity analysis with more sophisticated logic"""
        try:
            # Extract last digit
            last_digit = int(str(quote).replace('.', '')[-1])
            current_parity = "even" if last_digit % 2 == 0 else "odd"
            
            if len(self.tick_history) < 20:
                return None
            
            # Enhanced frequency analysis
            recent_digits = [t['last_digit'] for t in self.tick_history[-50:]]
            recent_parities = ["even" if d % 2 == 0 else "odd" for d in recent_digits]
            
            even_count = recent_parities.count("even")
            odd_count = recent_parities.count("odd")
            total_count = len(recent_parities)
            
            even_freq = even_count / total_count
            odd_freq = odd_count / total_count
            
            # Enhanced streak detection
            streak_length = 1
            streak_parity = current_parity
            
            for i in range(len(recent_parities) - 2, -1, -1):
                if recent_parities[i] == streak_parity:
                    streak_length += 1
                else:
                    break
            
            # Enhanced bias calculation with momentum
            bias_threshold = 0.15  # More sensitive than original
            momentum_window = 10
            
            if len(recent_parities) >= momentum_window:
                recent_momentum = recent_parities[-momentum_window:]
                momentum_even_freq = recent_momentum.count("even") / len(recent_momentum)
                momentum_bias = abs(momentum_even_freq - 0.5)
            else:
                momentum_bias = 0
            
            # Calculate probabilities with enhanced logic
            base_probability = 0.5
            
            # Frequency bias adjustment
            if abs(even_freq - 0.5) > bias_threshold:
                if even_freq > 0.5:
                    # More evens recently, expect odds (mean reversion)
                    base_probability += (even_freq - 0.5) * 0.5
                    predicted_parity = "odd"
                else:
                    # More odds recently, expect evens
                    base_probability += (0.5 - even_freq) * 0.5
                    predicted_parity = "even"
            else:
                # No strong bias, use momentum or streak
                if streak_length >= 4:
                    # Long streak, expect reversal
                    predicted_parity = "odd" if streak_parity == "even" else "even"
                    base_probability += min(0.15, streak_length * 0.02)
                else:
                    # Short streak, might continue with momentum
                    if momentum_bias > 0.1:
                        predicted_parity = "even" if momentum_even_freq > 0.5 else "odd"
                        base_probability += momentum_bias * 0.3
                    else:
                        return None  # No clear signal
            
            # Enhanced confidence calculation
            confidence_factors = [
                abs(even_freq - 0.5) * 2,  # Frequency deviation
                min(0.2, streak_length * 0.03),  # Streak factor
                momentum_bias * 0.5,  # Momentum factor
                min(0.1, len(self.tick_history) / 1000)  # History depth factor
            ]
            
            raw_confidence = min(0.85, sum(confidence_factors))
            
            # Only signal if confidence is reasonable
            if raw_confidence < 0.25:
                return None
            
            return {
                'predicted_parity': predicted_parity,
                'probability': base_probability,
                'confidence': raw_confidence,
                'even_frequency': even_freq,
                'odd_frequency': odd_freq,
                'streak_length': streak_length,
                'streak_parity': streak_parity,
                'momentum_bias': momentum_bias,
                'reasoning': f"Enhanced parity: {predicted_parity} (freq_bias: {abs(even_freq-0.5):.3f}, "
                           f"streak: {streak_length}{streak_parity[0]}, momentum: {momentum_bias:.3f})"
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced digit parity analysis: {e}")
            return None
    
    def combine_enhanced_signals(self, parity_analysis: Dict, pattern_features: Dict, 
                               ai_prediction: Optional[Tuple]) -> Optional[Dict]:
        """Combine all enhanced signals into final decision"""
        try:
            # Base probability from parity analysis
            base_prob = parity_analysis['probability']
            base_confidence = parity_analysis['confidence']
            predicted_parity = parity_analysis['predicted_parity']
            
            # Pattern engine contributions
            pattern_confidence = pattern_features.get('pattern_confidence', 0.0)
            even_signal = pattern_features.get('pattern_even_signal', 0.0)
            odd_signal = pattern_features.get('pattern_odd_signal', 0.0)
            
            # AI prediction contributions
            ai_prob = 0.5
            ai_confidence = 0.0
            ai_pattern_score = 0.0
            
            if ai_prediction:
                ai_pred, ai_conf, ai_metadata = ai_prediction
                ai_prob = ai_conf
                ai_confidence = ai_conf
                ai_pattern_score = ai_metadata.get('pattern_score', 0.0)
            
            # Weighted combination
            weights = {
                'parity': 0.5,     # 50% original parity analysis
                'pattern': 0.3,    # 30% pattern engine
                'ai': 0.2          # 20% AI prediction
            }
            
            # Combine probabilities
            if predicted_parity == "even":
                pattern_support = even_signal
                final_prob = (
                    weights['parity'] * base_prob +
                    weights['pattern'] * (0.5 + pattern_support) +
                    weights['ai'] * ai_prob
                )
            else:  # odd
                pattern_support = odd_signal
                final_prob = (
                    weights['parity'] * base_prob +
                    weights['pattern'] * (0.5 + pattern_support) +
                    weights['ai'] * ai_prob
                )
            
            # Combine confidence
            final_confidence = (
                weights['parity'] * base_confidence +
                weights['pattern'] * pattern_confidence +
                weights['ai'] * ai_confidence
            )
            
            # Enhanced decision thresholds
            min_probability = 0.55  # Slightly more conservative
            min_confidence = 0.30   # Reasonable confidence threshold
            
            should_signal = (final_prob >= min_probability and 
                           final_confidence >= min_confidence)
            
            # Determine contract type
            if predicted_parity == "even":
                contract_type = "DIGITEVEN"
            else:
                contract_type = "DIGITODD"
            
            return {
                'should_signal': should_signal,
                'contract_type': contract_type,
                'probability': final_prob,
                'confidence': final_confidence,
                'pattern_score': max(pattern_confidence, ai_pattern_score),
                'components': {
                    'parity_prob': base_prob,
                    'parity_conf': base_confidence,
                    'pattern_conf': pattern_confidence,
                    'ai_prob': ai_prob,
                    'ai_conf': ai_confidence
                },
                'reasoning': f"Enhanced {predicted_parity}: P={final_prob:.3f} C={final_confidence:.3f} "
                           f"(parity:{base_confidence:.2f} pattern:{pattern_confidence:.2f} ai:{ai_confidence:.2f})",
                'features': pattern_features
            }
            
        except Exception as e:
            logger.error(f"Error combining enhanced signals: {e}")
            return None
    
    def generate_enhanced_signal(self, analysis: Dict):
        """Generate enhanced trading signal"""
        try:
            # Calibrate the probability
            model_key = f"{config.PRIMARY_SYMBOL}_{analysis['contract_type']}"
            calibrated_probability = self.calibrator.calibrate_probability(
                model_key, analysis['probability']
            )
            
            # Get current proposal for payout calculation
            proposal = self.get_proposal(analysis['contract_type'])
            if not proposal:
                logger.warning("Could not get proposal for enhanced signal")
                return
            
            expected_payout = proposal["payout"]
            stake = config.PROBE_STAKE
            
            # Calculate expected value with calibrated probability
            expected_value = (calibrated_probability * expected_payout) - stake
            
            # Create enhanced signal
            signal = TradingSignal(
                bot_id=self.bot_id,
                signal_type=SIGNAL_TYPES["TRADE_SIGNAL"],
                timestamp=time.time(),
                symbol=config.PRIMARY_SYMBOL,
                contract_type=analysis['contract_type'],
                probability=calibrated_probability,  # Use calibrated probability
                confidence=analysis['confidence'],
                expected_payout=expected_payout,
                stake=stake,
                reasoning=analysis['reasoning'],
                features={
                    **analysis['features'],
                    'raw_probability': analysis['probability'],
                    'calibrated_probability': calibrated_probability,
                    'pattern_score': analysis['pattern_score'],
                    'components': analysis['components'],
                    'enhanced_probe': True
                }
            )
            
            # Publish signal
            signal_manager.publish_signal(signal)
            
            # Execute demo trade for learning
            if expected_value > 0.1:  # Only trade if positive EV
                trade_result = self.execute_demo_trade(analysis['contract_type'], analysis)
                if trade_result:
                    self.update_learning_systems(analysis, trade_result)
            
            logger.info(f"Enhanced signal generated: {analysis['contract_type']} "
                       f"P={calibrated_probability:.3f} (raw:{analysis['probability']:.3f}) "
                       f"C={analysis['confidence']:.3f} EV=${expected_value:.2f}")
            
        except Exception as e:
            logger.error(f"Error generating enhanced signal: {e}")
    
    def get_proposal(self, contract_type: str) -> Optional[Dict]:
        """Get trading proposal"""
        try:
            proposal_msg = {
                "proposal": 1,
                "amount": config.PROBE_STAKE,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "symbol": config.PRIMARY_SYMBOL
            }
            
            if contract_type in ["DIGITOVER", "DIGITUNDER", "DIGITEVEN", "DIGITODD"]:
                proposal_msg["barrier"] = config.DIGIT_BARRIER
            
            self.ws.send(json.dumps(proposal_msg))
            
            # Wait for proposal response
            for _ in range(20):  # Increased timeout
                try:
                    response = json.loads(self.ws.recv())
                    if "proposal" in response:
                        return response["proposal"]
                except:
                    time.sleep(0.05)
            
            logger.warning(f"No proposal response for {contract_type}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting proposal: {e}")
            return None
    
    def execute_demo_trade(self, contract_type: str, analysis: Dict) -> Optional[Dict]:
        """Execute demo trade for learning purposes"""
        try:
            proposal = self.get_proposal(contract_type)
            if not proposal:
                return None
            
            # Execute buy
            buy_msg = {
                "buy": proposal["id"],
                "price": proposal["ask_price"]
            }
            
            self.ws.send(json.dumps(buy_msg))
            
            # Store trade info for later result processing
            trade_info = {
                'timestamp': time.time(),
                'contract_type': contract_type,
                'proposal_id': proposal["id"],
                'stake': proposal["ask_price"],
                'expected_payout': proposal["payout"],
                'probability': analysis['probability'],
                'calibrated_probability': analysis.get('calibrated_probability', analysis['probability']),
                'confidence': analysis['confidence'],
                'analysis': analysis
            }
            
            logger.debug(f"Demo trade executed: {contract_type} ${proposal['ask_price']:.2f}")
            return trade_info
            
        except Exception as e:
            logger.error(f"Error executing demo trade: {e}")
            return None
    
    def update_learning_systems(self, analysis: Dict, trade_result: Dict):
        """Update calibration and online learning systems"""
        try:
            model_key = f"{config.PRIMARY_SYMBOL}_{analysis['contract_type']}"
            
            # For now, we'll update when we get the actual result
            # In practice, this would be called when the contract settles
            
            # Update online learning with features
            features = {
                **analysis['features'],
                'probability': analysis['probability'],
                'confidence': analysis['confidence']
            }
            
            # We'll add the actual outcome later when we get trade results
            # For now, just store the prediction
            self.calibration_samples.append({
                'model_key': model_key,
                'probability': analysis['probability'],
                'features': features,
                'timestamp': time.time(),
                'contract_type': analysis['contract_type']
            })
            
        except Exception as e:
            logger.error(f"Error updating learning systems: {e}")
    
    def handle_trade_result(self, result_data: Dict):
        """Handle trade result for learning updates"""
        try:
            if "buy" in result_data:
                buy_info = result_data["buy"]
                
                # This would normally wait for contract settlement
                # For demo purposes, we'll simulate or wait for actual results
                
                logger.info(f"Trade executed: {buy_info.get('contract_id', 'unknown')}")
                
                # Update performance tracking
                self.total_trades += 1
                
        except Exception as e:
            logger.error(f"Error handling trade result: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.running = False
            if self.ws:
                self.ws.close()
            logger.info("Enhanced Probe A cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def main():
    """Main function to run enhanced probe A"""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    probe = EnhancedProbeA()
    
    try:
        probe.run()
    except KeyboardInterrupt:
        logger.info("Enhanced Probe A stopped by user")
    except Exception as e:
        logger.error(f"Enhanced Probe A crashed: {e}")

if __name__ == "__main__":
    main()

