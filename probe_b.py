# probe_b.py - Probe B: Over/Under Model (OPPOSITE of Probe A)
import time
import json
import logging
import threading
import numpy as np
from typing import Dict, Optional
from websocket import create_connection, WebSocketConnectionClosedException
from signal_manager import signal_manager, TradingSignal, TradeResult, PerformanceUpdate
from shared_config import config, BOT_IDS, SIGNAL_TYPES

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class ProbeB:
    def __init__(self):
        self.bot_id = BOT_IDS["PROBE_B"]
        self.token = config.PROBE_B_TOKEN
        self.ws = None
        self.running = False
        
        # Strategy-specific parameters
        self.strategy = "digit_over_under"  # OVER/UNDER (opposite of Probe A)
        self.histogram_window = 100  # window for digit histogram analysis
        self.digit_history = []
        self.over_under_history = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.recent_trades = []
        
        # Current market state
        self.current_tick = None
        self.last_signal_time = 0
        
        logger.info(f"Probe B initialized: {self.bot_id}")
    
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
                logger.info(f"Probe B authorized: {response['authorize']['loginid']}")
                return True
            else:
                logger.error(f"Probe B authorization failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Probe B connection failed: {e}")
            return False
    
    def subscribe_to_ticks(self):
        """Subscribe to tick stream"""
        try:
            subscribe_msg = {
                "ticks": config.PRIMARY_SYMBOL,
                "subscribe": 1
            }
            self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Probe B subscribed to {config.PRIMARY_SYMBOL} ticks")
            
        except Exception as e:
            logger.error(f"Probe B tick subscription failed: {e}")
    
    def analyze_digit_over_under(self, quote: float) -> Dict:
        """Analyze digit over/under patterns using histogram and skew"""
        try:
            # Extract last digit - PROPERLY FIXED: get the actual last significant digit
            # For quote like 1124.98, we want the last significant digit (8)
            quote_str = str(quote)
            # Remove decimal point and get the last digit
            digits_only = quote_str.replace('.', '')
            last_digit = int(digits_only[-1])
            over_under = "over" if last_digit > 5 else "under"
            
            # Update history
            self.digit_history.append(last_digit)
            self.over_under_history.append(over_under)
            
            # Keep only recent history
            if len(self.digit_history) > self.histogram_window:
                self.digit_history = self.digit_history[-self.histogram_window:]
                self.over_under_history = self.over_under_history[-self.histogram_window:]
            
            # Need minimum data for analysis
            if len(self.digit_history) < 20:
                return None
            
            # Calculate digit histogram
            digit_counts = np.bincount(self.digit_history, minlength=10)
            digit_frequencies = digit_counts / len(self.digit_history)
            
            # Calculate over/under frequencies
            over_count = sum(1 for d in self.digit_history if d > 5)
            under_count = sum(1 for d in self.digit_history if d < 5)
            equal_count = sum(1 for d in self.digit_history if d == 5)
            
            total_count = len(self.digit_history)
            over_freq = over_count / total_count
            under_freq = under_count / total_count
            equal_freq = equal_count / total_count
            
            # Calculate skewness of digit distribution
            mean_digit = np.mean(self.digit_history)
            skewness = self.calculate_skewness(self.digit_history)
            
            # Expected frequencies (theoretical)
            expected_over = 4/10  # digits 6,7,8,9
            expected_under = 5/10  # digits 0,1,2,3,4
            expected_equal = 1/10  # digit 5
            
            # Calculate biases
            over_bias = abs(over_freq - expected_over)
            under_bias = abs(under_freq - expected_under)
            
            # Determine prediction based on OPPOSITE strategy to Probe A
            # If Probe A would predict EVEN, we predict the opposite tendency
            if over_bias > under_bias and over_freq < expected_over:
                # Over is underrepresented, predict digits > 5
                prediction = "DIGITODD"  # Use ODD as proxy for >5 tendency
                confidence = min(0.9, 0.5 + over_bias * 1.5)
                probability = 0.5 + over_bias
            elif under_bias > over_bias and under_freq < expected_under:
                # Under is underrepresented, predict digits < 5
                prediction = "DIGITEVEN"  # Use EVEN as proxy for <5 tendency
                confidence = min(0.9, 0.5 + under_bias * 1.5)
                probability = 0.5 + under_bias
            else:
                # No clear bias
                return None
            
            # Additional factors
            recent_trend = self.calculate_recent_trend()
            distribution_entropy = self.calculate_entropy(digit_frequencies)
            
            # Adjust confidence based on additional factors
            confidence *= (1 + min(0.15, abs(recent_trend) * 0.1))
            confidence *= (1 + min(0.1, (1 - distribution_entropy) * 0.2))
            confidence = min(0.95, confidence)
            
            return {
                'prediction': prediction,
                'probability': probability,
                'confidence': confidence,
                'over_freq': over_freq,
                'under_freq': under_freq,
                'equal_freq': equal_freq,
                'skewness': skewness,
                'recent_trend': recent_trend,
                'entropy': distribution_entropy,
                'reasoning': f"Over: {over_freq:.3f}, Under: {under_freq:.3f}, Skew: {skewness:.3f}"
            }
            
        except Exception as e:
            logger.error(f"Error in over/under analysis: {e}")
            return None
    
    def calculate_skewness(self, data) -> float:
        """Calculate skewness of digit distribution"""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        skew = np.mean([(x - mean) ** 3 for x in data]) / (std ** 3)
        return skew
    
    def calculate_entropy(self, frequencies) -> float:
        """Calculate entropy of digit distribution"""
        # Add small epsilon to avoid log(0)
        frequencies = frequencies + 1e-10
        entropy = -np.sum(frequencies * np.log2(frequencies))
        
        # Normalize to 0-1 range (max entropy for uniform distribution is log2(10))
        max_entropy = np.log2(10)
        return entropy / max_entropy
    
    def calculate_recent_trend(self) -> float:
        """Calculate recent trend in digit values"""
        if len(self.digit_history) < 10:
            return 0.0
        
        recent_digits = self.digit_history[-10:]
        
        # Simple linear trend
        x = np.arange(len(recent_digits))
        trend = np.polyfit(x, recent_digits, 1)[0]
        
        # Normalize to -1 to 1 range
        return np.clip(trend / 2.0, -1.0, 1.0)
    
    def get_proposal(self, contract_type: str) -> Optional[Dict]:
        """Get contract proposal from Deriv"""
        try:
            proposal_msg = {
                "proposal": 1,
                "amount": config.PROBE_STAKE,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "duration": config.CONTRACT_DURATION,
                "duration_unit": config.CONTRACT_DURATION_UNIT,
                "symbol": config.PRIMARY_SYMBOL,
                "barrier": config.DIGIT_BARRIER
            }
            
            self.ws.send(json.dumps(proposal_msg))
            
            # Wait for proposal response
            for _ in range(10):
                response = json.loads(self.ws.recv())
                
                if "proposal" in response:
                    proposal = response["proposal"]
                    if "ask_price" in proposal and "payout" in proposal:
                        return {
                            "id": proposal["id"],
                            "ask_price": float(proposal["ask_price"]),
                            "payout": float(proposal["payout"]),
                            "display_value": proposal.get("display_value", "")
                        }
                elif "error" in response:
                    logger.error(f"Proposal error: {response['error']}")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting proposal: {e}")
            return None
    
    def execute_trade(self, contract_type: str, analysis: Dict) -> Optional[Dict]:
        """Execute trade based on analysis"""
        try:
            # Get proposal
            proposal = self.get_proposal(contract_type)
            if not proposal:
                return None
            
            # Calculate expected value
            expected_payout = proposal["payout"]
            probability = analysis["probability"]
            stake = config.PROBE_STAKE
            
            expected_value = (probability * expected_payout) - stake
            
            # Publish signal before trading
            signal = TradingSignal(
                bot_id=self.bot_id,
                signal_type=SIGNAL_TYPES["TRADE_SIGNAL"],
                timestamp=time.time(),
                symbol=config.PRIMARY_SYMBOL,
                contract_type=contract_type,
                probability=probability,
                confidence=analysis["confidence"],
                expected_payout=expected_payout,
                stake=stake,
                barrier=config.DIGIT_BARRIER,
                reasoning=analysis["reasoning"],
                features=analysis
            )
            
            signal_manager.publish_signal(signal)
            
            # Execute trade
            buy_msg = {
                "buy": proposal["id"],
                "price": proposal["ask_price"]
            }
            
            self.ws.send(json.dumps(buy_msg))
            
            # Wait for buy response
            for _ in range(10):
                response = json.loads(self.ws.recv())
                
                if "buy" in response:
                    buy_result = response["buy"]
                    if "contract_id" in buy_result:
                        return {
                            "contract_id": buy_result["contract_id"],
                            "buy_price": float(buy_result.get("buy_price", proposal["ask_price"])),
                            "payout": float(buy_result.get("payout", expected_payout)),
                            "start_time": time.time(),
                            "signal": signal
                        }
                elif "error" in response:
                    logger.error(f"Buy error: {response['error']}")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None
    
    def monitor_contract(self, contract_info: Dict) -> Optional[Dict]:
        """Monitor contract until completion"""
        try:
            contract_id = contract_info["contract_id"]
            start_time = contract_info["start_time"]
            
            # Subscribe to contract updates
            status_msg = {
                "proposal_open_contract": 1,
                "contract_id": contract_id,
                "subscribe": 1
            }
            
            self.ws.send(json.dumps(status_msg))
            
            # Wait for contract completion (max 60 seconds)
            timeout = start_time + 60
            
            while time.time() < timeout:
                try:
                    response = json.loads(self.ws.recv())
                    
                    if "proposal_open_contract" in response:
                        contract = response["proposal_open_contract"]
                        
                        if contract.get("is_expired") or contract.get("status") in ["won", "lost"]:
                            profit = float(contract.get("profit", 0))
                            win = profit > 0
                            
                            # Create trade result
                            result = TradeResult(
                                bot_id=self.bot_id,
                                signal_id=f"{self.bot_id}_{int(contract_info['start_time'] * 1000)}",
                                timestamp=time.time(),
                                contract_id=contract_id,
                                symbol=config.PRIMARY_SYMBOL,
                                contract_type=contract_info["signal"].contract_type,
                                stake=config.PROBE_STAKE,
                                profit=profit,
                                win=win,
                                entry_spot=float(contract.get("entry_spot", 0)),
                                exit_spot=float(contract.get("current_spot", 0)),
                                execution_time=time.time() - start_time
                            )
                            
                            # Update performance
                            self.update_performance(result)
                            
                            # Publish result
                            signal_manager.publish_trade_result(result)
                            
                            return result
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error monitoring contract: {e}")
                    break
            
            logger.warning(f"Contract monitoring timeout: {contract_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error in contract monitoring: {e}")
            return None
    
    def update_performance(self, result: TradeResult):
        """Update performance metrics"""
        try:
            self.total_trades += 1
            self.total_profit += result.profit
            
            if result.win:
                self.winning_trades += 1
            
            # Update recent trades for recent performance
            self.recent_trades.append(result.win)
            if len(self.recent_trades) > config.PERFORMANCE_WINDOW:
                self.recent_trades = self.recent_trades[-config.PERFORMANCE_WINDOW:]
            
            # Calculate metrics
            win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
            avg_profit = self.total_profit / self.total_trades if self.total_trades > 0 else 0
            recent_performance = sum(self.recent_trades) / len(self.recent_trades) if self.recent_trades else 0
            
            # Publish performance update
            performance = PerformanceUpdate(
                bot_id=self.bot_id,
                timestamp=time.time(),
                total_trades=self.total_trades,
                winning_trades=self.winning_trades,
                win_rate=win_rate,
                total_profit=self.total_profit,
                avg_profit_per_trade=avg_profit,
                recent_performance=recent_performance
            )
            
            signal_manager.publish_performance_update(performance)
            
            logger.info(f"Probe B Performance: {self.total_trades} trades, {win_rate:.3f} win rate, ${self.total_profit:.2f} profit")
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def run(self):
        """Main trading loop"""
        logger.info("Starting Probe B...")
        
        if not self.connect_deriv():
            logger.error("Failed to connect to Deriv")
            return
        
        self.subscribe_to_ticks()
        self.running = True
        
        try:
            while self.running:
                try:
                    message = self.ws.recv()
                    data = json.loads(message)
                    
                    if "tick" in data:
                        tick = data["tick"]
                        self.current_tick = tick
                        
                        # Analyze current market state
                        analysis = self.analyze_digit_over_under(float(tick["quote"]))
                        
                        if analysis and analysis["confidence"] > config.MIN_PROBABILITY:
                            # Avoid too frequent signals
                            current_time = time.time()
                            if current_time - self.last_signal_time > 2.0:  # 2 second minimum interval
                                
                                # Execute trade
                                contract_info = self.execute_trade(analysis["prediction"], analysis)
                                
                                if contract_info:
                                    # Monitor contract in separate thread
                                    monitor_thread = threading.Thread(
                                        target=self.monitor_contract,
                                        args=(contract_info,),
                                        daemon=True
                                    )
                                    monitor_thread.start()
                                    
                                    self.last_signal_time = current_time
                    
                except WebSocketConnectionClosedException:
                    logger.warning("WebSocket connection lost, reconnecting...")
                    if not self.connect_deriv():
                        break
                    self.subscribe_to_ticks()
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            logger.info("Probe B stopped by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
        logger.info("Probe B cleaned up")

def main():
    probe = ProbeB()
    probe.run()

if __name__ == "__main__":
    main()
