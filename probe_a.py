# probe_a.py - Probe A: Digit Parity Model (EVEN/ODD)
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

class ProbeA:
    def __init__(self):
        self.bot_id = BOT_IDS["PROBE_A"]
        self.token = config.PROBE_A_TOKEN
        self.ws = None
        self.running = False
        
        # Strategy-specific parameters
        self.strategy = "digit_parity"  # EVEN/ODD
        self.frequency_window = 50  # rolling window for frequency analysis
        self.digit_history = []
        self.parity_history = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.recent_trades = []  # for recent performance calculation
        
        # Current market state
        self.current_tick = None
        self.last_signal_time = 0
        
        logger.info(f"Probe A initialized: {self.bot_id}")
    
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
                logger.info(f"Probe A authorized: {response['authorize']['loginid']}")
                return True
            else:
                logger.error(f"Probe A authorization failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Probe A connection failed: {e}")
            return False
    
    def subscribe_to_ticks(self):
        """Subscribe to tick stream"""
        try:
            subscribe_msg = {
                "ticks": config.PRIMARY_SYMBOL,
                "subscribe": 1
            }
            self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Probe A subscribed to {config.PRIMARY_SYMBOL} ticks")
            
        except Exception as e:
            logger.error(f"Probe A tick subscription failed: {e}")
    
    def analyze_digit_parity(self, quote: float) -> Dict:
        """Analyze digit parity patterns"""
        try:
            # Extract last digit - PROPERLY FIXED: get the actual last significant digit
            # For quote like 1124.98, we want the last significant digit (8)
            quote_str = str(quote)
            # Remove decimal point and get the last digit
            digits_only = quote_str.replace('.', '')
            last_digit = int(digits_only[-1])
            parity = "even" if last_digit % 2 == 0 else "odd"
            
            # Update history
            self.digit_history.append(last_digit)
            self.parity_history.append(parity)
            
            # Keep only recent history
            if len(self.digit_history) > self.frequency_window:
                self.digit_history = self.digit_history[-self.frequency_window:]
                self.parity_history = self.parity_history[-self.frequency_window:]
            
            # Calculate frequencies
            if len(self.parity_history) < 10:
                return None
            
            even_count = self.parity_history.count("even")
            odd_count = self.parity_history.count("odd")
            total_count = len(self.parity_history)
            
            even_freq = even_count / total_count
            odd_freq = odd_count / total_count
            
            # Calculate bias and confidence
            expected_freq = 0.5
            even_bias = abs(even_freq - expected_freq)
            odd_bias = abs(odd_freq - expected_freq)
            
            # Determine prediction
            if even_bias > odd_bias and even_freq < expected_freq:
                # Even is underrepresented, predict EVEN
                prediction = "DIGITEVEN"
                confidence = min(0.9, 0.5 + even_bias * 2)
                probability = 0.5 + even_bias
            elif odd_bias > even_bias and odd_freq < expected_freq:
                # Odd is underrepresented, predict ODD
                prediction = "DIGITODD"
                confidence = min(0.9, 0.5 + odd_bias * 2)
                probability = 0.5 + odd_bias
            else:
                # No clear bias
                return None
            
            # Additional pattern analysis
            recent_streak = self.calculate_recent_streak()
            volatility_factor = self.calculate_volatility_factor()
            
            # Adjust confidence based on additional factors
            confidence *= (1 + min(0.2, recent_streak * 0.05))
            confidence *= (1 + min(0.1, volatility_factor))
            confidence = min(0.95, confidence)
            
            return {
                'prediction': prediction,
                'probability': probability,
                'confidence': confidence,
                'even_freq': even_freq,
                'odd_freq': odd_freq,
                'recent_streak': recent_streak,
                'volatility_factor': volatility_factor,
                'reasoning': f"Even: {even_freq:.3f}, Odd: {odd_freq:.3f}, Streak: {recent_streak}"
            }
            
        except Exception as e:
            logger.error(f"Error in parity analysis: {e}")
            return None
    
    def calculate_recent_streak(self) -> int:
        """Calculate recent parity streak"""
        if len(self.parity_history) < 2:
            return 0
        
        current_parity = self.parity_history[-1]
        streak = 1
        
        for i in range(len(self.parity_history) - 2, -1, -1):
            if self.parity_history[i] == current_parity:
                streak += 1
            else:
                break
        
        return streak
    
    def calculate_volatility_factor(self) -> float:
        """Calculate volatility factor from digit changes"""
        if len(self.digit_history) < 5:
            return 0.0
        
        recent_digits = self.digit_history[-5:]
        changes = [abs(recent_digits[i] - recent_digits[i-1]) for i in range(1, len(recent_digits))]
        avg_change = np.mean(changes)
        
        # Normalize to 0-1 range
        return min(1.0, avg_change / 5.0)
    
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
                "symbol": config.PRIMARY_SYMBOL
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
            
            logger.info(f"Probe A Performance: {self.total_trades} trades, {win_rate:.3f} win rate, ${self.total_profit:.2f} profit")
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def run(self):
        """Main trading loop"""
        logger.info("Starting Probe A...")
        
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
                        analysis = self.analyze_digit_parity(float(tick["quote"]))
                        
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
            logger.info("Probe A stopped by user")
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
        logger.info("Probe A cleaned up")

def main():
    probe = ProbeA()
    probe.run()

if __name__ == "__main__":
    main()
