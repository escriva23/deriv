# probe_c.py - Probe C: Momentum-based Model (RISE/FALL)
import time
import json
import logging
import threading
import numpy as np
from typing import Dict, Optional, List
from websocket import create_connection, WebSocketConnectionClosedException
from signal_manager import signal_manager, TradingSignal, TradeResult, PerformanceUpdate
from shared_config import config, BOT_IDS, SIGNAL_TYPES

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class ProbeC:
    def __init__(self):
        self.bot_id = BOT_IDS["PROBE_C"]
        self.token = config.PROBE_C_TOKEN
        self.ws = None
        self.running = False
        
        # Strategy-specific parameters
        self.strategy = "momentum"  # RISE/FALL
        self.momentum_window = 20  # window for momentum analysis
        self.micro_volatility_window = 10  # window for micro-volatility
        self.price_history = []
        self.tick_times = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.recent_trades = []
        
        # Current market state
        self.current_tick = None
        self.last_signal_time = 0
        
        logger.info(f"Probe C initialized: {self.bot_id}")
    
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
                logger.info(f"Probe C authorized: {response['authorize']['loginid']}")
                return True
            else:
                logger.error(f"Probe C authorization failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Probe C connection failed: {e}")
            return False
    
    def subscribe_to_ticks(self):
        """Subscribe to tick stream"""
        try:
            subscribe_msg = {
                "ticks": config.PRIMARY_SYMBOL,
                "subscribe": 1
            }
            self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Probe C subscribed to {config.PRIMARY_SYMBOL} ticks")
            
        except Exception as e:
            logger.error(f"Probe C tick subscription failed: {e}")
    
    def analyze_momentum(self, quote: float, tick_time: float) -> Dict:
        """Analyze momentum and micro-volatility patterns"""
        try:
            # Update history
            self.price_history.append(quote)
            self.tick_times.append(tick_time)
            
            # Keep only recent history
            if len(self.price_history) > self.momentum_window:
                self.price_history = self.price_history[-self.momentum_window:]
                self.tick_times = self.tick_times[-self.momentum_window:]
            
            # Need minimum data for analysis
            if len(self.price_history) < 10:
                return None
            
            # Calculate momentum indicators
            momentum_score = self.calculate_momentum_score()
            micro_volatility = self.calculate_micro_volatility()
            tick_streak = self.calculate_tick_streak()
            price_acceleration = self.calculate_price_acceleration()
            
            # Calculate trend strength
            trend_strength = self.calculate_trend_strength()
            
            # Determine prediction based on momentum
            if momentum_score > 0.1 and trend_strength > 0.3:
                # Strong upward momentum
                prediction = "CALL"
                confidence = min(0.9, 0.5 + abs(momentum_score) * 2)
                probability = 0.5 + min(0.4, abs(momentum_score))
            elif momentum_score < -0.1 and trend_strength > 0.3:
                # Strong downward momentum
                prediction = "PUT"
                confidence = min(0.9, 0.5 + abs(momentum_score) * 2)
                probability = 0.5 + min(0.4, abs(momentum_score))
            else:
                # Weak momentum or sideways movement
                return None
            
            # Adjust confidence based on additional factors
            volatility_factor = min(0.2, micro_volatility * 0.5)
            streak_factor = min(0.15, abs(tick_streak) * 0.05)
            acceleration_factor = min(0.1, abs(price_acceleration) * 0.3)
            
            confidence *= (1 + volatility_factor + streak_factor + acceleration_factor)
            confidence = min(0.95, confidence)
            
            return {
                'prediction': prediction,
                'probability': probability,
                'confidence': confidence,
                'momentum_score': momentum_score,
                'micro_volatility': micro_volatility,
                'tick_streak': tick_streak,
                'trend_strength': trend_strength,
                'price_acceleration': price_acceleration,
                'reasoning': f"Momentum: {momentum_score:.4f}, Trend: {trend_strength:.3f}, Vol: {micro_volatility:.4f}"
            }
            
        except Exception as e:
            logger.error(f"Error in momentum analysis: {e}")
            return None
    
    def calculate_momentum_score(self) -> float:
        """Calculate momentum score based on price changes"""
        if len(self.price_history) < 5:
            return 0.0
        
        # Calculate recent price changes
        recent_changes = []
        for i in range(1, len(self.price_history)):
            change = (self.price_history[i] - self.price_history[i-1]) / self.price_history[i-1]
            recent_changes.append(change)
        
        # Weight recent changes more heavily
        weights = np.exp(np.linspace(0, 1, len(recent_changes)))
        weighted_momentum = np.average(recent_changes, weights=weights)
        
        return weighted_momentum
    
    def calculate_micro_volatility(self) -> float:
        """Calculate micro-volatility from recent price movements"""
        if len(self.price_history) < self.micro_volatility_window:
            return 0.0
        
        recent_prices = self.price_history[-self.micro_volatility_window:]
        
        # Calculate tick-to-tick changes
        changes = []
        for i in range(1, len(recent_prices)):
            change = abs(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
            changes.append(change)
        
        # Return standard deviation of changes
        return np.std(changes) if changes else 0.0
    
    def calculate_tick_streak(self) -> int:
        """Calculate current streak of up/down ticks"""
        if len(self.price_history) < 2:
            return 0
        
        # Determine direction of last tick
        last_direction = 1 if self.price_history[-1] > self.price_history[-2] else -1
        
        streak = 1
        for i in range(len(self.price_history) - 2, 0, -1):
            current_direction = 1 if self.price_history[i] > self.price_history[i-1] else -1
            
            if current_direction == last_direction:
                streak += 1
            else:
                break
        
        return streak * last_direction  # Positive for up streak, negative for down streak
    
    def calculate_price_acceleration(self) -> float:
        """Calculate price acceleration (second derivative)"""
        if len(self.price_history) < 3:
            return 0.0
        
        # Calculate first derivatives (velocities)
        velocities = []
        for i in range(1, len(self.price_history)):
            velocity = self.price_history[i] - self.price_history[i-1]
            velocities.append(velocity)
        
        if len(velocities) < 2:
            return 0.0
        
        # Calculate second derivative (acceleration)
        accelerations = []
        for i in range(1, len(velocities)):
            acceleration = velocities[i] - velocities[i-1]
            accelerations.append(acceleration)
        
        # Return recent average acceleration
        recent_acceleration = np.mean(accelerations[-3:]) if len(accelerations) >= 3 else np.mean(accelerations)
        
        # Normalize by price level
        return recent_acceleration / self.price_history[-1] if self.price_history[-1] != 0 else 0.0
    
    def calculate_trend_strength(self) -> float:
        """Calculate trend strength using linear regression"""
        if len(self.price_history) < 5:
            return 0.0
        
        # Use time indices as x values
        x = np.arange(len(self.price_history))
        y = np.array(self.price_history)
        
        # Calculate linear regression
        try:
            slope, intercept = np.polyfit(x, y, 1)
            
            # Calculate R-squared to measure trend strength
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return max(0, r_squared)  # Ensure non-negative
            
        except Exception:
            return 0.0
    
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
            
            logger.info(f"Probe C Performance: {self.total_trades} trades, {win_rate:.3f} win rate, ${self.total_profit:.2f} profit")
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def run(self):
        """Main trading loop"""
        logger.info("Starting Probe C...")
        
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
                        analysis = self.analyze_momentum(
                            float(tick["quote"]), 
                            float(tick.get("epoch", time.time()))
                        )
                        
                        if analysis and analysis["confidence"] > config.MIN_PROBABILITY:
                            # Avoid too frequent signals
                            current_time = time.time()
                            if current_time - self.last_signal_time > 3.0:  # 3 second minimum interval for momentum
                                
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
            logger.info("Probe C stopped by user")
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
        logger.info("Probe C cleaned up")

def main():
    probe = ProbeC()
    probe.run()

if __name__ == "__main__":
    main()
