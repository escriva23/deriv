# coordinator.py - Main Coordinator Bot (Real Account)
import time
import json
import logging
import threading
import sqlite3
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
from websocket import create_connection, WebSocketConnectionClosedException
from signal_manager import signal_manager, TradingSignal, TradeResult, PerformanceUpdate
from shared_config import config, BOT_IDS, SIGNAL_TYPES

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class TradingCoordinator:
    def __init__(self):
        self.bot_id = BOT_IDS["COORDINATOR"]
        self.token = config.COORDINATOR_TOKEN
        self.ws = None
        self.running = False
        
        # Signal processing
        self.recent_signals = deque(maxlen=100)
        self.probe_performances = {}
        self.signal_history = defaultdict(list)
        
        # Risk management
        self.current_balance = 0.0
        self.daily_start_balance = 0.0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.last_trade_time = 0
        self.in_cooldown = False
        self.cooldown_until = 0
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.recent_trades = []
        
        # Decision making
        self.winning_probe = None
        self.probe_switch_time = 0
        self.min_trades_for_switch = 10
        
        logger.info(f"Coordinator initialized: {self.bot_id}")
    
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
                logger.info(f"Coordinator authorized: {response['authorize']['loginid']}")
                return True
            else:
                logger.error(f"Coordinator authorization failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Coordinator connection failed: {e}")
            return False
    
    def get_account_balance(self) -> Optional[float]:
        """Get current account balance"""
        try:
            balance_msg = {"balance": 1, "subscribe": 1}
            self.ws.send(json.dumps(balance_msg))
            
            response = json.loads(self.ws.recv())
            
            if "balance" in response:
                balance = float(response["balance"]["balance"])
                self.current_balance = balance
                if self.daily_start_balance == 0:
                    self.daily_start_balance = balance
                return balance
            else:
                logger.error(f"Balance request failed: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return None
    
    def initialize_signal_subscriptions(self):
        """Initialize subscriptions to probe signals"""
        try:
            # Subscribe to signals in separate thread
            signal_thread = threading.Thread(
                target=signal_manager.subscribe_to_signals,
                args=(self.process_signal,),
                daemon=True
            )
            signal_thread.start()
            
            # Subscribe to results in separate thread
            results_thread = threading.Thread(
                target=signal_manager.subscribe_to_results,
                args=(self.process_trade_result,),
                daemon=True
            )
            results_thread.start()
            
            # Subscribe to performance updates
            performance_thread = threading.Thread(
                target=signal_manager.subscribe_to_performance,
                args=(self.process_performance_update,),
                daemon=True
            )
            performance_thread.start()
            
            logger.info("Signal subscriptions initialized")
            
        except Exception as e:
            logger.error(f"Error initializing subscriptions: {e}")
    
    def process_signal(self, signal: TradingSignal):
        """Process incoming signal from probe bots"""
        try:
            # Add to recent signals
            self.recent_signals.append(signal)
            
            # Store in history
            self.signal_history[signal.bot_id].append(signal)
            
            # Keep only recent history
            if len(self.signal_history[signal.bot_id]) > 1000:
                self.signal_history[signal.bot_id] = self.signal_history[signal.bot_id][-1000:]
            
            logger.debug(f"Received signal from {signal.bot_id}: {signal.contract_type} ({signal.confidence:.3f})")
            
            # DEBUG: Log signal processing
            logger.info(f"🔍 Processing signal: {signal.bot_id} → {signal.contract_type} (confidence: {signal.confidence:.3f})")
            
            # Analyze signals and potentially make trading decision
            self.analyze_and_trade()
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    def process_trade_result(self, result: TradeResult):
        """Process trade result from probe bots"""
        try:
            logger.debug(f"Received result from {result.bot_id}: {'WIN' if result.win else 'LOSS'} (${result.profit:.2f})")
            
            # Update probe performance tracking
            if result.bot_id not in self.probe_performances:
                self.probe_performances[result.bot_id] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'total_profit': 0.0,
                    'recent_results': deque(maxlen=config.PERFORMANCE_WINDOW)
                }
            
            perf = self.probe_performances[result.bot_id]
            perf['total_trades'] += 1
            perf['total_profit'] += result.profit
            perf['recent_results'].append(result.win)
            
            if result.win:
                perf['winning_trades'] += 1
            
        except Exception as e:
            logger.error(f"Error processing trade result: {e}")
    
    def process_performance_update(self, performance: PerformanceUpdate):
        """Process performance update from probe bots"""
        try:
            logger.debug(f"Performance update from {performance.bot_id}: {performance.win_rate:.3f} win rate")
            
            # Update our tracking
            if performance.bot_id not in self.probe_performances:
                self.probe_performances[performance.bot_id] = {}
            
            self.probe_performances[performance.bot_id].update({
                'win_rate': performance.win_rate,
                'recent_performance': performance.recent_performance,
                'total_trades': performance.total_trades,
                'avg_profit': performance.avg_profit_per_trade
            })
            
        except Exception as e:
            logger.error(f"Error processing performance update: {e}")
    
    def analyze_and_trade(self):
        """Analyze recent signals and make trading decision"""
        try:
            current_time = time.time()
            
            # Check if in cooldown
            if self.in_cooldown and current_time < self.cooldown_until:
                return
            elif self.in_cooldown and current_time >= self.cooldown_until:
                self.in_cooldown = False
                logger.info("Cooldown period ended")
            
            # Check risk management constraints
            if not self.check_risk_constraints():
                return
            
            # Get fresh signals (< 2.0s old)
            fresh_signals = [s for s in self.recent_signals 
                           if current_time - s.timestamp < config.MAX_SIGNAL_AGE]
            
            if len(fresh_signals) < 1:  # Need at least 1 signal for analysis
                return
            
            # Analyze signal consensus
            decision = self.make_trading_decision(fresh_signals)
            
            if decision:
                # Execute the trade
                self.execute_coordinated_trade(decision)
            
        except Exception as e:
            logger.error(f"Error in analyze_and_trade: {e}")
    
    def make_trading_decision(self, signals: List[TradingSignal]) -> Optional[Dict]:
        """Make trading decision based on probe signals"""
        try:
            # Group signals by contract type and bot
            signal_groups = defaultdict(list)
            for signal in signals:
                signal_groups[signal.contract_type].append(signal)
            
            best_decision = None
            best_ev = 0
            
            for contract_type, type_signals in signal_groups.items():
                if len(type_signals) < 1:
                    continue
                
                # Calculate weighted consensus
                total_weight = 0
                weighted_probability = 0
                weighted_confidence = 0
                weighted_payout = 0
                
                for signal in type_signals:
                    # Weight by probe performance
                    probe_perf = self.probe_performances.get(signal.bot_id, {})
                    recent_perf = probe_perf.get('recent_performance', 0.5)
                    
                    # Higher weight for better performing probes
                    weight = max(0.1, recent_perf)
                    
                    total_weight += weight
                    weighted_probability += signal.probability * weight
                    weighted_confidence += signal.confidence * weight
                    weighted_payout += signal.expected_payout * weight
                
                if total_weight == 0:
                    continue
                
                # Calculate consensus metrics
                avg_probability = weighted_probability / total_weight
                avg_confidence = weighted_confidence / total_weight
                avg_payout = weighted_payout / total_weight
                
                # Calculate position size
                stake = self.calculate_position_size(avg_confidence)
                
                # Calculate expected value
                expected_value = (avg_probability * avg_payout) - stake
                
                # Check if this decision meets our thresholds
                if (avg_probability > config.MIN_PROBABILITY and 
                    expected_value > config.MIN_EV_THRESHOLD and
                    expected_value > best_ev):
                    
                    best_decision = {
                        'contract_type': contract_type,
                        'probability': avg_probability,
                        'confidence': avg_confidence,
                        'expected_payout': avg_payout,
                        'stake': stake,
                        'expected_value': expected_value,
                        'supporting_signals': type_signals,
                        'reasoning': f"Consensus from {len(type_signals)} signals, EV: ${expected_value:.3f}"
                    }
                    best_ev = expected_value
            
            # Additional check: identify the currently winning probe
            self.update_winning_probe()
            
            # If we have a decision, check if it aligns with winning probe
            if best_decision and self.winning_probe:
                # Boost confidence if winning probe agrees
                winning_probe_agrees = any(
                    s.bot_id == self.winning_probe for s in best_decision['supporting_signals']
                )
                
                if winning_probe_agrees:
                    best_decision['confidence'] *= 1.2  # 20% boost
                    best_decision['confidence'] = min(0.95, best_decision['confidence'])
                    best_decision['reasoning'] += f" (Winning probe {self.winning_probe} agrees)"
            
            return best_decision
            
        except Exception as e:
            logger.error(f"Error making trading decision: {e}")
            return None
    
    def update_winning_probe(self):
        """Update which probe is currently performing best"""
        try:
            current_time = time.time()
            
            # Only update every 30 seconds and if we have enough data
            if current_time - self.probe_switch_time < 30:
                return
            
            best_probe = None
            best_performance = 0
            
            for bot_id, perf in self.probe_performances.items():
                if bot_id == self.bot_id:  # Skip coordinator
                    continue
                
                recent_perf = perf.get('recent_performance', 0)
                total_trades = perf.get('total_trades', 0)
                
                # Need minimum trades for consideration
                if total_trades >= self.min_trades_for_switch and recent_perf > best_performance:
                    best_performance = recent_perf
                    best_probe = bot_id
            
            # Switch if new probe is significantly better
            if (best_probe and best_probe != self.winning_probe and 
                best_performance > 0.55):  # At least 55% recent win rate
                
                old_probe = self.winning_probe
                self.winning_probe = best_probe
                self.probe_switch_time = current_time
                
                logger.info(f"Switched winning probe: {old_probe} → {best_probe} ({best_performance:.3f} win rate)")
            
        except Exception as e:
            logger.error(f"Error updating winning probe: {e}")
    
    def calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence and risk management"""
        try:
            # Base stake as percentage of balance
            base_stake = self.current_balance * config.MAX_STAKE_PCT
            
            # Adjust based on confidence (Kelly-like approach)
            confidence_multiplier = max(0.1, min(2.0, confidence * 2))
            adjusted_stake = base_stake * confidence_multiplier
            
            # Apply consecutive loss reduction
            if self.consecutive_losses > 0:
                loss_reduction = 0.8 ** self.consecutive_losses  # Reduce by 20% per loss
                adjusted_stake *= loss_reduction
            
            # Ensure within bounds
            adjusted_stake = max(config.MIN_STAKE, min(config.MAX_STAKE, adjusted_stake))
            
            return adjusted_stake
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return config.MIN_STAKE
    
    def check_risk_constraints(self) -> bool:
        """Check if trading is allowed based on risk constraints"""
        try:
            # Initialize balance if not set
            if self.current_balance == 0.0:
                balance = self.get_account_balance()
                if balance:
                    self.current_balance = balance
                    self.daily_start_balance = balance
                    logger.info(f"💰 Balance initialized: ${balance:.2f}")
                else:
                    logger.error("❌ Could not get account balance")
                    return False
            
            # Update daily P&L
            self.daily_pnl = self.current_balance - self.daily_start_balance
            
            logger.debug(f"🔍 Risk check: Balance=${self.current_balance:.2f}, Daily P&L=${self.daily_pnl:.2f}, Consecutive losses={self.consecutive_losses}")
            
            # Check daily loss limit
            if self.daily_pnl <= -config.DAILY_LOSS_LIMIT_PCT * self.daily_start_balance:
                logger.warning(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
                return False
            
            # Check consecutive losses
            if self.consecutive_losses >= config.MAX_CONSECUTIVE_LOSSES:
                if not self.in_cooldown:
                    self.in_cooldown = True
                    self.cooldown_until = time.time() + (config.COOLDOWN_MINUTES * 60)
                    logger.warning(f"Max consecutive losses reached, entering cooldown for {config.COOLDOWN_MINUTES} minutes")
                return False
            
            # Check minimum time between trades
            if time.time() - self.last_trade_time < 1.0:  # 1 second minimum
                return False
            
            logger.debug("✅ Risk constraints passed")
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk constraints: {e}")
            return False
    
    def execute_coordinated_trade(self, decision: Dict):
        """Execute trade based on coordinated decision"""
        try:
            contract_type = decision['contract_type']
            stake = decision['stake']
            confidence = decision['confidence']
            
            logger.info(f"🎯 COORDINATOR TRADE: {contract_type} ${stake:.2f} (confidence: {confidence:.3f})")
            logger.info(f"Reasoning: {decision['reasoning']}")
            
            # PUBLISH COORDINATOR SIGNAL TO REDIS for monitoring
            coordinator_signal = TradingSignal(
                bot_id=self.bot_id,
                signal_type=SIGNAL_TYPES["TRADE_SIGNAL"],
                timestamp=time.time(),
                symbol=config.PRIMARY_SYMBOL,
                contract_type=contract_type,
                probability=decision.get('probability', 0.5),
                confidence=confidence,
                expected_payout=decision.get('expected_payout', stake * 1.95),
                reasoning=f"COORDINATOR: {decision['reasoning']}"
            )
            
            # Publish to Redis for signal monitor
            signal_manager.publish_signal(coordinator_signal)
            
            # Get proposal
            proposal = self.get_proposal(contract_type, stake)
            if not proposal:
                logger.error("Failed to get proposal")
                return
            
            # Execute trade
            result = self.execute_trade(proposal, decision)
            
            if result:
                # Update tracking
                self.last_trade_time = time.time()
                
                # Monitor contract
                monitor_thread = threading.Thread(
                    target=self.monitor_contract,
                    args=(result,),
                    daemon=True
                )
                monitor_thread.start()
                
                # If we have a winning probe, execute simultaneous demo trade
                if self.winning_probe:
                    self.execute_mirror_trade(decision)
            
        except Exception as e:
            logger.error(f"Error executing coordinated trade: {e}")
    
    def execute_mirror_trade(self, decision: Dict):
        """Execute mirror trade on demo account (simulated)"""
        try:
            # This would normally execute the same trade on the winning probe's demo account
            # For now, we'll just log it
            logger.info(f"📋 MIRROR TRADE: Would execute {decision['contract_type']} on {self.winning_probe}")
            
            # In a full implementation, you could:
            # 1. Send a direct command to the winning probe
            # 2. Execute via a separate demo connection
            # 3. Track the mirror trade performance
            
        except Exception as e:
            logger.error(f"Error executing mirror trade: {e}")
    
    def get_proposal(self, contract_type: str, stake: float) -> Optional[Dict]:
        """Get contract proposal from Deriv"""
        try:
            proposal_msg = {
                "proposal": 1,
                "amount": stake,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "duration": config.CONTRACT_DURATION,
                "duration_unit": config.CONTRACT_DURATION_UNIT,
                "symbol": config.PRIMARY_SYMBOL
            }
            
            # Add barrier for digit contracts
            if contract_type in ["DIGITOVER", "DIGITUNDER"]:
                proposal_msg["barrier"] = config.DIGIT_BARRIER
            
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
    
    def execute_trade(self, proposal: Dict, decision: Dict) -> Optional[Dict]:
        """Execute the actual trade"""
        try:
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
                            "payout": float(buy_result.get("payout", proposal["payout"])),
                            "start_time": time.time(),
                            "decision": decision,
                            "stake": decision["stake"]
                        }
                elif "error" in response:
                    logger.error(f"Buy error: {response['error']}")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None
    
    def monitor_contract(self, contract_info: Dict):
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
            
            # Wait for contract completion
            timeout = start_time + 60
            
            while time.time() < timeout:
                try:
                    response = json.loads(self.ws.recv())
                    
                    if "proposal_open_contract" in response:
                        contract = response["proposal_open_contract"]
                        
                        if contract.get("is_expired") or contract.get("status") in ["won", "lost"]:
                            profit = float(contract.get("profit", 0))
                            win = profit > 0
                            
                            # Update performance
                            self.update_coordinator_performance(profit, win, contract_info)
                            
                            # Log result
                            result_msg = "✅ WIN" if win else "❌ LOSS"
                            logger.info(f"{result_msg}: ${profit:.2f} | Balance: ${self.current_balance:.2f}")
                            
                            return
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error monitoring contract: {e}")
                    break
            
            logger.warning(f"Contract monitoring timeout: {contract_id}")
            
        except Exception as e:
            logger.error(f"Error in contract monitoring: {e}")
    
    def update_coordinator_performance(self, profit: float, win: bool, contract_info: Dict):
        """Update coordinator performance metrics"""
        try:
            self.total_trades += 1
            self.total_profit += profit
            self.current_balance += profit
            
            if win:
                self.winning_trades += 1
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
            
            # Update recent trades
            self.recent_trades.append(win)
            if len(self.recent_trades) > config.PERFORMANCE_WINDOW:
                self.recent_trades = self.recent_trades[-config.PERFORMANCE_WINDOW:]
            
            # Calculate metrics
            win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
            recent_performance = sum(self.recent_trades) / len(self.recent_trades) if self.recent_trades else 0
            
            # Publish performance update
            performance = PerformanceUpdate(
                bot_id=self.bot_id,
                timestamp=time.time(),
                total_trades=self.total_trades,
                winning_trades=self.winning_trades,
                win_rate=win_rate,
                total_profit=self.total_profit,
                avg_profit_per_trade=self.total_profit / self.total_trades if self.total_trades > 0 else 0,
                recent_performance=recent_performance
            )
            
            signal_manager.publish_performance_update(performance)
            
        except Exception as e:
            logger.error(f"Error updating coordinator performance: {e}")
    
    def run(self):
        """Main coordinator loop"""
        logger.info("Starting Trading Coordinator...")
        
        if not self.connect_deriv():
            logger.error("Failed to connect to Deriv")
            return
        
        # Get initial balance
        balance = self.get_account_balance()
        if balance is None:
            logger.error("Failed to get account balance")
            return
        
        logger.info(f"Initial balance: ${balance:.2f}")
        
        # Initialize signal subscriptions
        self.initialize_signal_subscriptions()
        
        self.running = True
        
        try:
            # Main monitoring loop
            while self.running:
                try:
                    # Periodic status updates
                    time.sleep(30)  # Check every 30 seconds
                    
                    if not self.running:
                        break
                    
                    # Update balance
                    self.get_account_balance()
                    
                    # Log status
                    logger.info("=" * 60)
                    logger.info("📊 COORDINATOR STATUS")
                    logger.info(f"Balance: ${self.current_balance:.2f}")
                    logger.info(f"Daily P&L: ${self.daily_pnl:.2f}")
                    logger.info(f"Total Trades: {self.total_trades}")
                    logger.info(f"Win Rate: {self.winning_trades/max(1,self.total_trades):.1%}")
                    logger.info(f"Consecutive Losses: {self.consecutive_losses}")
                    logger.info(f"Winning Probe: {self.winning_probe}")
                    logger.info(f"Recent Signals: {len(self.recent_signals)}")
                    logger.info("=" * 60)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(5)
                    
        except KeyboardInterrupt:
            logger.info("Coordinator stopped by user")
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
        logger.info("Coordinator cleaned up")

def main():
    coordinator = TradingCoordinator()
    coordinator.run()

if __name__ == "__main__":
    main()
