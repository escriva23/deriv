# trading_executor.py - Real-time trading execution engine
import json
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple
from websocket import create_connection, WebSocketConnectionClosedException
from config import config

logger = logging.getLogger(__name__)

class TradingExecutor:
    def __init__(self, token: str):
        self.token = token
        self.ws = None
        self.authorized = False
        self.pending_proposals = {}
        self.active_contracts = {}
        self.connection_lock = threading.Lock()
        
    def connect(self) -> bool:
        """Establish WebSocket connection and authorize"""
        try:
            with self.connection_lock:
                url = f"{config.DERIV_WS_URL}?app_id={config.DERIV_APP_ID}"
                self.ws = create_connection(url)
                
                # Authorize
                auth_msg = {"authorize": self.token}
                self.ws.send(json.dumps(auth_msg))
                
                response = json.loads(self.ws.recv())
                
                if "authorize" in response and response["authorize"].get("loginid"):
                    self.authorized = True
                    logger.info(f"Authorized successfully: {response['authorize']['loginid']}")
                    return True
                else:
                    logger.error(f"Authorization failed: {response}")
                    return False
                    
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close WebSocket connection"""
        try:
            with self.connection_lock:
                if self.ws:
                    self.ws.close()
                    self.ws = None
                self.authorized = False
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
    
    def get_account_balance(self) -> Optional[float]:
        """Get current account balance"""
        if not self.authorized:
            return None
            
        try:
            balance_msg = {"balance": 1, "subscribe": 1}
            self.ws.send(json.dumps(balance_msg))
            
            response = json.loads(self.ws.recv())
            
            if "balance" in response:
                return float(response["balance"]["balance"])
            else:
                logger.error(f"Balance request failed: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return None
    
    def get_proposal(self, contract_type: str, symbol: str, stake: float, 
                    duration: int = 1, duration_unit: str = "t", 
                    barrier: Optional[str] = None) -> Optional[Dict]:
        """Get contract proposal with pricing"""
        if not self.authorized:
            logger.error("Not authorized")
            return None
            
        try:
            # Build proposal request
            proposal_msg = {
                "proposal": 1,
                "amount": stake,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "duration": duration,
                "duration_unit": duration_unit,
                "symbol": symbol
            }
            
            # Add barrier for digit contracts
            if contract_type in ["DIGITOVER", "DIGITUNDER", "DIGITMATCHES", "DIGITDIFFERS"]:
                if barrier is None:
                    barrier = "5"  # Default barrier
                proposal_msg["barrier"] = barrier
            
            self.ws.send(json.dumps(proposal_msg))
            
            # Wait for proposal response
            max_attempts = 10
            for _ in range(max_attempts):
                response = json.loads(self.ws.recv())
                
                if "proposal" in response:
                    proposal = response["proposal"]
                    
                    # Validate proposal
                    if "id" in proposal and "ask_price" in proposal:
                        return {
                            "id": proposal["id"],
                            "ask_price": float(proposal["ask_price"]),
                            "payout": float(proposal.get("payout", 0)),
                            "spot": float(proposal.get("spot", 0)),
                            "display_value": proposal.get("display_value", ""),
                            "contract_type": contract_type,
                            "symbol": symbol,
                            "stake": stake
                        }
                    else:
                        logger.error(f"Invalid proposal: {proposal}")
                        return None
                        
                elif "error" in response:
                    logger.error(f"Proposal error: {response['error']}")
                    return None
                    
            logger.error("Proposal timeout")
            return None
            
        except Exception as e:
            logger.error(f"Error getting proposal: {e}")
            return None
    
    def buy_contract(self, proposal_id: str, price: float) -> Optional[Dict]:
        """Buy a contract using proposal ID"""
        if not self.authorized:
            logger.error("Not authorized")
            return None
            
        try:
            buy_msg = {
                "buy": proposal_id,
                "price": price
            }
            
            self.ws.send(json.dumps(buy_msg))
            
            # Wait for buy response
            max_attempts = 10
            for _ in range(max_attempts):
                response = json.loads(self.ws.recv())
                
                if "buy" in response:
                    buy_result = response["buy"]
                    
                    if "contract_id" in buy_result:
                        contract_info = {
                            "contract_id": buy_result["contract_id"],
                            "transaction_id": buy_result.get("transaction_id"),
                            "buy_price": float(buy_result.get("buy_price", price)),
                            "payout": float(buy_result.get("payout", 0)),
                            "start_time": int(buy_result.get("start_time", time.time())),
                            "purchase_time": int(time.time())
                        }
                        
                        # Store active contract
                        self.active_contracts[contract_info["contract_id"]] = contract_info
                        
                        logger.info(f"Contract purchased: {contract_info['contract_id']}")
                        return contract_info
                    else:
                        logger.error(f"Buy failed: {buy_result}")
                        return None
                        
                elif "error" in response:
                    logger.error(f"Buy error: {response['error']}")
                    return None
                    
            logger.error("Buy timeout")
            return None
            
        except Exception as e:
            logger.error(f"Error buying contract: {e}")
            return None
    
    def get_contract_status(self, contract_id: str) -> Optional[Dict]:
        """Get current status of a contract"""
        if not self.authorized:
            return None
            
        try:
            status_msg = {
                "proposal_open_contract": 1,
                "contract_id": contract_id,
                "subscribe": 1
            }
            
            self.ws.send(json.dumps(status_msg))
            
            # Wait for status response
            max_attempts = 10
            for _ in range(max_attempts):
                response = json.loads(self.ws.recv())
                
                if "proposal_open_contract" in response:
                    contract = response["proposal_open_contract"]
                    
                    return {
                        "contract_id": contract_id,
                        "is_expired": contract.get("is_expired", False),
                        "is_settleable": contract.get("is_settleable", False),
                        "current_spot": float(contract.get("current_spot", 0)),
                        "entry_spot": float(contract.get("entry_spot", 0)),
                        "exit_tick": contract.get("exit_tick"),
                        "profit": float(contract.get("profit", 0)),
                        "status": contract.get("status", "open")
                    }
                    
                elif "error" in response:
                    logger.error(f"Contract status error: {response['error']}")
                    return None
                    
            return None
            
        except Exception as e:
            logger.error(f"Error getting contract status: {e}")
            return None
    
    def execute_trade(self, contract_type: str, symbol: str, stake: float, 
                     confidence: float, barrier: Optional[str] = None) -> Optional[Dict]:
        """Execute a complete trade: proposal -> buy -> monitor"""
        
        # Reconnect if needed
        if not self.authorized:
            if not self.connect():
                return None
        
        try:
            # Get proposal
            logger.info(f"Getting proposal: {contract_type} {symbol} ${stake}")
            proposal = self.get_proposal(contract_type, symbol, stake, barrier=barrier)
            
            if not proposal:
                logger.error("Failed to get proposal")
                return None
            
            # Validate ask price (basic slippage protection)
            ask_price = proposal["ask_price"]
            if ask_price > stake * 1.1:  # Allow 10% slippage
                logger.warning(f"High ask price: ${ask_price} vs stake ${stake}")
                return None
            
            # Buy contract
            logger.info(f"Buying contract: {proposal['id']} at ${ask_price}")
            contract = self.buy_contract(proposal["id"], ask_price)
            
            if not contract:
                logger.error("Failed to buy contract")
                return None
            
            # Return trade execution result
            return {
                "success": True,
                "contract_id": contract["contract_id"],
                "buy_price": contract["buy_price"],
                "payout": contract["payout"],
                "contract_type": contract_type,
                "symbol": symbol,
                "stake": stake,
                "confidence": confidence,
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "contract_type": contract_type,
                "symbol": symbol,
                "stake": stake,
                "timestamp": int(time.time())
            }
    
    def monitor_contract(self, contract_id: str, timeout: int = 60) -> Optional[Dict]:
        """Monitor a contract until expiry"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                status = self.get_contract_status(contract_id)
                
                if not status:
                    time.sleep(1)
                    continue
                
                # Check if contract is finished
                if status["is_expired"] or status["status"] in ["won", "lost"]:
                    profit = status["profit"]
                    
                    # Remove from active contracts
                    if contract_id in self.active_contracts:
                        del self.active_contracts[contract_id]
                    
                    return {
                        "contract_id": contract_id,
                        "profit": profit,
                        "win": profit > 0,
                        "exit_spot": status.get("current_spot", 0),
                        "entry_spot": status.get("entry_spot", 0),
                        "status": status["status"],
                        "exit_time": int(time.time())
                    }
                
                time.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                logger.error(f"Error monitoring contract {contract_id}: {e}")
                time.sleep(1)
        
        logger.warning(f"Contract monitoring timeout: {contract_id}")
        return None
    
    def execute_and_wait(self, contract_type: str, symbol: str, stake: float, 
                        confidence: float, barrier: Optional[str] = None) -> Optional[Dict]:
        """Execute trade and wait for result"""
        
        # Execute trade
        trade_result = self.execute_trade(contract_type, symbol, stake, confidence, barrier)
        
        if not trade_result or not trade_result["success"]:
            return trade_result
        
        # Monitor until completion
        contract_id = trade_result["contract_id"]
        monitor_result = self.monitor_contract(contract_id)
        
        if monitor_result:
            # Combine execution and monitoring results
            return {
                **trade_result,
                **monitor_result,
                "complete": True
            }
        else:
            return {
                **trade_result,
                "complete": False,
                "error": "Monitoring timeout"
            }
    
    def get_available_symbols(self) -> Optional[List[str]]:
        """Get list of available trading symbols"""
        if not self.authorized:
            return None
            
        try:
            symbols_msg = {"active_symbols": "brief", "product_type": "basic"}
            self.ws.send(json.dumps(symbols_msg))
            
            response = json.loads(self.ws.recv())
            
            if "active_symbols" in response:
                symbols = []
                for symbol_info in response["active_symbols"]:
                    if symbol_info.get("market") == "synthetic_index":
                        symbols.append(symbol_info["symbol"])
                return symbols
            else:
                logger.error(f"Failed to get symbols: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return None
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Unsubscribe from any active subscriptions
            if self.ws and self.authorized:
                for contract_id in list(self.active_contracts.keys()):
                    try:
                        unsub_msg = {
                            "forget": contract_id
                        }
                        self.ws.send(json.dumps(unsub_msg))
                    except:
                        pass
            
            self.disconnect()
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

if __name__ == "__main__":
    # Test the executor (requires demo token)
    import os
    
    token = os.getenv("DERIV_TOKEN")
    if not token:
        print("Please set DERIV_TOKEN environment variable")
        exit(1)
    
    executor = TradingExecutor(token)
    
    try:
        # Connect and test
        if executor.connect():
            print("Connected successfully")
            
            # Get balance
            balance = executor.get_account_balance()
            print(f"Account balance: ${balance}")
            
            # Get available symbols
            symbols = executor.get_available_symbols()
            print(f"Available symbols: {symbols[:5]}...")  # Show first 5
            
            # Test proposal (don't buy)
            proposal = executor.get_proposal("DIGITEVEN", "R_100", 1.0)
            if proposal:
                print(f"Proposal: {proposal}")
            
        else:
            print("Connection failed")
            
    finally:
        executor.cleanup()
