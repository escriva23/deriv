#!/usr/bin/env python3
# coordinator_debug.py - Debug coordinator decision-making process
import redis
import json
import time
import logging
from datetime import datetime, timedelta
from shared_config import config
from websocket import create_connection

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CoordinatorDebugger:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True
        )
        
        self.probe_performance = {
            'probe_a_parity': {'wins': 0, 'losses': 0, 'total': 0},
            'probe_b_opposite': {'wins': 0, 'losses': 0, 'total': 0},
            'probe_c_momentum': {'wins': 0, 'losses': 0, 'total': 0}
        }
        
        self.recent_signals = []
        
    def test_deriv_connection(self):
        """Test coordinator's Deriv API connection"""
        print("\n🔍 Testing Coordinator Deriv Connection...")
        
        try:
            url = f"{config.DERIV_WS_URL}?app_id={config.DERIV_APP_ID}"
            ws = create_connection(url, timeout=10)
            
            # Test authorization
            auth_msg = {"authorize": config.COORDINATOR_TOKEN}
            ws.send(json.dumps(auth_msg))
            
            response = json.loads(ws.recv())
            
            if "authorize" in response and response["authorize"].get("loginid"):
                print(f"✅ Coordinator connected: {response['authorize']['loginid']}")
                print(f"   💰 Balance: ${response['authorize'].get('balance', 'Unknown')}")
                print(f"   🏦 Currency: {response['authorize'].get('currency', 'Unknown')}")
                ws.close()
                return True
            else:
                print(f"❌ Coordinator authorization failed: {response}")
                ws.close()
                return False
                
        except Exception as e:
            print(f"❌ Coordinator connection failed: {e}")
            return False
    
    def get_contract_proposal(self, contract_type, symbol="R_100"):
        """Test getting contract proposal (same as coordinator would do)"""
        print(f"\n🔍 Testing Contract Proposal: {contract_type}")
        
        try:
            url = f"{config.DERIV_WS_URL}?app_id={config.DERIV_APP_ID}"
            ws = create_connection(url, timeout=10)
            
            # Authorize
            auth_msg = {"authorize": config.COORDINATOR_TOKEN}
            ws.send(json.dumps(auth_msg))
            auth_response = json.loads(ws.recv())
            
            if "authorize" not in auth_response:
                print("❌ Authorization failed for proposal test")
                return None
            
            # Get proposal
            proposal_msg = {
                "proposal": 1,
                "amount": 2.0,  # Test with $2 stake
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "duration": 1,
                "duration_unit": "t",
                "symbol": symbol
            }
            
            ws.send(json.dumps(proposal_msg))
            
            # Wait for proposal response
            for _ in range(10):
                response = json.loads(ws.recv())
                
                if "proposal" in response:
                    proposal = response["proposal"]
                    if "ask_price" in proposal and "payout" in proposal:
                        ask_price = proposal["ask_price"]
                        payout = proposal["payout"]
                        ev = payout - ask_price
                        
                        print(f"✅ Proposal received:")
                        print(f"   💰 Stake: ${ask_price}")
                        print(f"   🎯 Payout: ${payout}")
                        print(f"   📊 EV: ${ev:.3f}")
                        print(f"   ✅ Above threshold: {ev > config.MIN_EV_THRESHOLD}")
                        
                        ws.close()
                        return {
                            'ask_price': ask_price,
                            'payout': payout,
                            'ev': ev,
                            'tradeable': ev > config.MIN_EV_THRESHOLD
                        }
                
                if "error" in response:
                    print(f"❌ Proposal error: {response['error']}")
                    break
            
            ws.close()
            print("❌ No valid proposal received")
            return None
            
        except Exception as e:
            print(f"❌ Proposal test failed: {e}")
            return None
    
    def analyze_recent_signals(self):
        """Analyze recent probe signals from Redis"""
        print("\n🔍 Analyzing Recent Probe Signals...")
        
        try:
            # Get recent signals (if stored in Redis)
            signal_keys = self.redis_client.keys("signal:*")
            
            if not signal_keys:
                print("❌ No stored signals found in Redis")
                print("💡 Signals might be published but not stored")
                return
            
            print(f"✅ Found {len(signal_keys)} stored signals")
            
            for key in signal_keys[-10:]:  # Last 10 signals
                signal_data = self.redis_client.get(key)
                if signal_data:
                    try:
                        signal = json.loads(signal_data)
                        timestamp = datetime.fromtimestamp(signal.get('timestamp', 0))
                        age = (datetime.now() - timestamp).total_seconds()
                        
                        print(f"📊 Signal: {signal.get('bot_id', 'Unknown')}")
                        print(f"   🎯 Type: {signal.get('contract_type', 'Unknown')}")
                        print(f"   💪 Confidence: {signal.get('confidence', 0):.3f}")
                        print(f"   ⏰ Age: {age:.1f}s")
                        print(f"   ✅ Fresh: {age <= config.MAX_SIGNAL_AGE}")
                        print()
                        
                    except json.JSONDecodeError:
                        print(f"❌ Invalid signal data in {key}")
                        
        except Exception as e:
            print(f"❌ Signal analysis failed: {e}")
    
    def simulate_coordinator_decision(self, signal_data):
        """Simulate coordinator decision-making process"""
        print(f"\n🧠 Simulating Coordinator Decision for Signal:")
        print(f"   🤖 Bot: {signal_data.get('bot_id', 'Unknown')}")
        print(f"   📈 Signal: {signal_data.get('contract_type', 'Unknown')}")
        print(f"   💪 Confidence: {signal_data.get('confidence', 0):.3f}")
        print(f"   🎯 Probability: {signal_data.get('probability', 0):.3f}")
        
        # Step 1: Confidence check
        confidence = signal_data.get('confidence', 0)
        if confidence < config.MIN_PROBABILITY:
            print(f"❌ REJECTED: Confidence {confidence:.3f} < {config.MIN_PROBABILITY}")
            return False
        print(f"✅ Confidence check passed: {confidence:.3f} >= {config.MIN_PROBABILITY}")
        
        # Step 2: Get proposal and calculate EV
        contract_type = signal_data.get('contract_type', '')
        proposal = self.get_contract_proposal(contract_type)
        
        if not proposal:
            print("❌ REJECTED: Could not get contract proposal")
            return False
        
        # Step 3: EV check
        ev = proposal['ev']
        if ev <= config.MIN_EV_THRESHOLD:
            print(f"❌ REJECTED: EV {ev:.3f} <= {config.MIN_EV_THRESHOLD}")
            return False
        print(f"✅ EV check passed: {ev:.3f} > {config.MIN_EV_THRESHOLD}")
        
        # Step 4: Risk management
        balance = 11053.47  # Approximate balance
        stake = min(proposal['ask_price'], balance * config.MAX_STAKE_PCT)
        
        print(f"✅ TRADE APPROVED:")
        print(f"   💰 Stake: ${stake:.2f}")
        print(f"   🎯 Expected Payout: ${proposal['payout']:.2f}")
        print(f"   📊 Expected Value: ${ev:.3f}")
        
        return True
    
    def run_full_diagnostic(self):
        """Run complete coordinator diagnostic"""
        print("=" * 60)
        print("🔧 COORDINATOR DIAGNOSTIC TOOL")
        print("=" * 60)
        
        # Test 1: Redis connection
        try:
            self.redis_client.ping()
            print("✅ Redis connection: OK")
        except Exception as e:
            print(f"❌ Redis connection: FAILED - {e}")
            return
        
        # Test 2: Deriv API connection
        deriv_ok = self.test_deriv_connection()
        
        # Test 3: Contract proposals
        if deriv_ok:
            print("\n🔍 Testing Contract Proposals...")
            test_contracts = ["DIGITEVEN", "DIGITODD", "DIGITOVER", "DIGITUNDER", "CALL", "PUT"]
            
            for contract in test_contracts:
                proposal = self.get_contract_proposal(contract)
                if proposal and proposal['tradeable']:
                    print(f"✅ {contract}: Tradeable (EV: ${proposal['ev']:.3f})")
                else:
                    print(f"❌ {contract}: Not tradeable")
        
        # Test 4: Recent signals analysis
        self.analyze_recent_signals()
        
        # Test 5: Simulate decisions on recent probe signals
        print("\n🧠 SIMULATING COORDINATOR DECISIONS...")
        
        # Simulate with your recent signals
        test_signals = [
            {
                'bot_id': 'probe_a_parity',
                'contract_type': 'DIGITEVEN',
                'confidence': 0.799,
                'probability': 0.580,
                'timestamp': time.time()
            },
            {
                'bot_id': 'probe_b_opposite', 
                'contract_type': 'DIGITUNDER',
                'confidence': 0.629,
                'probability': 0.580,
                'timestamp': time.time()
            }
        ]
        
        for signal in test_signals:
            self.simulate_coordinator_decision(signal)
        
        print("\n" + "=" * 60)
        print("📊 DIAGNOSTIC COMPLETE")
        print("=" * 60)

def main():
    debugger = CoordinatorDebugger()
    debugger.run_full_diagnostic()

if __name__ == "__main__":
    main()
