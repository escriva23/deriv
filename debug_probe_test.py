#!/usr/bin/env python3
# debug_probe_test.py - Test probe bot logic with debug output
import time
import json
import logging
import numpy as np
from websocket import create_connection
from shared_config import config

# Set up debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_probe_a_logic():
    """Test Probe A digit parity analysis with debug output"""
    print("🔍 Testing Probe A Logic (Digit Parity Analysis)")
    print("=" * 60)
    
    # Connect to Deriv
    try:
        url = f"{config.DERIV_WS_URL}?app_id={config.DERIV_APP_ID}"
        ws = create_connection(url)
        
        # Authorize
        auth_msg = {"authorize": config.PROBE_A_TOKEN}
        ws.send(json.dumps(auth_msg))
        response = json.loads(ws.recv())
        
        if "authorize" not in response:
            print("❌ Authorization failed")
            return
            
        print(f"✅ Connected as: {response['authorize']['loginid']}")
        
        # Subscribe to ticks
        subscribe_msg = {
            "ticks": config.PRIMARY_SYMBOL,
            "subscribe": 1
        }
        ws.send(json.dumps(subscribe_msg))
        print(f"📡 Subscribed to {config.PRIMARY_SYMBOL} ticks")
        
        # Collect some ticks for analysis
        digit_history = []
        parity_history = []
        tick_count = 0
        
        print("\n📊 Collecting ticks for analysis...")
        print("Tick | Quote     | Digit | Parity | Analysis")
        print("-" * 50)
        
        while tick_count < 20:  # Collect 20 ticks for testing
            try:
                message = ws.recv()
                data = json.loads(message)
                
                if "tick" in data:
                    tick_count += 1
                    tick = data["tick"]
                    quote = float(tick["quote"])
                    
                    # Extract last digit (PROPERLY FIXED logic)
                    # For quote like 1124.98, we want the last significant digit (8)
                    # Convert to string without unnecessary trailing zeros
                    quote_str = str(quote)
                    # Remove decimal point and get the last digit
                    digits_only = quote_str.replace('.', '')
                    last_digit = int(digits_only[-1])
                    parity = "even" if last_digit % 2 == 0 else "odd"
                    
                    # Update history
                    digit_history.append(last_digit)
                    parity_history.append(parity)
                    
                    # Keep only recent history (same as probe_a.py)
                    if len(digit_history) > 50:
                        digit_history = digit_history[-50:]
                        parity_history = parity_history[-50:]
                    
                    # Analyze (simplified version of probe_a logic)
                    analysis_result = "Waiting..."
                    if len(parity_history) >= 10:
                        even_count = parity_history.count("even")
                        odd_count = parity_history.count("odd")
                        total_count = len(parity_history)
                        
                        even_freq = even_count / total_count
                        odd_freq = odd_count / total_count
                        
                        # Predict underrepresented parity
                        if even_freq < 0.45:
                            prediction = "DIGITEVEN"
                            confidence = (0.5 - even_freq) * 2
                        elif odd_freq < 0.45:
                            prediction = "DIGITODD"  
                            confidence = (0.5 - odd_freq) * 2
                        else:
                            prediction = "NONE"
                            confidence = 0.0
                        
                        analysis_result = f"{prediction} ({confidence:.3f})"
                        
                        # Check if this would generate a signal
                        if confidence > config.MIN_PROBABILITY:
                            analysis_result += " 🚨 SIGNAL!"
                        elif confidence > 0.5:
                            analysis_result += " ⚠️  Close"
                    
                    print(f"{tick_count:4d} | {quote:9.5f} | {last_digit:5d} | {parity:6s} | {analysis_result}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                break
        
        ws.close()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 60)
        
        if len(parity_history) >= 10:
            even_count = parity_history.count("even")
            odd_count = parity_history.count("odd")
            total_count = len(parity_history)
            
            print(f"Total Ticks Analyzed: {total_count}")
            print(f"Even Count: {even_count} ({even_count/total_count:.1%})")
            print(f"Odd Count: {odd_count} ({odd_count/total_count:.1%})")
            print(f"Current Confidence Threshold: {config.MIN_PROBABILITY:.1%}")
            
            # Check if any signals would be generated
            even_freq = even_count / total_count
            odd_freq = odd_count / total_count
            
            max_confidence = 0
            if even_freq < 0.45:
                max_confidence = max(max_confidence, (0.5 - even_freq) * 2)
            if odd_freq < 0.45:
                max_confidence = max(max_confidence, (0.5 - odd_freq) * 2)
            
            print(f"Max Confidence Achieved: {max_confidence:.1%}")
            
            if max_confidence > config.MIN_PROBABILITY:
                print("✅ Bot SHOULD generate signals with this data!")
            else:
                print("❌ Confidence too low - no signals would be generated")
                print(f"💡 Consider lowering MIN_PROBABILITY from {config.MIN_PROBABILITY:.1%} to {max_confidence:.1%}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_probe_a_logic()
