#!/usr/bin/env python3
# bot_status_checker.py - Check individual bot status and connectivity
import os
import sys
import time
import json
import redis
import websocket
from shared_config import config

def test_deriv_connection(token, bot_name):
    """Test if a bot can connect to Deriv API"""
    print(f"\n🔍 Testing {bot_name} connection...")
    
    if not token:
        print(f"❌ {bot_name}: No API token found")
        return False
    
    try:
        # Test WebSocket connection
        url = f"{config.DERIV_WS_URL}?app_id={config.DERIV_APP_ID}"
        print(f"   📡 Connecting to: {url}")
        
        ws = websocket.create_connection(url, timeout=10)
        
        # Test authorization
        auth_msg = {"authorize": token}
        ws.send(json.dumps(auth_msg))
        
        response = json.loads(ws.recv())
        
        if "authorize" in response and response["authorize"].get("loginid"):
            print(f"✅ {bot_name}: Connected successfully")
            print(f"   👤 Account: {response['authorize']['loginid']}")
            print(f"   💰 Balance: ${response['authorize'].get('balance', 'Unknown')}")
            print(f"   🏦 Currency: {response['authorize'].get('currency', 'Unknown')}")
            ws.close()
            return True
        else:
            print(f"❌ {bot_name}: Authorization failed")
            print(f"   📄 Response: {response}")
            ws.close()
            return False
            
    except Exception as e:
        print(f"❌ {bot_name}: Connection failed - {e}")
        return False

def test_redis_connection():
    """Test Redis connection"""
    print("\n🔍 Testing Redis connection...")
    
    try:
        r = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True
        )
        
        r.ping()
        print("✅ Redis: Connected successfully")
        
        # Test publishing
        test_msg = {"test": "message", "timestamp": time.time()}
        r.publish(config.SIGNAL_CHANNEL, json.dumps(test_msg))
        print(f"✅ Redis: Test message published to {config.SIGNAL_CHANNEL}")
        
        return True
        
    except Exception as e:
        print(f"❌ Redis: Connection failed - {e}")
        return False

def main():
    """Run all connectivity tests"""
    print("=" * 60)
    print("🔧 BOT CONNECTIVITY DIAGNOSTIC")
    print("=" * 60)
    
    # Test Redis
    redis_ok = test_redis_connection()
    
    # Test Deriv API connections
    tokens = {
        "Probe A": config.PROBE_A_TOKEN,
        "Probe B": config.PROBE_B_TOKEN,
        "Probe C": config.PROBE_C_TOKEN,
        "Coordinator": config.COORDINATOR_TOKEN
    }
    
    deriv_results = {}
    for bot_name, token in tokens.items():
        deriv_results[bot_name] = test_deriv_connection(token, bot_name)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 CONNECTIVITY SUMMARY")
    print("=" * 60)
    
    print(f"Redis Connection: {'✅ OK' if redis_ok else '❌ FAILED'}")
    
    for bot_name, result in deriv_results.items():
        status = '✅ OK' if result else '❌ FAILED'
        print(f"{bot_name:12}: {status}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if not redis_ok:
        print("   - Check if Redis Docker container is running")
        print("   - Verify Redis port 6379 is accessible")
    
    failed_bots = [name for name, result in deriv_results.items() if not result]
    if failed_bots:
        print("   - Check API tokens are valid and not expired")
        print("   - Verify internet connection")
        print("   - Check if tokens have proper permissions")
        print(f"   - Failed bots: {', '.join(failed_bots)}")
    
    if redis_ok and not failed_bots:
        print("   ✅ All connections OK - bots should be working!")
        print("   - If no signals, check bot logic or add debug logging")

if __name__ == "__main__":
    main()
