#!/usr/bin/env python3
# coordinator_live_debug.py - Real-time coordinator debugging
import redis
import json
import time
import logging
from datetime import datetime
from shared_config import config

# Set up logging to see coordinator activity
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def monitor_coordinator_activity():
    """Monitor coordinator's real-time activity"""
    print("🔍 LIVE COORDINATOR DEBUG MONITOR")
    print("=" * 50)
    
    try:
        # Connect to Redis
        r = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True
        )
        
        # Test connection
        r.ping()
        print("✅ Redis connected")
        
        # Subscribe to the signal channel
        pubsub = r.pubsub()
        pubsub.subscribe(config.SIGNAL_CHANNEL)
        
        print(f"📡 Listening to channel: {config.SIGNAL_CHANNEL}")
        print("🎯 Waiting for signals... (Press Ctrl+C to stop)")
        print()
        
        signal_count = 0
        coordinator_count = 0
        
        for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    signal_data = json.loads(message['data'])
                    signal_count += 1
                    
                    bot_id = signal_data.get('bot_id', 'Unknown')
                    contract_type = signal_data.get('contract_type', 'Unknown')
                    confidence = signal_data.get('confidence', 0)
                    timestamp = signal_data.get('timestamp', 0)
                    
                    # Format timestamp
                    dt = datetime.fromtimestamp(timestamp)
                    time_str = dt.strftime("%H:%M:%S")
                    
                    print(f"[{time_str}] 🤖 {bot_id}")
                    print(f"   📈 Signal: {contract_type}")
                    print(f"   💪 Confidence: {confidence:.3f}")
                    print(f"   📊 Total Signals: {signal_count}")
                    
                    if 'coordinator' in bot_id.lower():
                        coordinator_count += 1
                        print(f"   🎯 COORDINATOR SIGNAL #{coordinator_count}")
                    
                    print("-" * 40)
                    
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON: {message['data']}")
                except Exception as e:
                    print(f"❌ Error processing message: {e}")
                    
    except KeyboardInterrupt:
        print(f"\n📊 Final Stats:")
        print(f"   Total Signals: {signal_count}")
        print(f"   Coordinator Signals: {coordinator_count}")
        print("👋 Monitor stopped")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    monitor_coordinator_activity()
