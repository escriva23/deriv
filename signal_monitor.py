#!/usr/bin/env python3
# signal_monitor.py - Monitor bot signals in real-time
import redis
import json
import time
from datetime import datetime
from shared_config import config

def monitor_signals():
    """Monitor Redis signals from probe bots"""
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
        print("✅ Connected to Redis successfully")
        print(f"📡 Monitoring channel: {config.SIGNAL_CHANNEL}")
        print("=" * 60)
        
        # Subscribe to signal channel
        pubsub = r.pubsub()
        pubsub.subscribe(config.SIGNAL_CHANNEL)
        
        signal_count = 0
        
        print("🎯 Waiting for bot signals... (Press Ctrl+C to stop)")
        print()
        
        for message in pubsub.listen():
            if message['type'] == 'message':
                signal_count += 1
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                try:
                    # Parse signal data
                    signal_data = json.loads(message['data'])
                    bot_id = signal_data.get('bot_id', 'Unknown')
                    contract_type = signal_data.get('contract_type', 'Unknown')
                    probability = signal_data.get('probability', 0)
                    confidence = signal_data.get('confidence', 0)
                    
                    print(f"[{timestamp}] 🤖 {bot_id}")
                    print(f"   📈 Signal: {contract_type}")
                    print(f"   🎯 Probability: {probability:.3f}")
                    print(f"   💪 Confidence: {confidence:.3f}")
                    print(f"   📊 Total Signals: {signal_count}")
                    print("-" * 40)
                    
                except json.JSONDecodeError:
                    print(f"[{timestamp}] ⚠️  Invalid signal format: {message['data']}")
                except Exception as e:
                    print(f"[{timestamp}] ❌ Error parsing signal: {e}")
                    
    except KeyboardInterrupt:
        print(f"\n🛑 Signal monitoring stopped. Total signals received: {signal_count}")
    except Exception as e:
        print(f"❌ Error connecting to Redis: {e}")
        print("Make sure Redis is running and accessible")

if __name__ == "__main__":
    monitor_signals()
