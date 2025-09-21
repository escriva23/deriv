#!/usr/bin/env python3
# data_downloader.py - Download historical tick data for AI training
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from websocket import create_connection
from shared_config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TickDataDownloader:
    def __init__(self):
        self.ws = None
        self.tick_data = []
        self.digit_data = []
        
    def connect(self):
        """Connect to Deriv WebSocket"""
        try:
            url = f"{config.DERIV_WS_URL}?app_id={config.DERIV_APP_ID}"
            self.ws = create_connection(url, timeout=10)
            logger.info("✅ Connected to Deriv WebSocket")
            return True
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    def download_historical_ticks(self, symbol="R_100", count=10000):
        """Download historical tick data"""
        logger.info(f"📥 Downloading {count} historical ticks for {symbol}...")
        
        try:
            # Request historical ticks
            request = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": count,
                "end": "latest",
                "start": 1,
                "style": "ticks"
            }
            
            self.ws.send(json.dumps(request))
            response = json.loads(self.ws.recv())
            
            if "history" in response:
                history = response["history"]
                times = history["times"]
                prices = history["prices"]
                
                logger.info(f"✅ Downloaded {len(times)} ticks")
                
                # Convert to DataFrame
                df = pd.DataFrame({
                    'timestamp': times,
                    'price': prices,
                    'datetime': pd.to_datetime(times, unit='s')
                })
                
                # Extract last digits
                df['last_digit'] = df['price'].apply(lambda x: int(str(x).replace('.', '')[-1]))
                
                # Save to CSV
                filename = f"historical_ticks_{symbol}_{count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(filename, index=False)
                logger.info(f"💾 Saved to {filename}")
                
                return df
            else:
                logger.error(f"❌ No history data: {response}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return None
    
    def download_live_ticks(self, symbol="R_100", duration_minutes=60):
        """Download live tick data for specified duration"""
        logger.info(f"📡 Collecting live ticks for {duration_minutes} minutes...")
        
        try:
            # Subscribe to live ticks
            request = {
                "ticks": symbol,
                "subscribe": 1
            }
            
            self.ws.send(json.dumps(request))
            
            start_time = time.time()
            end_time = start_time + (duration_minutes * 60)
            
            ticks = []
            
            while time.time() < end_time:
                try:
                    response = json.loads(self.ws.recv())
                    
                    if "tick" in response:
                        tick = response["tick"]
                        
                        tick_data = {
                            'timestamp': tick['epoch'],
                            'price': tick['quote'],
                            'datetime': datetime.fromtimestamp(tick['epoch']),
                            'last_digit': int(str(tick['quote']).replace('.', '')[-1])
                        }
                        
                        ticks.append(tick_data)
                        
                        if len(ticks) % 100 == 0:
                            logger.info(f"📊 Collected {len(ticks)} live ticks...")
                            
                except Exception as e:
                    logger.error(f"❌ Error receiving tick: {e}")
                    continue
            
            # Convert to DataFrame
            df = pd.DataFrame(ticks)
            
            # Save to CSV
            filename = f"live_ticks_{symbol}_{duration_minutes}min_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"💾 Saved {len(ticks)} live ticks to {filename}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Live collection failed: {e}")
            return None
    
    def analyze_digit_patterns(self, df):
        """Analyze digit patterns in the data"""
        if df is None or df.empty:
            return None
            
        logger.info("🔍 Analyzing digit patterns...")
        
        analysis = {
            'total_ticks': len(df),
            'digit_counts': df['last_digit'].value_counts().sort_index(),
            'digit_frequencies': df['last_digit'].value_counts(normalize=True).sort_index(),
            'even_odd_ratio': {
                'even': len(df[df['last_digit'] % 2 == 0]) / len(df),
                'odd': len(df[df['last_digit'] % 2 == 1]) / len(df)
            },
            'over_under_ratio': {
                'over_5': len(df[df['last_digit'] > 5]) / len(df),
                'under_5': len(df[df['last_digit'] < 5]) / len(df),
                'equal_5': len(df[df['last_digit'] == 5]) / len(df)
            }
        }
        
        print("\n📊 DIGIT PATTERN ANALYSIS")
        print("=" * 50)
        print(f"Total Ticks: {analysis['total_ticks']:,}")
        print("\nDigit Frequencies:")
        for digit in range(10):
            count = analysis['digit_counts'].get(digit, 0)
            freq = analysis['digit_frequencies'].get(digit, 0)
            print(f"  Digit {digit}: {count:,} ({freq:.3%})")
        
        print(f"\nEven/Odd Distribution:")
        print(f"  Even: {analysis['even_odd_ratio']['even']:.3%}")
        print(f"  Odd:  {analysis['even_odd_ratio']['odd']:.3%}")
        
        print(f"\nOver/Under 5 Distribution:")
        print(f"  Over 5:  {analysis['over_under_ratio']['over_5']:.3%}")
        print(f"  Under 5: {analysis['over_under_ratio']['under_5']:.3%}")
        print(f"  Equal 5: {analysis['over_under_ratio']['equal_5']:.3%}")
        
        return analysis
    
    def close(self):
        """Close WebSocket connection"""
        if self.ws:
            self.ws.close()
            logger.info("🔌 WebSocket closed")

def main():
    """Main function to download and analyze data"""
    downloader = TickDataDownloader()
    
    if not downloader.connect():
        return
    
    try:
        print("📥 DATA DOWNLOAD OPTIONS:")
        print("1. Download 10,000 historical ticks")
        print("2. Download 50,000 historical ticks") 
        print("3. Collect 30 minutes of live ticks")
        print("4. Collect 60 minutes of live ticks")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            df = downloader.download_historical_ticks(count=10000)
        elif choice == "2":
            df = downloader.download_historical_ticks(count=50000)
        elif choice == "3":
            df = downloader.download_live_ticks(duration_minutes=30)
        elif choice == "4":
            df = downloader.download_live_ticks(duration_minutes=60)
        else:
            print("❌ Invalid choice")
            return
        
        if df is not None:
            # Analyze patterns
            analysis = downloader.analyze_digit_patterns(df)
            
            # Save analysis
            analysis_filename = f"digit_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(analysis_filename, 'w') as f:
                # Convert numpy types to regular Python types for JSON serialization
                json_analysis = {
                    'total_ticks': int(analysis['total_ticks']),
                    'digit_counts': {str(k): int(v) for k, v in analysis['digit_counts'].items()},
                    'digit_frequencies': {str(k): float(v) for k, v in analysis['digit_frequencies'].items()},
                    'even_odd_ratio': analysis['even_odd_ratio'],
                    'over_under_ratio': analysis['over_under_ratio']
                }
                json.dump(json_analysis, f, indent=2)
            
            logger.info(f"💾 Analysis saved to {analysis_filename}")
            
    finally:
        downloader.close()

if __name__ == "__main__":
    main()
