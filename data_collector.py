# data_collector.py - Advanced tick data collection and storage
import json
import time
import sqlite3
import threading
import logging
from datetime import datetime
from typing import Dict, List, Optional
from websocket import create_connection, WebSocketConnectionClosedException
import pandas as pd
import numpy as np
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TickDataCollector:
    def __init__(self):
        self.ws = None
        self.running = False
        self.tick_buffer = []
        self.buffer_lock = threading.Lock()
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for tick storage"""
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        # Create ticks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                epoch INTEGER,
                symbol TEXT,
                quote REAL,
                bid REAL,
                ask REAL,
                pip_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(epoch, symbol)
            )
        """)
        
        # Create features table for ML
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tick_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp INTEGER,
                last_digit INTEGER,
                digit_parity TEXT,
                price_change REAL,
                volatility_5 REAL,
                volatility_20 REAL,
                streak_length INTEGER,
                streak_direction INTEGER,
                momentum_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_id TEXT,
                symbol TEXT,
                contract_type TEXT,
                stake REAL,
                prediction_confidence REAL,
                entry_time INTEGER,
                exit_time INTEGER,
                profit_loss REAL,
                win INTEGER,
                features_used TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def connect_websocket(self):
        """Establish WebSocket connection to Deriv"""
        try:
            url = f"{config.DERIV_WS_URL}?app_id={config.DERIV_APP_ID}"
            self.ws = create_connection(url)
            logger.info("WebSocket connected successfully")
            return True
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False
    
    def subscribe_to_ticks(self, symbols: List[str]):
        """Subscribe to tick streams for multiple symbols"""
        if not self.ws:
            return False
            
        for symbol in symbols:
            try:
                subscribe_msg = {
                    "ticks": symbol,
                    "subscribe": 1
                }
                self.ws.send(json.dumps(subscribe_msg))
                logger.info(f"Subscribed to {symbol} ticks")
            except Exception as e:
                logger.error(f"Failed to subscribe to {symbol}: {e}")
                return False
        return True
    
    def save_tick(self, tick_data: Dict):
        """Save tick data to database"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR IGNORE INTO ticks 
                (timestamp, epoch, symbol, quote, bid, ask, pip_size)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                int(time.time()),
                tick_data.get('epoch'),
                tick_data.get('symbol'),
                tick_data.get('quote'),
                tick_data.get('bid'),
                tick_data.get('ask'),
                tick_data.get('pip_size')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save tick: {e}")
    
    def calculate_features(self, symbol: str, current_tick: Dict):
        """Calculate ML features from recent tick data"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            
            # Get recent ticks for feature calculation
            df = pd.read_sql_query("""
                SELECT * FROM ticks 
                WHERE symbol = ? 
                ORDER BY epoch DESC 
                LIMIT ?
            """, conn, params=(symbol, config.LOOKBACK_WINDOW))
            
            if len(df) < 20:
                conn.close()
                return None
            
            # Calculate features
            quotes = df['quote'].values
            
            # Last digit analysis
            last_digit = int(str(current_tick['quote']).split('.')[-1][-1])
            digit_parity = 'even' if last_digit % 2 == 0 else 'odd'
            
            # Price changes and volatility
            price_changes = np.diff(quotes)
            volatility_5 = np.std(price_changes[-5:]) if len(price_changes) >= 5 else 0
            volatility_20 = np.std(price_changes[-20:]) if len(price_changes) >= 20 else 0
            
            # Streak detection
            directions = np.sign(price_changes)
            streak_length = 0
            streak_direction = 0
            
            if len(directions) > 0:
                current_dir = directions[-1]
                for i in range(len(directions)-1, -1, -1):
                    if directions[i] == current_dir:
                        streak_length += 1
                    else:
                        break
                streak_direction = int(current_dir)
            
            # Momentum score (custom indicator)
            momentum_score = 0
            if len(price_changes) >= 10:
                recent_changes = price_changes[-10:]
                momentum_score = np.sum(recent_changes) / np.std(recent_changes) if np.std(recent_changes) > 0 else 0
            
            # Save features
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO tick_features 
                (symbol, timestamp, last_digit, digit_parity, price_change, 
                 volatility_5, volatility_20, streak_length, streak_direction, momentum_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, current_tick['epoch'], last_digit, digit_parity,
                price_changes[-1] if len(price_changes) > 0 else 0,
                volatility_5, volatility_20, streak_length, streak_direction, momentum_score
            ))
            
            conn.commit()
            conn.close()
            
            return {
                'last_digit': last_digit,
                'digit_parity': digit_parity,
                'price_change': price_changes[-1] if len(price_changes) > 0 else 0,
                'volatility_5': volatility_5,
                'volatility_20': volatility_20,
                'streak_length': streak_length,
                'streak_direction': streak_direction,
                'momentum_score': momentum_score
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate features: {e}")
            return None
    
    def listen_for_ticks(self):
        """Main loop to listen for incoming ticks"""
        while self.running:
            try:
                if not self.ws:
                    if not self.connect_websocket():
                        time.sleep(5)
                        continue
                    self.subscribe_to_ticks(config.SYMBOLS)
                
                message = self.ws.recv()
                data = json.loads(message)
                
                if 'tick' in data:
                    tick = data['tick']
                    
                    # Save raw tick data
                    self.save_tick(tick)
                    
                    # Calculate and save features
                    features = self.calculate_features(tick['symbol'], tick)
                    
                    # Add to buffer for real-time processing
                    with self.buffer_lock:
                        self.tick_buffer.append({
                            'tick': tick,
                            'features': features,
                            'timestamp': time.time()
                        })
                        
                        # Keep buffer size manageable
                        if len(self.tick_buffer) > 1000:
                            self.tick_buffer = self.tick_buffer[-500:]
                    
                    logger.info(f"Processed tick: {tick['symbol']} @ {tick['quote']}")
                
            except WebSocketConnectionClosedException:
                logger.warning("WebSocket connection lost, reconnecting...")
                self.ws = None
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error in tick listener: {e}")
                time.sleep(1)
    
    def get_recent_features(self, symbol: str, count: int = 50) -> pd.DataFrame:
        """Get recent features for ML model"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            df = pd.read_sql_query("""
                SELECT * FROM tick_features 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, conn, params=(symbol, count))
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Failed to get recent features: {e}")
            return pd.DataFrame()
    
    def start_collection(self):
        """Start the data collection process"""
        self.running = True
        logger.info("Starting tick data collection...")
        self.listen_for_ticks()
    
    def stop_collection(self):
        """Stop the data collection process"""
        self.running = False
        if self.ws:
            self.ws.close()
        logger.info("Stopped tick data collection")

if __name__ == "__main__":
    collector = TickDataCollector()
    try:
        collector.start_collection()
    except KeyboardInterrupt:
        collector.stop_collection()
        logger.info("Collection stopped by user")
