# create_database_tables.py - Create required database tables for enhanced system
import sqlite3
import logging
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_enhanced_tables():
    """Create all required tables for the enhanced pattern-aware system"""
    
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        # Pattern features table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                last_digit INTEGER,
                pattern_confidence REAL,
                even_signal REAL,
                odd_signal REAL,
                over_signal REAL,
                under_signal REAL,
                momentum_signal REAL,
                pattern_strength REAL,
                n_gram_score REAL,
                sequence_score REAL,
                histogram_score REAL,
                streak_length INTEGER,
                streak_direction TEXT,
                features TEXT,  -- JSON encoded features
                UNIQUE(timestamp, symbol)
            )
        """)
        logger.info("✓ Created pattern_features table")
        
        # Calibration data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibration_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                model_key TEXT NOT NULL,
                raw_probability REAL NOT NULL,
                actual_outcome INTEGER NOT NULL,
                symbol TEXT,
                contract_type TEXT,
                features TEXT  -- JSON encoded features
            )
        """)
        logger.info("✓ Created calibration_data table")
        
        # Online learning samples table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS online_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                model_key TEXT NOT NULL,
                features TEXT NOT NULL,  -- JSON encoded
                label INTEGER NOT NULL,
                prediction REAL,
                confidence REAL,
                loss REAL
            )
        """)
        logger.info("✓ Created online_samples table")
        
        # Model updates table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                model_key TEXT NOT NULL,
                update_type TEXT NOT NULL,
                old_performance REAL,
                new_performance REAL,
                sample_count INTEGER,
                metadata TEXT  -- JSON encoded
            )
        """)
        logger.info("✓ Created model_updates table")
        
        # Martingale sequences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS martingale_sequences (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                contract_type TEXT NOT NULL,
                start_time INTEGER NOT NULL,
                end_time INTEGER,
                status TEXT NOT NULL,
                initial_probability REAL,
                pattern_score REAL,
                total_stake REAL DEFAULT 0,
                total_profit REAL DEFAULT 0,
                levels_used INTEGER DEFAULT 0,
                max_level INTEGER DEFAULT 0,
                trades_count INTEGER DEFAULT 0,
                success_probability REAL,
                risk_budget REAL,
                metadata TEXT  -- JSON encoded
            )
        """)
        logger.info("✓ Created martingale_sequences table")
        
        # Martingale trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS martingale_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_id TEXT NOT NULL,
                level INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                stake REAL NOT NULL,
                profit_loss REAL,
                win INTEGER,
                probability REAL,
                contract_id TEXT,
                FOREIGN KEY (sequence_id) REFERENCES martingale_sequences (id)
            )
        """)
        logger.info("✓ Created martingale_trades table")
        
        # Enhanced trades table (if not exists from original system)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_time INTEGER NOT NULL,
                exit_time INTEGER,
                symbol TEXT NOT NULL,
                contract_type TEXT NOT NULL,
                contract_id TEXT,
                stake REAL NOT NULL,
                payout REAL,
                profit_loss REAL,
                win INTEGER,
                raw_confidence REAL,
                calibrated_probability REAL,
                pattern_score REAL,
                ev_calculated REAL,
                martingale_sequence_id TEXT,
                martingale_level INTEGER,
                features TEXT,  -- JSON encoded
                metadata TEXT   -- JSON encoded
            )
        """)
        logger.info("✓ Created/verified trades table")
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_pattern_features_symbol_time ON pattern_features (symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_calibration_model_time ON calibration_data (model_key, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_online_samples_model ON online_samples (model_key, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_martingale_seq_symbol ON martingale_sequences (symbol, start_time)",
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades (symbol, entry_time)"
        ]
        
        for idx_sql in indexes:
            cursor.execute(idx_sql)
        
        logger.info("✓ Created database indexes")
        
        conn.commit()
        conn.close()
        
        logger.info(f"🎉 Successfully created all enhanced database tables in {config.DB_PATH}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        return False

def verify_tables():
    """Verify all tables were created successfully"""
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        required_tables = [
            'pattern_features', 'calibration_data', 'online_samples', 
            'model_updates', 'martingale_sequences', 'martingale_trades', 'trades'
        ]
        
        missing_tables = [t for t in required_tables if t not in tables]
        
        if missing_tables:
            logger.warning(f"Missing tables: {missing_tables}")
        else:
            logger.info("✅ All required tables are present")
        
        logger.info(f"Available tables: {', '.join(sorted(tables))}")
        
        conn.close()
        return len(missing_tables) == 0
        
    except Exception as e:
        logger.error(f"Error verifying tables: {e}")
        return False

if __name__ == "__main__":
    print("🗄️ Creating Enhanced Pattern-Aware Trading Database Tables")
    print("=" * 60)
    
    success = create_enhanced_tables()
    
    if success:
        print("\n📋 Verifying tables...")
        verify_tables()
        print("\n🎉 Database setup complete!")
        print("\nYour enhanced pattern-aware AI trading system database is ready.")
        print("You can now run the enhanced modules without table errors.")
    else:
        print("\n❌ Database setup failed. Check the logs above.")

