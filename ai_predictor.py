# ai_predictor.py - Advanced ML prediction engine
import numpy as np
import pandas as pd
import sqlite3
import joblib
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb
from config import config

logger = logging.getLogger(__name__)

class AIPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = [
            'last_digit', 'price_change', 'volatility_5', 'volatility_20',
            'streak_length', 'streak_direction', 'momentum_score',
            'digit_freq_even', 'digit_freq_odd', 'price_trend_5', 'price_trend_20'
        ]
        self.prediction_cache = {}
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML models"""
        if len(df) < 50:
            return pd.DataFrame()
            
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate additional features
        df['digit_freq_even'] = df['digit_parity'].rolling(20, min_periods=1).apply(
            lambda x: (x == 'even').sum() / len(x)
        )
        df['digit_freq_odd'] = 1 - df['digit_freq_even']
        
        # Price trend indicators
        df['price_trend_5'] = df['price_change'].rolling(5, min_periods=1).mean()
        df['price_trend_20'] = df['price_change'].rolling(20, min_periods=1).mean()
        
        # Volatility ratios
        df['vol_ratio'] = df['volatility_5'] / (df['volatility_20'] + 1e-8)
        
        # Momentum indicators
        df['momentum_change'] = df['momentum_score'].diff()
        df['streak_change'] = df['streak_length'].diff()
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'last_digit_lag_{lag}'] = df['last_digit'].shift(lag)
            df[f'parity_lag_{lag}'] = (df['digit_parity'].shift(lag) == 'even').astype(int)
        
        return df.dropna()
    
    def create_labels(self, df: pd.DataFrame, prediction_type: str = 'digit_parity') -> pd.Series:
        """Create prediction labels"""
        if prediction_type == 'digit_parity':
            # Predict next tick's digit parity
            return (df['digit_parity'].shift(-1) == 'even').astype(int)
        elif prediction_type == 'digit_over_under':
            # Predict if next digit > 5
            return (df['last_digit'].shift(-1) > 5).astype(int)
        elif prediction_type == 'price_direction':
            # Predict price direction
            return (df['price_change'].shift(-1) > 0).astype(int)
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")
    
    def train_models(self, symbol: str, prediction_type: str = 'digit_parity'):
        """Train ML models for a specific symbol and prediction type"""
        logger.info(f"Training models for {symbol} - {prediction_type}")
        
        # Load training data
        conn = sqlite3.connect(config.DB_PATH)
        df = pd.read_sql_query("""
            SELECT * FROM tick_features 
            WHERE symbol = ? 
            ORDER BY timestamp
        """, conn, params=(symbol,))
        conn.close()
        
        if len(df) < 1000:
            logger.warning(f"Insufficient data for {symbol}: {len(df)} samples")
            return False
        
        # Prepare features and labels
        df_features = self.prepare_features(df)
        if len(df_features) < 500:
            logger.warning(f"Insufficient features for {symbol}")
            return False
            
        labels = self.create_labels(df_features, prediction_type)
        
        # Select feature columns that exist
        available_features = [col for col in self.feature_columns if col in df_features.columns]
        X = df_features[available_features].fillna(0)
        y = labels.dropna()
        
        # Align X and y
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len]
        y = y.iloc[:min_len]
        
        if len(X) < 100:
            logger.warning(f"Not enough aligned data for {symbol}")
            return False
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train multiple models
        models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            ),
            'logistic': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy')
                avg_score = np.mean(scores)
                logger.info(f"{name} CV score: {avg_score:.4f} (+/- {np.std(scores) * 2:.4f})")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = (name, model)
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        if best_model is None:
            logger.error(f"No models successfully trained for {symbol}")
            return False
        
        # Train best model on full dataset
        model_name, model = best_model
        model.fit(X_scaled, y)
        
        # Store model and scaler
        key = f"{symbol}_{prediction_type}"
        self.models[key] = model
        self.scalers[key] = scaler
        
        # Save to disk
        joblib.dump(model, f"model_{key}_{model_name}.pkl")
        joblib.dump(scaler, f"scaler_{key}.pkl")
        
        logger.info(f"Best model for {symbol}: {model_name} (score: {best_score:.4f})")
        return True
    
    def predict(self, symbol: str, features: Dict, prediction_type: str = 'digit_parity') -> Tuple[int, float]:
        """Make prediction with confidence score"""
        key = f"{symbol}_{prediction_type}"
        
        if key not in self.models:
            logger.warning(f"No trained model for {key}")
            return 0, 0.0
        
        try:
            # Prepare feature vector
            feature_vector = []
            for col in self.feature_columns:
                if col in features:
                    feature_vector.append(features[col])
                else:
                    feature_vector.append(0.0)  # Default value
            
            # Scale features
            X = np.array(feature_vector).reshape(1, -1)
            X_scaled = self.scalers[key].transform(X)
            
            # Make prediction
            model = self.models[key]
            prediction = model.predict(X_scaled)[0]
            
            # Get confidence (probability)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_scaled)[0]
                confidence = max(probabilities)
            else:
                confidence = 0.6  # Default confidence for models without probability
            
            return int(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Prediction error for {key}: {e}")
            return 0, 0.0
    
    def get_ensemble_prediction(self, symbol: str, features: Dict) -> Dict:
        """Get ensemble predictions from multiple prediction types"""
        predictions = {}
        
        for pred_type in ['digit_parity', 'digit_over_under', 'price_direction']:
            pred, conf = self.predict(symbol, features, pred_type)
            predictions[pred_type] = {
                'prediction': pred,
                'confidence': conf
            }
        
        return predictions
    
    def should_trade(self, predictions: Dict) -> Tuple[bool, str, float]:
        """Decide if we should trade based on ensemble predictions"""
        # Get highest confidence prediction
        best_pred = None
        best_conf = 0
        best_type = None
        
        for pred_type, pred_data in predictions.items():
            if pred_data['confidence'] > best_conf:
                best_conf = pred_data['confidence']
                best_pred = pred_data['prediction']
                best_type = pred_type
        
        # Only trade if confidence is above threshold
        if best_conf < config.MIN_CONFIDENCE:
            return False, "low_confidence", best_conf
        
        # Map prediction to contract type
        contract_mapping = {
            'digit_parity': 'DIGITEVEN' if best_pred == 1 else 'DIGITODD',
            'digit_over_under': 'DIGITOVER' if best_pred == 1 else 'DIGITUNDER',
            'price_direction': 'CALL' if best_pred == 1 else 'PUT'
        }
        
        contract_type = contract_mapping.get(best_type, 'DIGITEVEN')
        
        return True, contract_type, best_conf
    
    def update_model_performance(self, symbol: str, prediction_type: str, 
                               predicted: int, actual: int, profit: float):
        """Update model performance tracking"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            # Create performance tracking table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    prediction_type TEXT,
                    predicted INTEGER,
                    actual INTEGER,
                    correct INTEGER,
                    profit REAL,
                    timestamp INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert performance record
            cursor.execute("""
                INSERT INTO model_performance 
                (symbol, prediction_type, predicted, actual, correct, profit, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, prediction_type, predicted, actual,
                1 if predicted == actual else 0, profit, int(time.time())
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
    
    def get_model_stats(self, symbol: str, days: int = 7) -> Dict:
        """Get model performance statistics"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            
            # Get recent performance
            df = pd.read_sql_query("""
                SELECT * FROM model_performance 
                WHERE symbol = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, conn, params=(symbol, int(time.time()) - (days * 24 * 3600)))
            
            conn.close()
            
            if len(df) == 0:
                return {'accuracy': 0, 'profit': 0, 'trades': 0}
            
            stats = {
                'accuracy': df['correct'].mean(),
                'profit': df['profit'].sum(),
                'trades': len(df),
                'win_rate': (df['profit'] > 0).mean()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting model stats: {e}")
            return {'accuracy': 0, 'profit': 0, 'trades': 0}

if __name__ == "__main__":
    # Example usage
    predictor = AIPredictor()
    
    # Train models for all symbols
    for symbol in config.SYMBOLS:
        predictor.train_models(symbol, 'digit_parity')
        predictor.train_models(symbol, 'digit_over_under')
