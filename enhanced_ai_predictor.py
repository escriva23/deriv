# enhanced_ai_predictor.py - Enhanced AI Predictor with Pattern-Aware Features
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
import lightgbm as lgb
import time

from config import config
from pattern_calibration import probability_calibrator
from pattern_detectors import pattern_engine
from meta_controller import meta_controller
from online_learning import online_learning_system

logger = logging.getLogger(__name__)

class EnhancedAIPredictor:
    """Enhanced AI predictor with pattern-aware features and calibrated probabilities"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.prediction_cache = {}
        self.model_performance = {}
        
        # Enhanced feature set
        self.base_features = [
            'last_digit', 'price_change', 'volatility_5', 'volatility_20',
            'streak_length', 'streak_direction', 'momentum_score',
            'digit_freq_even', 'digit_freq_odd', 'price_trend_5', 'price_trend_20'
        ]
        
        # Pattern features will be added dynamically
        self.pattern_features = []
        
    def prepare_enhanced_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Prepare enhanced features including patterns"""
        if len(df) < 50:
            return pd.DataFrame()
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Start with base features from original predictor
        df_enhanced = self.prepare_base_features(df)
        
        # Add pattern features for each row
        pattern_features_list = []
        
        for i, row in df_enhanced.iterrows():
            if pd.isna(row.get('last_digit')):
                # Use default pattern features
                pattern_features = self._get_default_pattern_features()
            else:
                # Update pattern engine and get features
                price = row.get('quote', 0) if 'quote' in row else row.get('price', 0)
                last_digit = int(row['last_digit'])
                
                try:
                    pattern_features = pattern_engine.update_patterns(price, last_digit, symbol)
                except Exception as e:
                    logger.warning(f"Error getting pattern features: {e}")
                    pattern_features = self._get_default_pattern_features()
            
            pattern_features_list.append(pattern_features)
        
        # Convert pattern features to DataFrame
        pattern_df = pd.DataFrame(pattern_features_list)
        
        # Combine base and pattern features
        df_combined = pd.concat([df_enhanced.reset_index(drop=True), pattern_df], axis=1)
        
        # Update feature columns list
        self.feature_columns = list(df_combined.columns)
        
        # Store pattern feature names
        self.pattern_features = [col for col in pattern_df.columns if col.startswith(('pattern_', 'hist_', 'ngram_'))]
        
        return df_combined.dropna()
    
    def prepare_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare base features (original functionality)"""
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
        
        return df
    
    def _get_default_pattern_features(self) -> Dict[str, float]:
        """Get default pattern features when pattern engine fails"""
        return {
            'pattern_even_signal': 0.0,
            'pattern_odd_signal': 0.0,
            'pattern_over_signal': 0.0,
            'pattern_under_signal': 0.0,
            'pattern_momentum_signal': 0.0,
            'pattern_confidence': 0.5,
            'ngram_pattern_strength': 0.0,
            'hist_50_even_ratio': 0.5,
            'hist_50_odd_ratio': 0.5,
            'even_odd_bias': 0.0,
            'high_low_bias': 0.0
        }
    
    def train_enhanced_models(self, symbol: str, prediction_type: str = 'digit_parity'):
        """Train enhanced ML models with pattern features"""
        logger.info(f"Training enhanced models for {symbol} - {prediction_type}")
        
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
        
        # Prepare enhanced features
        df_features = self.prepare_enhanced_features(df, symbol)
        if len(df_features) < 500:
            logger.warning(f"Insufficient enhanced features for {symbol}")
            return False
        
        # Create labels
        labels = self.create_labels(df_features, prediction_type)
        
        # Select available features
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
        
        # Enhanced model ensemble
        models = {
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                random_state=42,
                verbose=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        best_model = None
        best_score = 0
        model_scores = {}
        
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy')
                avg_score = np.mean(scores)
                model_scores[name] = avg_score
                
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
        self.model_performance[key] = {
            'best_model': model_name,
            'cv_score': best_score,
            'all_scores': model_scores,
            'feature_count': len(available_features),
            'training_samples': len(X),
            'pattern_features_used': len([f for f in available_features if f in self.pattern_features])
        }
        
        # Save to disk
        joblib.dump(model, f"enhanced_model_{key}_{model_name}.pkl")
        joblib.dump(scaler, f"enhanced_scaler_{key}.pkl")
        
        # Initialize calibration training
        if probability_calibrator.should_retrain(key):
            probability_calibrator.train_calibrators(key)
        
        logger.info(f"Enhanced model for {symbol}: {model_name} (score: {best_score:.4f}, "
                   f"pattern features: {len([f for f in available_features if f in self.pattern_features])})")
        
        return True
    
    def create_labels(self, df: pd.DataFrame, prediction_type: str = 'digit_parity') -> pd.Series:
        """Create prediction labels (same as original)"""
        if prediction_type == 'digit_parity':
            return (df['digit_parity'].shift(-1) == 'even').astype(int)
        elif prediction_type == 'digit_over_under':
            return (df['last_digit'].shift(-1) > 5).astype(int)
        elif prediction_type == 'price_direction':
            return (df['price_change'].shift(-1) > 0).astype(int)
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")
    
    def predict_enhanced(self, symbol: str, features: Dict, 
                        prediction_type: str = 'digit_parity') -> Tuple[int, float, Dict]:
        """Enhanced prediction with calibration and meta-analysis"""
        key = f"{symbol}_{prediction_type}"
        
        if key not in self.models:
            logger.warning(f"No trained model for {key}")
            return 0, 0.0, {}
        
        try:
            # Get current price and last digit for pattern features
            price = features.get('price', 0) or features.get('quote', 0)
            last_digit = features.get('last_digit', 0)
            
            # Update pattern engine and get pattern features
            try:
                pattern_features = pattern_engine.update_patterns(price, last_digit, symbol)
            except Exception as e:
                logger.warning(f"Error getting pattern features: {e}")
                pattern_features = self._get_default_pattern_features()
            
            # Combine base and pattern features
            combined_features = {**features, **pattern_features}
            
            # Prepare feature vector
            feature_vector = []
            for col in self.feature_columns:
                if col in combined_features:
                    feature_vector.append(combined_features[col])
                else:
                    feature_vector.append(0.0)  # Default value
            
            # Scale features
            X = np.array(feature_vector).reshape(1, -1)
            X_scaled = self.scalers[key].transform(X)
            
            # Make prediction
            model = self.models[key]
            prediction = model.predict(X_scaled)[0]
            
            # Get raw probability
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_scaled)[0]
                raw_probability = probabilities[1]  # Probability of positive class
                raw_confidence = max(probabilities)
            else:
                raw_probability = 0.6 if prediction == 1 else 0.4
                raw_confidence = 0.6
            
            # Calibrate probability
            calibrated_probability = probability_calibrator.calibrate_probability(
                key, raw_probability
            )
            
            # Get uncertainty estimate
            uncertainty = probability_calibrator.get_uncertainty_estimate(key, raw_probability)
            
            # Enhanced prediction metadata
            prediction_metadata = {
                'raw_probability': float(raw_probability),
                'calibrated_probability': float(calibrated_probability),
                'raw_confidence': float(raw_confidence),
                'uncertainty': float(uncertainty),
                'pattern_features_used': len([f for f in self.feature_columns if f in self.pattern_features]),
                'pattern_signals': {k: v for k, v in pattern_features.items() if k.startswith('pattern_')},
                'model_name': self.model_performance.get(key, {}).get('best_model', 'unknown')
            }
            
            # Use calibrated probability as final confidence
            final_confidence = calibrated_probability if calibrated_probability > 0.5 else (1 - calibrated_probability)
            
            return int(prediction), float(final_confidence), prediction_metadata
            
        except Exception as e:
            logger.error(f"Enhanced prediction error for {key}: {e}")
            return 0, 0.0, {}
    
    def get_enhanced_ensemble_prediction(self, symbol: str, features: Dict) -> Dict:
        """Get enhanced ensemble predictions with meta-controller analysis"""
        predictions = {}
        all_metadata = {}
        
        # Get predictions from all types
        for pred_type in ['digit_parity', 'digit_over_under', 'price_direction']:
            pred, conf, metadata = self.predict_enhanced(symbol, features, pred_type)
            predictions[pred_type] = {
                'prediction': pred,
                'confidence': conf,
                'metadata': metadata
            }
            all_metadata[pred_type] = metadata
        
        # Use meta-controller for final decision
        try:
            price = features.get('price', 0) or features.get('quote', 0)
            last_digit = features.get('last_digit', 0)
            
            # Convert predictions to format expected by meta-controller
            model_predictions = {}
            for pred_type, pred_data in predictions.items():
                model_predictions[f"enhanced_{pred_type}"] = {
                    'probability': pred_data['metadata'].get('calibrated_probability', 0.5),
                    'confidence': pred_data['confidence']
                }
            
            # Get meta-controller decision
            meta_decision = meta_controller.analyze_trading_opportunity(
                symbol, price, last_digit, model_predictions
            )
            
            if meta_decision:
                # Add meta-controller decision to results
                predictions['meta_decision'] = {
                    'contract_type': meta_decision.contract_type,
                    'probability': meta_decision.calibrated_probability,
                    'confidence': meta_decision.confidence,
                    'expected_value': meta_decision.expected_value,
                    'stake': meta_decision.stake,
                    'reasoning': meta_decision.reasoning,
                    'drift_detected': meta_decision.drift_detected,
                    'uncertainty': meta_decision.uncertainty
                }
        except Exception as e:
            logger.error(f"Error in meta-controller analysis: {e}")
        
        # Add overall metadata
        predictions['ensemble_metadata'] = {
            'total_pattern_features': len(pattern_engine.pattern_features) if hasattr(pattern_engine, 'pattern_features') else 0,
            'calibration_active': True,
            'meta_controller_active': True,
            'prediction_timestamp': time.time()
        }
        
        return predictions
    
    def should_trade_enhanced(self, predictions: Dict) -> Tuple[bool, str, float, Dict]:
        """Enhanced trading decision with meta-controller integration"""
        # Check if meta-controller made a decision
        if 'meta_decision' in predictions:
            meta_decision = predictions['meta_decision']
            
            # Use meta-controller's decision
            return (
                True,
                meta_decision['contract_type'],
                meta_decision['confidence'],
                {
                    'reasoning': meta_decision['reasoning'],
                    'expected_value': meta_decision['expected_value'],
                    'stake': meta_decision['stake'],
                    'drift_detected': meta_decision.get('drift_detected', False),
                    'uncertainty': meta_decision.get('uncertainty', 0.5)
                }
            )
        
        # Fallback to original logic if meta-controller didn't decide
        best_pred = None
        best_conf = 0
        best_type = None
        
        for pred_type, pred_data in predictions.items():
            if pred_type in ['ensemble_metadata']:
                continue
                
            confidence = pred_data.get('confidence', 0)
            if confidence > best_conf:
                best_conf = confidence
                best_pred = pred_data.get('prediction', 0)
                best_type = pred_type
        
        # Check confidence threshold
        if best_conf < config.MIN_CONFIDENCE:
            return False, "low_confidence", best_conf, {'reasoning': f"Confidence {best_conf:.3f} below threshold {config.MIN_CONFIDENCE}"}
        
        # Map prediction to contract type
        contract_mapping = {
            'digit_parity': 'DIGITEVEN' if best_pred == 1 else 'DIGITODD',
            'digit_over_under': 'DIGITOVER' if best_pred == 1 else 'DIGITUNDER',
            'price_direction': 'CALL' if best_pred == 1 else 'PUT'
        }
        
        contract_type = contract_mapping.get(best_type, 'DIGITEVEN')
        
        return True, contract_type, best_conf, {
            'reasoning': f"Fallback decision: {best_type} with {best_conf:.3f} confidence",
            'expected_value': 0.05,  # Placeholder
            'stake': 2.0  # Placeholder
        }
    
    def update_enhanced_performance(self, symbol: str, prediction_type: str,
                                  predicted: int, actual: int, profit: float,
                                  calibrated_probability: float = None):
        """Update enhanced model performance including calibration"""
        # Update original performance tracking
        try:
            # Store in database (original functionality)
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS enhanced_model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    prediction_type TEXT,
                    predicted INTEGER,
                    actual INTEGER,
                    correct INTEGER,
                    profit REAL,
                    calibrated_probability REAL,
                    timestamp INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                INSERT INTO enhanced_model_performance 
                (symbol, prediction_type, predicted, actual, correct, profit, 
                 calibrated_probability, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, prediction_type, predicted, actual,
                1 if predicted == actual else 0, profit,
                calibrated_probability, int(time.time())
            ))
            
            conn.commit()
            conn.close()
            
            # Update calibration system
            model_key = f"{symbol}_{prediction_type}"
            if calibrated_probability is not None:
                probability_calibrator.collect_calibration_data(
                    model_key, calibrated_probability, actual == 1, symbol, prediction_type
                )
            
            # Update online learning system
            features = {'last_digit': 0}  # Simplified for now
            online_learning_system.add_sample(
                model_key, features, actual, calibrated_probability or 0.5, 0.7
            )
            
        except Exception as e:
            logger.error(f"Error updating enhanced performance: {e}")
    
    def get_enhanced_model_stats(self, symbol: str, days: int = 7) -> Dict:
        """Get enhanced model performance statistics"""
        try:
            base_stats = self.get_model_stats(symbol, days)  # Original stats
            
            # Add enhanced statistics
            conn = sqlite3.connect(config.DB_PATH)
            
            df = pd.read_sql_query("""
                SELECT * FROM enhanced_model_performance 
                WHERE symbol = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, conn, params=(symbol, int(time.time()) - (days * 24 * 3600)))
            
            conn.close()
            
            enhanced_stats = base_stats.copy()
            
            if len(df) > 0:
                enhanced_stats.update({
                    'enhanced_accuracy': df['correct'].mean(),
                    'enhanced_profit': df['profit'].sum(),
                    'enhanced_trades': len(df),
                    'calibrated_predictions': len(df[df['calibrated_probability'].notna()]),
                    'avg_calibrated_probability': df['calibrated_probability'].mean()
                })
            
            # Add calibration stats
            for pred_type in ['digit_parity', 'digit_over_under']:
                model_key = f"{symbol}_{pred_type}"
                cal_stats = probability_calibrator.get_calibration_stats(model_key)
                enhanced_stats[f'{pred_type}_calibration'] = cal_stats
            
            # Add online learning stats
            for pred_type in ['digit_parity', 'digit_over_under']:
                model_key = f"{symbol}_{pred_type}"
                online_stats = online_learning_system.get_model_stats(model_key)
                enhanced_stats[f'{pred_type}_online'] = online_stats
            
            return enhanced_stats
            
        except Exception as e:
            logger.error(f"Error getting enhanced model stats: {e}")
            return self.get_model_stats(symbol, days)  # Fallback to original
    
    def get_model_stats(self, symbol: str, days: int = 7) -> Dict:
        """Original model stats method (fallback)"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            
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

# Global enhanced predictor instance
enhanced_ai_predictor = EnhancedAIPredictor()

if __name__ == "__main__":
    # Test enhanced AI predictor
    predictor = EnhancedAIPredictor()
    
    print("Testing Enhanced AI Predictor...")
    
    # Test feature preparation
    test_data = pd.DataFrame({
        'timestamp': [1000000, 1000001, 1000002],
        'last_digit': [5, 6, 7],
        'digit_parity': ['odd', 'even', 'odd'],
        'price_change': [0.1, -0.2, 0.3],
        'volatility_5': [1.0, 1.2, 0.8],
        'volatility_20': [0.9, 1.1, 0.7],
        'streak_length': [1, 2, 1],
        'streak_direction': [1, -1, 1],
        'momentum_score': [0.5, -0.3, 0.7]
    })
    
    enhanced_features = predictor.prepare_enhanced_features(test_data, "R_100")
    print(f"Enhanced features shape: {enhanced_features.shape}")
    print(f"Feature columns: {len(predictor.feature_columns)}")
    print(f"Pattern features: {len(predictor.pattern_features)}")
    
    # Test prediction
    test_features = {
        'last_digit': 6,
        'price': 1234.56,
        'price_change': 0.1,
        'volatility_5': 1.0,
        'volatility_20': 0.8,
        'streak_length': 2,
        'streak_direction': 1,
        'momentum_score': 0.3
    }
    
    pred, conf, metadata = predictor.predict_enhanced("R_100", test_features)
    print(f"Prediction: {pred}, Confidence: {conf:.3f}")
    print(f"Metadata keys: {list(metadata.keys())}")
    
    # Test ensemble prediction
    ensemble = predictor.get_enhanced_ensemble_prediction("R_100", test_features)
    print(f"Ensemble keys: {list(ensemble.keys())}")
    
    print("Enhanced AI Predictor test completed!")

