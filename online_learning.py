# online_learning.py - Online Learning and Model Adaptation System
import numpy as np
import pandas as pd
import sqlite3
import logging
import time
import pickle
import os
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, defaultdict
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import joblib
import json
from config import config

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

logger = logging.getLogger(__name__)

@dataclass
class ModelUpdate:
    """Represents a model update event"""
    timestamp: float
    model_key: str
    update_type: str  # 'incremental', 'retrain', 'drift_adaptation'
    samples_used: int
    performance_before: float
    performance_after: float
    drift_detected: bool
    adaptation_strategy: str

class OnlineModel:
    """Online learning model wrapper"""
    
    def __init__(self, model_type: str = 'sgd', model_key: str = None):
        self.model_type = model_type
        self.model_key = model_key
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.sample_count = 0
        self.performance_history = deque(maxlen=100)
        self.last_update_time = 0
        self.update_frequency = 10  # Update every N samples
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the online learning model"""
        if self.model_type == 'sgd':
            self.model = SGDClassifier(
                loss='log_loss',  # For probability estimates
                learning_rate='adaptive',
                eta0=0.01,
                random_state=42,
                max_iter=1000
            )
        elif self.model_type == 'passive_aggressive':
            self.model = PassiveAggressiveClassifier(
                random_state=42,
                max_iter=1000
            )
        else:
            # Default to SGD
            self.model = SGDClassifier(
                loss='log_loss',
                learning_rate='adaptive',
                eta0=0.01,
                random_state=42,
                max_iter=1000
            )
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray = None) -> bool:
        """Incrementally fit the model"""
        try:
            if not self.is_fitted:
                # First fit - need to fit scaler and provide classes
                if classes is None:
                    classes = np.array([0, 1])  # Binary classification
                
                X_scaled = self.scaler.fit_transform(X)
                self.model.partial_fit(X_scaled, y, classes=classes)
                self.is_fitted = True
            else:
                # Incremental update
                X_scaled = self.scaler.transform(X)
                self.model.partial_fit(X_scaled, y)
            
            self.sample_count += len(X)
            self.last_update_time = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in partial fit for {self.model_key}: {e}")
            return False
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        if not self.is_fitted:
            return np.array([[0.5, 0.5]] * len(X))
        
        try:
            X_scaled = self.scaler.transform(X)
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X_scaled)
            else:
                # For models without predict_proba, use decision function
                decision = self.model.decision_function(X_scaled)
                # Convert to probabilities using sigmoid
                proba_positive = 1 / (1 + np.exp(-decision))
                return np.column_stack([1 - proba_positive, proba_positive])
                
        except Exception as e:
            logger.error(f"Error in prediction for {self.model_key}: {e}")
            return np.array([[0.5, 0.5]] * len(X))
    
    def update_performance(self, accuracy: float):
        """Update performance tracking"""
        self.performance_history.append(accuracy)
    
    def get_recent_performance(self, window: int = 20) -> float:
        """Get recent performance average"""
        if len(self.performance_history) == 0:
            return 0.5
        
        recent = list(self.performance_history)[-window:]
        return np.mean(recent)
    
    def save_model(self, filepath: str) -> bool:
        """Save model to disk"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'is_fitted': self.is_fitted,
                'sample_count': self.sample_count,
                'performance_history': list(self.performance_history)
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving model {self.model_key}: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load model from disk"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.model_type = model_data['model_type']
            self.is_fitted = model_data['is_fitted']
            self.sample_count = model_data['sample_count']
            self.performance_history = deque(model_data['performance_history'], maxlen=100)
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_key}: {e}")
            return False

class AdaptiveLearningSystem:
    """Adaptive online learning system with drift detection"""
    
    def __init__(self):
        self.online_models: Dict[str, OnlineModel] = {}
        self.feature_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.label_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.model_updates = deque(maxlen=500)
        self.adaptation_strategies = {
            'conservative': {'retrain_threshold': 0.05, 'update_frequency': 20},
            'aggressive': {'retrain_threshold': 0.02, 'update_frequency': 5},
            'balanced': {'retrain_threshold': 0.03, 'update_frequency': 10}
        }
        self.current_strategy = 'balanced'
        self.init_database()
    
    def init_database(self):
        """Initialize online learning database tables"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            # Online learning samples table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS online_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    model_key TEXT,
                    features TEXT,
                    label INTEGER,
                    prediction REAL,
                    confidence REAL,
                    loss REAL,
                    used_for_training INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Model updates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    model_key TEXT,
                    update_type TEXT,
                    samples_used INTEGER,
                    performance_before REAL,
                    performance_after REAL,
                    drift_detected INTEGER,
                    adaptation_strategy TEXT,
                    update_details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Online performance metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS online_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    model_key TEXT,
                    window_size INTEGER,
                    accuracy REAL,
                    log_loss REAL,
                    prediction_count INTEGER,
                    drift_score REAL,
                    adaptation_triggered INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing online learning database: {e}")
    
    def get_or_create_model(self, model_key: str, model_type: str = 'sgd') -> OnlineModel:
        """Get existing model or create new one"""
        if model_key not in self.online_models:
            self.online_models[model_key] = OnlineModel(model_type, model_key)
            
            # Try to load existing model
            model_path = f"online_model_{model_key}.pkl"
            if os.path.exists(model_path):
                self.online_models[model_key].load_model(model_path)
                logger.info(f"Loaded existing online model: {model_key}")
        
        return self.online_models[model_key]
    
    def add_sample(self, model_key: str, features: Dict[str, Any], 
                   label: int, prediction: float = None, confidence: float = None):
        """Add a new training sample"""
        try:
            # Convert features to array
            feature_array = self._dict_to_array(features)
            
            # Store in buffers
            self.feature_buffer[model_key].append(feature_array)
            self.label_buffer[model_key].append(label)
            
            # Calculate loss if prediction provided
            loss = None
            if prediction is not None:
                loss = -label * np.log(prediction + 1e-15) - (1 - label) * np.log(1 - prediction + 1e-15)
            
            # Store in database
            self._store_sample(model_key, features, label, prediction, confidence, loss)
            
            # Check if we should trigger an update
            if self._should_update_model(model_key):
                self._update_model(model_key)
            
        except Exception as e:
            logger.error(f"Error adding sample for {model_key}: {e}")
    
    def _dict_to_array(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dictionary to numpy array"""
        # Define feature order (should match training data)
        feature_names = [
            'last_digit', 'price_change', 'volatility_5', 'volatility_20',
            'streak_length', 'streak_direction', 'momentum_score',
            'digit_freq_even', 'digit_freq_odd', 'price_trend_5', 'price_trend_20'
        ]
        
        # Add pattern features if available
        pattern_features = [k for k in features.keys() if k.startswith('pattern_') or k.startswith('hist_')]
        feature_names.extend(pattern_features)
        
        # Extract values
        feature_values = []
        for name in feature_names:
            value = features.get(name, 0.0)
            if isinstance(value, (int, float)):
                feature_values.append(float(value))
            else:
                feature_values.append(0.0)  # Default for non-numeric
        
        return np.array(feature_values)
    
    def _should_update_model(self, model_key: str) -> bool:
        """Determine if model should be updated"""
        strategy = self.adaptation_strategies[self.current_strategy]
        
        # Check if we have enough samples
        if len(self.feature_buffer[model_key]) < strategy['update_frequency']:
            return False
        
        # Check time since last update
        model = self.online_models.get(model_key)
        if model and time.time() - model.last_update_time < 60:  # At least 1 minute
            return False
        
        return True
    
    def _update_model(self, model_key: str):
        """Update model with recent samples"""
        try:
            if len(self.feature_buffer[model_key]) == 0:
                return
            
            # Get model
            model = self.get_or_create_model(model_key)
            
            # Prepare training data
            X = np.array(list(self.feature_buffer[model_key]))
            y = np.array(list(self.label_buffer[model_key]))
            
            if len(X) == 0 or len(y) == 0:
                return
            
            # Calculate performance before update
            performance_before = model.get_recent_performance()
            
            # Perform incremental update
            success = model.partial_fit(X, y)
            
            if success:
                # Evaluate performance after update
                if len(X) > 10:
                    y_pred_proba = model.predict_proba(X)
                    y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)
                    accuracy = accuracy_score(y, y_pred)
                    model.update_performance(accuracy)
                    performance_after = accuracy
                else:
                    performance_after = performance_before
                
                # Log update
                update = ModelUpdate(
                    timestamp=time.time(),
                    model_key=model_key,
                    update_type='incremental',
                    samples_used=len(X),
                    performance_before=performance_before,
                    performance_after=performance_after,
                    drift_detected=False,
                    adaptation_strategy=self.current_strategy
                )
                
                self.model_updates.append(update)
                self._store_model_update(update)
                
                # Save updated model
                model_path = f"online_model_{model_key}.pkl"
                model.save_model(model_path)
                
                logger.info(f"Updated online model {model_key}: {len(X)} samples, "
                           f"performance: {performance_before:.3f} → {performance_after:.3f}")
                
                # Clear processed samples
                self.feature_buffer[model_key].clear()
                self.label_buffer[model_key].clear()
            
        except Exception as e:
            logger.error(f"Error updating model {model_key}: {e}")
    
    def predict(self, model_key: str, features: Dict[str, Any]) -> Tuple[float, float]:
        """Get prediction from online model"""
        try:
            model = self.get_or_create_model(model_key)
            
            if not model.is_fitted:
                return 0.5, 0.5  # Default prediction
            
            feature_array = self._dict_to_array(features)
            X = feature_array.reshape(1, -1)
            
            proba = model.predict_proba(X)
            probability = proba[0, 1]  # Probability of positive class
            confidence = max(proba[0])  # Maximum probability as confidence
            
            return float(probability), float(confidence)
            
        except Exception as e:
            logger.error(f"Error predicting with {model_key}: {e}")
            return 0.5, 0.5
    
    def detect_concept_drift(self, model_key: str, window_size: int = 50) -> Tuple[bool, float]:
        """Detect concept drift in model performance"""
        try:
            model = self.online_models.get(model_key)
            if not model or len(model.performance_history) < window_size:
                return False, 0.0
            
            performance_history = list(model.performance_history)
            
            # Split into recent and historical windows
            recent_window = performance_history[-window_size//2:]
            historical_window = performance_history[-window_size:-window_size//2]
            
            if len(recent_window) < 10 or len(historical_window) < 10:
                return False, 0.0
            
            # Statistical test for difference in means
            recent_mean = np.mean(recent_window)
            historical_mean = np.mean(historical_window)
            
            # Calculate drift score (normalized difference)
            drift_score = abs(recent_mean - historical_mean)
            
            # Threshold for drift detection
            drift_threshold = self.adaptation_strategies[self.current_strategy]['retrain_threshold']
            
            return drift_score > drift_threshold, drift_score
            
        except Exception as e:
            logger.error(f"Error detecting drift for {model_key}: {e}")
            return False, 0.0
    
    def adapt_to_drift(self, model_key: str):
        """Adapt model to detected concept drift"""
        try:
            logger.warning(f"Adapting to concept drift for {model_key}")
            
            # Strategy 1: Reset model with recent data
            if len(self.feature_buffer[model_key]) > 50:
                # Get recent samples
                recent_X = np.array(list(self.feature_buffer[model_key])[-50:])
                recent_y = np.array(list(self.label_buffer[model_key])[-50:])
                
                # Create new model
                old_model = self.online_models[model_key]
                new_model = OnlineModel(old_model.model_type, model_key)
                
                # Train on recent data
                success = new_model.partial_fit(recent_X, recent_y)
                
                if success:
                    # Replace old model
                    self.online_models[model_key] = new_model
                    
                    # Log adaptation
                    update = ModelUpdate(
                        timestamp=time.time(),
                        model_key=model_key,
                        update_type='drift_adaptation',
                        samples_used=len(recent_X),
                        performance_before=old_model.get_recent_performance(),
                        performance_after=0.5,  # Unknown yet
                        drift_detected=True,
                        adaptation_strategy=self.current_strategy
                    )
                    
                    self.model_updates.append(update)
                    self._store_model_update(update)
                    
                    logger.info(f"Successfully adapted model {model_key} to concept drift")
            
        except Exception as e:
            logger.error(f"Error adapting to drift for {model_key}: {e}")
    
    def _store_sample(self, model_key: str, features: Dict, label: int,
                     prediction: float, confidence: float, loss: float):
        """Store sample in database"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO online_samples 
                (timestamp, model_key, features, label, prediction, confidence, loss)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                int(time.time()), model_key, json.dumps(features, cls=NumpyEncoder),
                label, prediction, confidence, loss
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing sample: {e}")
    
    def _store_model_update(self, update: ModelUpdate):
        """Store model update in database"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO model_updates 
                (timestamp, model_key, update_type, samples_used, performance_before,
                 performance_after, drift_detected, adaptation_strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(update.timestamp), update.model_key, update.update_type,
                update.samples_used, update.performance_before, update.performance_after,
                int(update.drift_detected), update.adaptation_strategy
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing model update: {e}")
    
    def get_model_stats(self, model_key: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a model"""
        try:
            model = self.online_models.get(model_key)
            if not model:
                return {'error': 'model_not_found'}
            
            # Check for concept drift
            drift_detected, drift_score = self.detect_concept_drift(model_key)
            
            return {
                'model_key': model_key,
                'is_fitted': model.is_fitted,
                'sample_count': model.sample_count,
                'recent_performance': model.get_recent_performance(),
                'performance_history_length': len(model.performance_history),
                'last_update_time': model.last_update_time,
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'buffer_size': len(self.feature_buffer[model_key]),
                'adaptation_strategy': self.current_strategy
            }
            
        except Exception as e:
            logger.error(f"Error getting model stats: {e}")
            return {'error': str(e)}
    
    def get_system_performance(self, days: int = 7) -> Dict[str, Any]:
        """Get overall system performance"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            
            cutoff_time = int(time.time()) - (days * 24 * 3600)
            
            # Get update statistics
            df_updates = pd.read_sql_query("""
                SELECT * FROM model_updates 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, conn, params=(cutoff_time,))
            
            conn.close()
            
            stats = {
                'period_days': days,
                'total_models': len(self.online_models),
                'fitted_models': sum(1 for m in self.online_models.values() if m.is_fitted),
                'total_updates': len(df_updates),
                'drift_adaptations': len(df_updates[df_updates['drift_detected'] == 1]),
                'current_strategy': self.current_strategy,
                'model_stats': {}
            }
            
            # Add individual model stats
            for model_key in self.online_models.keys():
                stats['model_stats'][model_key] = self.get_model_stats(model_key)
            
            if len(df_updates) > 0:
                stats['avg_performance_improvement'] = (
                    df_updates['performance_after'] - df_updates['performance_before']
                ).mean()
                stats['recent_updates'] = len(df_updates[df_updates['timestamp'] > cutoff_time])
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting system performance: {e}")
            return {'error': str(e)}
    
    def set_adaptation_strategy(self, strategy: str):
        """Set adaptation strategy"""
        if strategy in self.adaptation_strategies:
            self.current_strategy = strategy
            logger.info(f"Adaptation strategy set to: {strategy}")
        else:
            logger.warning(f"Unknown adaptation strategy: {strategy}")

# Global online learning system instance
online_learning_system = AdaptiveLearningSystem()

if __name__ == "__main__":
    import os
    
    # Test online learning system
    system = AdaptiveLearningSystem()
    
    print("Testing Online Learning System...")
    
    model_key = "R_100_DIGITEVEN"
    
    # Simulate training data
    np.random.seed(42)
    
    for i in range(100):
        # Generate synthetic features
        features = {
            'last_digit': np.random.randint(0, 10),
            'price_change': np.random.normal(0, 1),
            'volatility_5': np.random.exponential(1),
            'volatility_20': np.random.exponential(1),
            'streak_length': np.random.poisson(3),
            'streak_direction': np.random.choice([-1, 1]),
            'momentum_score': np.random.normal(0, 1),
            'digit_freq_even': np.random.beta(2, 2),
            'digit_freq_odd': np.random.beta(2, 2),
            'price_trend_5': np.random.normal(0, 0.5),
            'price_trend_20': np.random.normal(0, 0.5)
        }
        
        # Generate label based on features (synthetic relationship)
        prob = 0.5 + 0.1 * (features['last_digit'] % 2) + 0.05 * features['momentum_score']
        prob = np.clip(prob, 0.1, 0.9)
        label = int(np.random.random() < prob)
        
        # Add sample to system
        prediction, confidence = system.predict(model_key, features)
        system.add_sample(model_key, features, label, prediction, confidence)
        
        if i % 20 == 0 and i > 0:
            print(f"Step {i}: Added sample, prediction: {prediction:.3f}, actual: {label}")
    
    # Test concept drift detection
    drift_detected, drift_score = system.detect_concept_drift(model_key)
    print(f"Concept drift detected: {drift_detected} (score: {drift_score:.3f})")
    
    # Get model statistics
    stats = system.get_model_stats(model_key)
    print(f"Model stats: {stats}")
    
    # Test system performance
    perf = system.get_system_performance()
    print(f"System performance: {perf}")
    
    # Test prediction after training
    test_features = {
        'last_digit': 6,
        'price_change': 0.5,
        'volatility_5': 1.2,
        'volatility_20': 0.8,
        'streak_length': 2,
        'streak_direction': 1,
        'momentum_score': 0.3,
        'digit_freq_even': 0.6,
        'digit_freq_odd': 0.4,
        'price_trend_5': 0.1,
        'price_trend_20': -0.05
    }
    
    final_pred, final_conf = system.predict(model_key, test_features)
    print(f"Final prediction: {final_pred:.3f} (confidence: {final_conf:.3f})")
