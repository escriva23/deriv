# pattern_calibration.py - Advanced Probability Calibration System
import numpy as np
import pandas as pd
import sqlite3
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy import stats
import time
from collections import deque
from config import config

logger = logging.getLogger(__name__)

class BayesianCalibrator:
    """Bayesian smoothing for probability calibration with sliding window updates"""
    
    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0, window_size: int = 100):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.window_size = window_size
        self.outcomes = deque(maxlen=window_size)
        self.predictions = deque(maxlen=window_size)
        
    def update(self, prediction: float, outcome: bool):
        """Update with new prediction-outcome pair"""
        self.predictions.append(prediction)
        self.outcomes.append(int(outcome))
        
    def calibrate(self, raw_probability: float) -> float:
        """Apply Bayesian smoothing to raw probability"""
        if len(self.outcomes) < 10:
            return raw_probability  # Not enough data for calibration
        
        # Calculate empirical win rate in window
        wins = sum(self.outcomes)
        total = len(self.outcomes)
        
        # Bayesian update: posterior = (alpha + wins) / (alpha + beta + total)
        alpha_post = self.alpha_prior + wins
        beta_post = self.beta_prior + (total - wins)
        
        # Blend with raw probability (weight recent data more)
        bayesian_prob = alpha_post / (alpha_post + beta_post)
        
        # Weighted combination (70% calibrated, 30% Bayesian)
        return 0.7 * raw_probability + 0.3 * bayesian_prob
    
    def get_confidence_interval(self, confidence_level: float = 0.95) -> Tuple[float, float]:
        """Get confidence interval for current probability estimate"""
        if len(self.outcomes) < 10:
            return (0.0, 1.0)
        
        wins = sum(self.outcomes)
        total = len(self.outcomes)
        alpha_post = self.alpha_prior + wins
        beta_post = self.beta_prior + (total - wins)
        
        # Beta distribution confidence interval
        from scipy.stats import beta
        lower = beta.ppf((1 - confidence_level) / 2, alpha_post, beta_post)
        upper = beta.ppf(1 - (1 - confidence_level) / 2, alpha_post, beta_post)
        
        return (lower, upper)

class ProbabilityCalibrator:
    """Advanced probability calibration with multiple methods"""
    
    def __init__(self):
        self.platt_calibrators = {}  # Per model/symbol
        self.isotonic_calibrators = {}
        self.bayesian_calibrators = {}
        self.calibration_data = {}
        self.init_database()
        
    def init_database(self):
        """Initialize calibration tracking database"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            # Calibration history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS calibration_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_key TEXT,
                    raw_probability REAL,
                    calibrated_probability REAL,
                    actual_outcome INTEGER,
                    timestamp INTEGER,
                    symbol TEXT,
                    contract_type TEXT,
                    features TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Calibration performance metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS calibration_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_key TEXT,
                    brier_score REAL,
                    log_loss REAL,
                    calibration_error REAL,
                    reliability_score REAL,
                    sharpness_score REAL,
                    sample_size INTEGER,
                    timestamp INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing calibration database: {e}")
    
    def collect_calibration_data(self, model_key: str, raw_prob: float, 
                                outcome: bool, symbol: str, contract_type: str,
                                features: Dict = None) -> None:
        """Collect data for calibration training"""
        try:
            # Store in database
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO calibration_history 
                (model_key, raw_probability, calibrated_probability, actual_outcome, 
                 timestamp, symbol, contract_type, features)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_key, raw_prob, raw_prob,  # Will update calibrated later
                int(outcome), int(time.time()), symbol, contract_type,
                str(features) if features else None
            ))
            
            conn.commit()
            conn.close()
            
            # Update Bayesian calibrator
            if model_key not in self.bayesian_calibrators:
                self.bayesian_calibrators[model_key] = BayesianCalibrator()
            
            self.bayesian_calibrators[model_key].update(raw_prob, outcome)
            
        except Exception as e:
            logger.error(f"Error collecting calibration data: {e}")
    
    def train_calibrators(self, model_key: str, min_samples: int = 100) -> bool:
        """Train calibration models for a specific model key"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            
            # Get calibration data
            df = pd.read_sql_query("""
                SELECT raw_probability, actual_outcome, timestamp
                FROM calibration_history 
                WHERE model_key = ?
                ORDER BY timestamp DESC
                LIMIT 10000
            """, conn, params=(model_key,))
            
            conn.close()
            
            if len(df) < min_samples:
                logger.info(f"Insufficient calibration data for {model_key}: {len(df)} samples")
                return False
            
            X = df['raw_probability'].values.reshape(-1, 1)
            y = df['actual_outcome'].values
            
            # Split data (use recent 20% for validation)
            split_idx = int(len(df) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train Platt scaling (logistic regression)
            platt_calibrator = LogisticRegression()
            platt_calibrator.fit(X_train, y_train)
            self.platt_calibrators[model_key] = platt_calibrator
            
            # Train Isotonic regression
            isotonic_calibrator = IsotonicRegression(out_of_bounds='clip')
            isotonic_calibrator.fit(X_train.ravel(), y_train)
            self.isotonic_calibrators[model_key] = isotonic_calibrator
            
            # Evaluate calibration performance
            self._evaluate_calibration(model_key, X_val, y_val)
            
            logger.info(f"Trained calibrators for {model_key} with {len(df)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training calibrators for {model_key}: {e}")
            return False
    
    def _evaluate_calibration(self, model_key: str, X_val: np.ndarray, y_val: np.ndarray):
        """Evaluate calibration performance"""
        try:
            # Get predictions from different calibrators
            platt_probs = self.platt_calibrators[model_key].predict_proba(X_val)[:, 1]
            isotonic_probs = self.isotonic_calibrators[model_key].predict(X_val.ravel())
            
            # Calculate metrics
            brier_score = np.mean((platt_probs - y_val) ** 2)
            log_loss = -np.mean(y_val * np.log(platt_probs + 1e-15) + 
                               (1 - y_val) * np.log(1 - platt_probs + 1e-15))
            
            # Calibration error (Expected Calibration Error)
            calibration_error = self._calculate_ece(platt_probs, y_val)
            
            # Store metrics
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO calibration_metrics 
                (model_key, brier_score, log_loss, calibration_error, 
                 reliability_score, sharpness_score, sample_size, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_key, brier_score, log_loss, calibration_error,
                0.0, 0.0,  # Placeholder for additional metrics
                len(y_val), int(time.time())
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error evaluating calibration: {e}")
    
    def _calculate_ece(self, probabilities: np.ndarray, outcomes: np.ndarray, 
                      n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = outcomes[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def calibrate_probability(self, model_key: str, raw_probability: float, 
                            method: str = 'ensemble') -> float:
        """Calibrate raw probability using trained calibrators"""
        try:
            # Ensure probability is in valid range
            raw_probability = np.clip(raw_probability, 0.001, 0.999)
            
            if method == 'platt' and model_key in self.platt_calibrators:
                calibrated = self.platt_calibrators[model_key].predict_proba(
                    [[raw_probability]])[0, 1]
                
            elif method == 'isotonic' and model_key in self.isotonic_calibrators:
                calibrated = self.isotonic_calibrators[model_key].predict([raw_probability])[0]
                
            elif method == 'bayesian' and model_key in self.bayesian_calibrators:
                calibrated = self.bayesian_calibrators[model_key].calibrate(raw_probability)
                
            elif method == 'ensemble':
                # Ensemble of available calibrators
                calibrated_probs = []
                
                if model_key in self.platt_calibrators:
                    platt_prob = self.platt_calibrators[model_key].predict_proba(
                        [[raw_probability]])[0, 1]
                    calibrated_probs.append(platt_prob)
                
                if model_key in self.isotonic_calibrators:
                    isotonic_prob = self.isotonic_calibrators[model_key].predict([raw_probability])[0]
                    calibrated_probs.append(isotonic_prob)
                
                if model_key in self.bayesian_calibrators:
                    bayesian_prob = self.bayesian_calibrators[model_key].calibrate(raw_probability)
                    calibrated_probs.append(bayesian_prob)
                
                if calibrated_probs:
                    calibrated = np.mean(calibrated_probs)
                else:
                    calibrated = raw_probability  # Fallback to raw
            else:
                calibrated = raw_probability  # No calibration available
            
            # Ensure result is in valid range
            calibrated = np.clip(calibrated, 0.001, 0.999)
            
            return float(calibrated)
            
        except Exception as e:
            logger.error(f"Error calibrating probability: {e}")
            return raw_probability
    
    def get_uncertainty_estimate(self, model_key: str, raw_probability: float) -> float:
        """Get uncertainty estimate for the probability"""
        try:
            if model_key in self.bayesian_calibrators:
                lower, upper = self.bayesian_calibrators[model_key].get_confidence_interval()
                uncertainty = (upper - lower) / 2  # Half-width of confidence interval
                return float(uncertainty)
            else:
                # Fallback: use distance from 0.5 as inverse uncertainty measure
                return float(abs(0.5 - raw_probability))
                
        except Exception as e:
            logger.error(f"Error getting uncertainty estimate: {e}")
            return 0.5  # Maximum uncertainty
    
    def should_retrain(self, model_key: str, threshold_samples: int = 500) -> bool:
        """Check if calibrator should be retrained"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            # Get last training timestamp
            cursor.execute("""
                SELECT MAX(timestamp) FROM calibration_metrics 
                WHERE model_key = ?
            """, (model_key,))
            
            last_training = cursor.fetchone()[0] or 0
            
            # Get new samples since last training
            cursor.execute("""
                SELECT COUNT(*) FROM calibration_history 
                WHERE model_key = ? AND timestamp > ?
            """, (model_key, last_training))
            
            new_samples = cursor.fetchone()[0] or 0
            conn.close()
            
            return new_samples >= threshold_samples
            
        except Exception as e:
            logger.error(f"Error checking retrain status: {e}")
            return False
    
    def get_calibration_stats(self, model_key: str) -> Dict:
        """Get calibration statistics for a model"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            
            # Get recent calibration metrics
            df_metrics = pd.read_sql_query("""
                SELECT * FROM calibration_metrics 
                WHERE model_key = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, conn, params=(model_key,))
            
            # Get recent calibration data
            df_data = pd.read_sql_query("""
                SELECT raw_probability, actual_outcome 
                FROM calibration_history 
                WHERE model_key = ? 
                ORDER BY timestamp DESC 
                LIMIT 1000
            """, conn, params=(model_key,))
            
            conn.close()
            
            stats = {
                'model_key': model_key,
                'has_calibrators': model_key in self.platt_calibrators,
                'sample_size': len(df_data),
                'recent_win_rate': df_data['actual_outcome'].mean() if len(df_data) > 0 else 0,
                'avg_raw_probability': df_data['raw_probability'].mean() if len(df_data) > 0 else 0,
            }
            
            if len(df_metrics) > 0:
                latest_metrics = df_metrics.iloc[0]
                stats.update({
                    'brier_score': latest_metrics['brier_score'],
                    'log_loss': latest_metrics['log_loss'],
                    'calibration_error': latest_metrics['calibration_error'],
                    'last_training': latest_metrics['timestamp']
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting calibration stats: {e}")
            return {'model_key': model_key, 'error': str(e)}

# Global calibrator instance
probability_calibrator = ProbabilityCalibrator()

if __name__ == "__main__":
    # Test calibration system
    calibrator = ProbabilityCalibrator()
    
    # Simulate some calibration data
    model_key = "test_model"
    
    print("Testing probability calibration system...")
    
    # Simulate training data
    np.random.seed(42)
    for i in range(200):
        raw_prob = np.random.beta(2, 2)  # Random probability
        actual_outcome = np.random.random() < raw_prob  # True outcome
        
        calibrator.collect_calibration_data(
            model_key, raw_prob, actual_outcome, "R_100", "DIGITEVEN"
        )
    
    # Train calibrators
    success = calibrator.train_calibrators(model_key)
    print(f"Training successful: {success}")
    
    # Test calibration
    test_prob = 0.7
    calibrated = calibrator.calibrate_probability(model_key, test_prob)
    uncertainty = calibrator.get_uncertainty_estimate(model_key, test_prob)
    
    print(f"Raw probability: {test_prob}")
    print(f"Calibrated probability: {calibrated:.4f}")
    print(f"Uncertainty: {uncertainty:.4f}")
    
    # Get stats
    stats = calibrator.get_calibration_stats(model_key)
    print(f"Calibration stats: {stats}")

