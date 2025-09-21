# meta_controller.py - Advanced Meta-Controller with EV, Uncertainty, and Drift Detection
import numpy as np
import pandas as pd
import sqlite3
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, defaultdict
from dataclasses import dataclass
from scipy import stats
import json
from config import config
from pattern_calibration import probability_calibrator
from pattern_detectors import pattern_engine

logger = logging.getLogger(__name__)

@dataclass
class TradingDecision:
    """Enhanced trading decision with meta-analysis"""
    contract_type: str
    probability: float
    calibrated_probability: float
    confidence: float
    expected_value: float
    uncertainty: float
    stake: float
    reasoning: str
    pattern_signals: Dict[str, float]
    model_agreement: float
    drift_detected: bool
    risk_adjusted_ev: float

class DriftDetector:
    """Advanced drift detection using multiple methods"""
    
    def __init__(self, window_size: int = 100, sensitivity: float = 0.05):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.win_rates = deque(maxlen=window_size)
        self.probabilities = deque(maxlen=window_size)
        self.outcomes = deque(maxlen=window_size)
        self.last_drift_time = 0
        self.drift_cooldown = 300  # 5 minutes
        
        # CUSUM parameters
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.cusum_threshold = 5.0
        
        # Page-Hinkley parameters
        self.ph_sum = 0.0
        self.ph_min = 0.0
        self.ph_threshold = 10.0
        
    def update(self, predicted_prob: float, actual_outcome: bool):
        """Update drift detection with new prediction-outcome pair"""
        self.probabilities.append(predicted_prob)
        self.outcomes.append(int(actual_outcome))
        
        if len(self.outcomes) >= 10:
            recent_win_rate = sum(list(self.outcomes)[-10:]) / 10
            self.win_rates.append(recent_win_rate)
            
            # Update CUSUM
            self._update_cusum(predicted_prob, actual_outcome)
            
            # Update Page-Hinkley
            self._update_page_hinkley(predicted_prob, actual_outcome)
    
    def _update_cusum(self, predicted_prob: float, actual_outcome: bool):
        """Update CUSUM drift detection"""
        # Expected vs actual difference
        error = int(actual_outcome) - predicted_prob
        
        # Update CUSUM statistics
        self.cusum_pos = max(0, self.cusum_pos + error - self.sensitivity)
        self.cusum_neg = min(0, self.cusum_neg + error + self.sensitivity)
    
    def _update_page_hinkley(self, predicted_prob: float, actual_outcome: bool):
        """Update Page-Hinkley drift detection"""
        # Calculate error
        error = int(actual_outcome) - predicted_prob
        
        # Update Page-Hinkley sum
        self.ph_sum += error
        self.ph_min = min(self.ph_min, self.ph_sum)
    
    def detect_drift(self) -> Tuple[bool, str, float]:
        """Detect if concept drift has occurred"""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_drift_time < self.drift_cooldown:
            return False, "cooldown", 0.0
        
        if len(self.win_rates) < 20:
            return False, "insufficient_data", 0.0
        
        # Method 1: CUSUM detection
        if abs(self.cusum_pos) > self.cusum_threshold or abs(self.cusum_neg) > self.cusum_threshold:
            self.last_drift_time = current_time
            self._reset_detectors()
            return True, "cusum", max(abs(self.cusum_pos), abs(self.cusum_neg))
        
        # Method 2: Page-Hinkley detection
        ph_diff = self.ph_sum - self.ph_min
        if ph_diff > self.ph_threshold:
            self.last_drift_time = current_time
            self._reset_detectors()
            return True, "page_hinkley", ph_diff
        
        # Method 3: Statistical test on recent vs historical performance
        if len(self.win_rates) >= 50:
            recent_rates = list(self.win_rates)[-20:]
            historical_rates = list(self.win_rates)[-50:-20]
            
            # Welch's t-test for difference in means
            t_stat, p_value = stats.ttest_ind(recent_rates, historical_rates, equal_var=False)
            
            if p_value < 0.05:  # Significant difference
                self.last_drift_time = current_time
                return True, "statistical", abs(t_stat)
        
        # Method 4: Variance change detection
        if len(self.win_rates) >= 40:
            recent_var = np.var(list(self.win_rates)[-20:])
            historical_var = np.var(list(self.win_rates)[-40:-20])
            
            # F-test for variance change
            f_stat = recent_var / historical_var if historical_var > 0 else 1.0
            
            if f_stat > 2.0 or f_stat < 0.5:  # Significant variance change
                self.last_drift_time = current_time
                return True, "variance_change", f_stat
        
        return False, "no_drift", 0.0
    
    def _reset_detectors(self):
        """Reset drift detection statistics"""
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.ph_sum = 0.0
        self.ph_min = 0.0
    
    def get_drift_stats(self) -> Dict[str, Any]:
        """Get current drift detection statistics"""
        return {
            'cusum_pos': self.cusum_pos,
            'cusum_neg': self.cusum_neg,
            'ph_diff': self.ph_sum - self.ph_min,
            'recent_win_rate': np.mean(list(self.win_rates)[-10:]) if len(self.win_rates) >= 10 else 0.0,
            'overall_win_rate': np.mean(self.win_rates) if self.win_rates else 0.0,
            'last_drift_time': self.last_drift_time,
            'samples': len(self.outcomes)
        }

class ModelEnsemble:
    """Enhanced model ensemble with agreement tracking"""
    
    def __init__(self):
        self.model_predictions = {}
        self.model_confidences = {}
        self.model_weights = defaultdict(lambda: 1.0)
        self.model_performance = defaultdict(lambda: deque(maxlen=100))
        
    def add_prediction(self, model_name: str, probability: float, confidence: float):
        """Add prediction from a model"""
        self.model_predictions[model_name] = probability
        self.model_confidences[model_name] = confidence
    
    def update_performance(self, model_name: str, predicted_prob: float, actual_outcome: bool):
        """Update model performance tracking"""
        # Calculate Brier score for this prediction
        brier_score = (predicted_prob - int(actual_outcome)) ** 2
        self.model_performance[model_name].append(1.0 - brier_score)  # Convert to accuracy-like metric
        
        # Update model weights based on recent performance
        if len(self.model_performance[model_name]) >= 10:
            recent_performance = np.mean(list(self.model_performance[model_name])[-10:])
            self.model_weights[model_name] = max(0.1, recent_performance)
    
    def get_ensemble_prediction(self) -> Tuple[float, float, float]:
        """Get weighted ensemble prediction with agreement measure"""
        if not self.model_predictions:
            return 0.5, 0.0, 0.0
        
        # Calculate weighted average
        total_weight = 0.0
        weighted_prob = 0.0
        weighted_conf = 0.0
        
        for model_name, prob in self.model_predictions.items():
            weight = self.model_weights[model_name]
            total_weight += weight
            weighted_prob += prob * weight
            weighted_conf += self.model_confidences.get(model_name, 0.5) * weight
        
        if total_weight == 0:
            return 0.5, 0.0, 0.0
        
        ensemble_prob = weighted_prob / total_weight
        ensemble_conf = weighted_conf / total_weight
        
        # Calculate model agreement (inverse of prediction variance)
        if len(self.model_predictions) > 1:
            pred_values = list(self.model_predictions.values())
            agreement = 1.0 - np.var(pred_values)  # High variance = low agreement
        else:
            agreement = 1.0
        
        return ensemble_prob, ensemble_conf, max(0.0, agreement)
    
    def clear_predictions(self):
        """Clear current predictions for next iteration"""
        self.model_predictions.clear()
        self.model_confidences.clear()

class MetaController:
    """Advanced meta-controller for trading decisions"""
    
    def __init__(self):
        self.drift_detector = DriftDetector()
        self.model_ensemble = ModelEnsemble()
        self.decision_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(list)
        self.init_database()
        
        # Configuration parameters
        self.min_probability = 0.55  # Minimum calibrated probability
        self.min_ev_threshold = 0.02  # Minimum expected value
        self.uncertainty_penalty = 2.0  # Penalty factor for uncertainty
        self.drift_penalty = 0.5  # Stake reduction when drift detected
        self.agreement_threshold = 0.3  # Minimum model agreement
        
    def init_database(self):
        """Initialize meta-controller database tables"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            # Enhanced trading decisions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS meta_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    symbol TEXT,
                    contract_type TEXT,
                    raw_probability REAL,
                    calibrated_probability REAL,
                    confidence REAL,
                    expected_value REAL,
                    risk_adjusted_ev REAL,
                    uncertainty REAL,
                    stake REAL,
                    model_agreement REAL,
                    drift_detected INTEGER,
                    pattern_signals TEXT,
                    reasoning TEXT,
                    executed INTEGER,
                    actual_outcome INTEGER,
                    profit_loss REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Drift detection events
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drift_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    symbol TEXT,
                    drift_method TEXT,
                    drift_strength REAL,
                    pre_drift_performance REAL,
                    post_drift_performance REAL,
                    action_taken TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing meta-controller database: {e}")
    
    def analyze_trading_opportunity(self, symbol: str, current_price: float, 
                                  last_digit: int, model_predictions: Dict[str, Dict],
                                  live_proposal: Dict = None) -> Optional[TradingDecision]:
        """Comprehensive analysis of trading opportunity"""
        try:
            # Clear previous ensemble predictions
            self.model_ensemble.clear_predictions()
            
            # Add model predictions to ensemble
            for model_name, pred_data in model_predictions.items():
                self.model_ensemble.add_prediction(
                    model_name, 
                    pred_data.get('probability', 0.5),
                    pred_data.get('confidence', 0.5)
                )
            
            # Get ensemble prediction
            ensemble_prob, ensemble_conf, model_agreement = self.model_ensemble.get_ensemble_prediction()
            
            # Update pattern engine
            pattern_features = pattern_engine.update_patterns(current_price, last_digit, symbol)
            
            # Get pattern signals
            pattern_signals = {k: v for k, v in pattern_features.items() if k.startswith('pattern_')}
            
            # Determine best contract type based on signals
            contract_candidates = []
            
            # Even/Odd analysis
            even_signal = pattern_signals.get('pattern_even_signal', 0)
            odd_signal = pattern_signals.get('pattern_odd_signal', 0)
            
            if even_signal > 0.1:
                contract_candidates.append(('DIGITEVEN', ensemble_prob + even_signal, ensemble_conf))
            if odd_signal > 0.1:
                contract_candidates.append(('DIGITODD', ensemble_prob + odd_signal, ensemble_conf))
            
            # Over/Under analysis
            over_signal = pattern_signals.get('pattern_over_signal', 0)
            under_signal = pattern_signals.get('pattern_under_signal', 0)
            
            if over_signal > 0.1:
                contract_candidates.append(('DIGITOVER', ensemble_prob + over_signal, ensemble_conf))
            if under_signal > 0.1:
                contract_candidates.append(('DIGITUNDER', ensemble_prob + under_signal, ensemble_conf))
            
            # Momentum analysis
            momentum_signal = pattern_signals.get('pattern_momentum_signal', 0)
            if abs(momentum_signal) > 0.1:
                if momentum_signal > 0:
                    contract_candidates.append(('CALL', ensemble_prob + abs(momentum_signal), ensemble_conf))
                else:
                    contract_candidates.append(('PUT', ensemble_prob + abs(momentum_signal), ensemble_conf))
            
            if not contract_candidates:
                return None
            
            # Select best candidate
            best_contract = max(contract_candidates, key=lambda x: x[1] * x[2])
            contract_type, raw_probability, confidence = best_contract
            
            # Calibrate probability
            model_key = f"{symbol}_{contract_type}"
            calibrated_prob = probability_calibrator.calibrate_probability(
                model_key, raw_probability
            )
            
            # Get uncertainty estimate
            uncertainty = probability_calibrator.get_uncertainty_estimate(model_key, raw_probability)
            
            # Check drift
            drift_detected, drift_method, drift_strength = self.drift_detector.detect_drift()
            
            # Calculate expected value
            if live_proposal:
                payout = live_proposal.get('payout', 0)
                ask_price = live_proposal.get('ask_price', 1.0)
                expected_value = (calibrated_prob * payout) - ask_price
            else:
                # Estimate payout (typical Deriv payout is ~1.8x stake)
                estimated_payout = 1.8
                expected_value = (calibrated_prob * estimated_payout) - 1.0
            
            # Apply uncertainty penalty
            uncertainty_adjusted_ev = expected_value - (uncertainty * self.uncertainty_penalty)
            
            # Apply drift penalty if detected
            if drift_detected:
                uncertainty_adjusted_ev *= (1.0 - self.drift_penalty)
            
            # Check decision thresholds
            if (calibrated_prob < self.min_probability or 
                uncertainty_adjusted_ev < self.min_ev_threshold or
                model_agreement < self.agreement_threshold):
                return None
            
            # Calculate stake
            stake = self._calculate_optimal_stake(
                calibrated_prob, uncertainty, confidence, drift_detected
            )
            
            # Create trading decision
            decision = TradingDecision(
                contract_type=contract_type,
                probability=raw_probability,
                calibrated_probability=calibrated_prob,
                confidence=confidence,
                expected_value=expected_value,
                uncertainty=uncertainty,
                stake=stake,
                reasoning=self._generate_reasoning(
                    contract_type, calibrated_prob, pattern_signals, 
                    model_agreement, drift_detected
                ),
                pattern_signals=pattern_signals,
                model_agreement=model_agreement,
                drift_detected=drift_detected,
                risk_adjusted_ev=uncertainty_adjusted_ev
            )
            
            # Store decision
            self._store_decision(symbol, decision)
            
            return decision
            
        except Exception as e:
            logger.error(f"Error analyzing trading opportunity: {e}")
            return None
    
    def _calculate_optimal_stake(self, probability: float, uncertainty: float, 
                               confidence: float, drift_detected: bool) -> float:
        """Calculate optimal stake using Kelly Criterion with adjustments"""
        try:
            # Base Kelly calculation
            payout_ratio = 0.8  # Typical Deriv payout ratio
            kelly_fraction = (probability * payout_ratio - (1 - probability)) / payout_ratio
            
            # Conservative Kelly (25% of full Kelly)
            conservative_kelly = max(0, kelly_fraction * 0.25)
            
            # Base stake from config
            base_stake = config.INITIAL_STAKE
            
            # Confidence scaling
            confidence_multiplier = max(0.5, min(2.0, confidence * 1.5))
            
            # Uncertainty penalty
            uncertainty_multiplier = max(0.3, 1.0 - uncertainty)
            
            # Drift penalty
            drift_multiplier = 0.5 if drift_detected else 1.0
            
            # Calculate final stake
            calculated_stake = (base_stake * conservative_kelly * 
                              confidence_multiplier * uncertainty_multiplier * 
                              drift_multiplier)
            
            # Apply bounds
            final_stake = max(config.MIN_STAKE, min(config.MAX_STAKE, calculated_stake))
            
            return final_stake
            
        except Exception as e:
            logger.error(f"Error calculating optimal stake: {e}")
            return config.MIN_STAKE
    
    def _generate_reasoning(self, contract_type: str, probability: float, 
                          pattern_signals: Dict, agreement: float, 
                          drift_detected: bool) -> str:
        """Generate human-readable reasoning for the decision"""
        reasons = []
        
        reasons.append(f"{contract_type} with {probability:.1%} calibrated probability")
        
        # Pattern reasoning
        if pattern_signals.get('pattern_confidence', 0) > 0.5:
            reasons.append(f"Strong pattern detected (confidence: {pattern_signals['pattern_confidence']:.2f})")
        
        # Model agreement
        if agreement > 0.7:
            reasons.append(f"High model agreement ({agreement:.2f})")
        elif agreement < 0.3:
            reasons.append(f"Low model agreement ({agreement:.2f}) - caution")
        
        # Drift detection
        if drift_detected:
            reasons.append("Concept drift detected - reduced stake")
        
        # Specific pattern signals
        for signal_name, signal_value in pattern_signals.items():
            if abs(signal_value) > 0.2:
                direction = "bullish" if signal_value > 0 else "bearish"
                reasons.append(f"{signal_name.replace('pattern_', '').replace('_signal', '')}: {direction}")
        
        return " | ".join(reasons)
    
    def _store_decision(self, symbol: str, decision: TradingDecision):
        """Store decision in database"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO meta_decisions 
                (timestamp, symbol, contract_type, raw_probability, calibrated_probability,
                 confidence, expected_value, risk_adjusted_ev, uncertainty, stake,
                 model_agreement, drift_detected, pattern_signals, reasoning, executed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(time.time()), symbol, decision.contract_type,
                decision.probability, decision.calibrated_probability,
                decision.confidence, decision.expected_value,
                decision.risk_adjusted_ev, decision.uncertainty, decision.stake,
                decision.model_agreement, int(decision.drift_detected),
                json.dumps(decision.pattern_signals), decision.reasoning, 0
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing decision: {e}")
    
    def update_decision_outcome(self, decision_id: int, actual_outcome: bool, 
                              profit_loss: float):
        """Update decision with actual outcome"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE meta_decisions 
                SET actual_outcome = ?, profit_loss = ?, executed = 1
                WHERE id = ?
            """, (int(actual_outcome), profit_loss, decision_id))
            
            conn.commit()
            conn.close()
            
            # Update drift detector
            cursor.execute("""
                SELECT calibrated_probability FROM meta_decisions WHERE id = ?
            """, (decision_id,))
            
            result = cursor.fetchone()
            if result:
                predicted_prob = result[0]
                self.drift_detector.update(predicted_prob, actual_outcome)
            
        except Exception as e:
            logger.error(f"Error updating decision outcome: {e}")
    
    def get_performance_metrics(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            
            cutoff_time = int(time.time()) - (days * 24 * 3600)
            
            df = pd.read_sql_query("""
                SELECT * FROM meta_decisions 
                WHERE symbol = ? AND timestamp > ? AND executed = 1
                ORDER BY timestamp DESC
            """, conn, params=(symbol, cutoff_time))
            
            conn.close()
            
            if len(df) == 0:
                return {}
            
            # Basic metrics
            total_trades = len(df)
            winning_trades = len(df[df['actual_outcome'] == 1])
            win_rate = winning_trades / total_trades
            
            # Profit metrics
            total_profit = df['profit_loss'].sum()
            avg_profit = df['profit_loss'].mean()
            
            # Calibration metrics
            predicted_probs = df['calibrated_probability'].values
            actual_outcomes = df['actual_outcome'].values
            
            # Brier score
            brier_score = np.mean((predicted_probs - actual_outcomes) ** 2)
            
            # Expected vs actual performance
            expected_wins = predicted_probs.sum()
            actual_wins = actual_outcomes.sum()
            calibration_error = abs(expected_wins - actual_wins) / len(predicted_probs)
            
            # Risk-adjusted metrics
            sharpe_ratio = avg_profit / df['profit_loss'].std() if df['profit_loss'].std() > 0 else 0
            
            # Pattern performance
            high_confidence_trades = df[df['confidence'] > 0.7]
            high_conf_win_rate = len(high_confidence_trades[high_confidence_trades['actual_outcome'] == 1]) / len(high_confidence_trades) if len(high_confidence_trades) > 0 else 0
            
            # Drift detection stats
            drift_stats = self.drift_detector.get_drift_stats()
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'avg_profit': avg_profit,
                'brier_score': brier_score,
                'calibration_error': calibration_error,
                'sharpe_ratio': sharpe_ratio,
                'high_confidence_win_rate': high_conf_win_rate,
                'high_confidence_trades': len(high_confidence_trades),
                'drift_stats': drift_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}

# Global meta-controller instance
meta_controller = MetaController()

if __name__ == "__main__":
    # Test meta-controller
    controller = MetaController()
    
    print("Testing meta-controller system...")
    
    # Simulate model predictions
    model_predictions = {
        'xgboost': {'probability': 0.65, 'confidence': 0.7},
        'random_forest': {'probability': 0.62, 'confidence': 0.6},
        'logistic': {'probability': 0.68, 'confidence': 0.8}
    }
    
    # Test decision making
    decision = controller.analyze_trading_opportunity(
        symbol="R_100",
        current_price=1234.56,
        last_digit=6,
        model_predictions=model_predictions
    )
    
    if decision:
        print(f"\nTrading Decision:")
        print(f"Contract: {decision.contract_type}")
        print(f"Calibrated Probability: {decision.calibrated_probability:.3f}")
        print(f"Expected Value: {decision.expected_value:.3f}")
        print(f"Risk-Adjusted EV: {decision.risk_adjusted_ev:.3f}")
        print(f"Stake: ${decision.stake:.2f}")
        print(f"Reasoning: {decision.reasoning}")
    else:
        print("No trading opportunity found")
    
    # Test drift detection
    for i in range(50):
        predicted = np.random.beta(2, 2)
        actual = np.random.random() < predicted
        controller.drift_detector.update(predicted, actual)
    
    drift_detected, method, strength = controller.drift_detector.detect_drift()
    print(f"\nDrift Detection: {drift_detected} ({method}, strength: {strength:.2f})")
    
    # Get performance metrics
    metrics = controller.get_performance_metrics("R_100")
    print(f"Performance Metrics: {metrics}")

