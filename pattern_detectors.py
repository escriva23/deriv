# pattern_detectors.py - Advanced Pattern Detection and Feature Engineering
import numpy as np
import pandas as pd
import sqlite3
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque, Counter
from scipy import stats
from sklearn.preprocessing import StandardScaler
import time
import json
from config import config

logger = logging.getLogger(__name__)

class NGramPatternDetector:
    """N-gram pattern detection for digit sequences"""
    
    def __init__(self, max_n: int = 5, window_size: int = 1000):
        self.max_n = max_n
        self.window_size = window_size
        self.digit_history = deque(maxlen=window_size)
        self.ngram_counts = defaultdict(lambda: defaultdict(int))
        self.transition_matrices = {}
        
    def update(self, last_digit: int):
        """Update with new digit"""
        self.digit_history.append(last_digit)
        self._update_ngram_counts()
    
    def _update_ngram_counts(self):
        """Update n-gram count statistics"""
        if len(self.digit_history) < self.max_n + 1:
            return
        
        # Update n-gram counts for all n from 1 to max_n
        for n in range(1, min(self.max_n + 1, len(self.digit_history))):
            for i in range(len(self.digit_history) - n):
                context = tuple(list(self.digit_history)[i:i+n])
                next_digit = list(self.digit_history)[i+n]
                self.ngram_counts[n][context] += 1
    
    def get_next_digit_probabilities(self, context_length: int = 3) -> Dict[int, float]:
        """Get probability distribution for next digit given recent context"""
        if len(self.digit_history) < context_length:
            return {i: 0.1 for i in range(10)}  # Uniform prior
        
        # Get recent context
        context = tuple(list(self.digit_history)[-context_length:])
        
        # Count occurrences of this context followed by each digit
        digit_counts = defaultdict(int)
        total_count = 0
        
        for i in range(len(self.digit_history) - context_length):
            if tuple(list(self.digit_history)[i:i+context_length]) == context:
                if i + context_length < len(self.digit_history):
                    next_digit = list(self.digit_history)[i + context_length]
                    digit_counts[next_digit] += 1
                    total_count += 1
        
        if total_count == 0:
            return {i: 0.1 for i in range(10)}  # Uniform fallback
        
        # Convert to probabilities with smoothing
        probabilities = {}
        alpha = 0.1  # Laplace smoothing
        for digit in range(10):
            probabilities[digit] = (digit_counts[digit] + alpha) / (total_count + 10 * alpha)
        
        return probabilities
    
    def get_pattern_strength(self, context_length: int = 3) -> float:
        """Get strength of current pattern (entropy-based)"""
        probs = self.get_next_digit_probabilities(context_length)
        prob_values = list(probs.values())
        
        # Calculate entropy (lower entropy = stronger pattern)
        entropy = -sum(p * np.log2(p + 1e-10) for p in prob_values)
        max_entropy = np.log2(10)  # Maximum entropy for uniform distribution
        
        # Pattern strength = 1 - normalized_entropy
        return 1.0 - (entropy / max_entropy)
    
    def detect_repeating_patterns(self, min_length: int = 2, min_occurrences: int = 3) -> List[Tuple]:
        """Detect repeating patterns in recent history"""
        if len(self.digit_history) < min_length * min_occurrences:
            return []
        
        patterns = []
        history_list = list(self.digit_history)
        
        for pattern_length in range(min_length, min(10, len(history_list) // 2)):
            pattern_counts = Counter()
            
            for i in range(len(history_list) - pattern_length + 1):
                pattern = tuple(history_list[i:i + pattern_length])
                pattern_counts[pattern] += 1
            
            # Find patterns that occur frequently
            for pattern, count in pattern_counts.items():
                if count >= min_occurrences:
                    patterns.append((pattern, count, pattern_length))
        
        return sorted(patterns, key=lambda x: x[1], reverse=True)

class SequencePatternDetector:
    """Advanced sequence pattern detection"""
    
    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self.price_history = deque(maxlen=window_size)
        self.digit_history = deque(maxlen=window_size)
        self.parity_history = deque(maxlen=window_size)
        
    def update(self, price: float, last_digit: int):
        """Update with new price and digit data"""
        self.price_history.append(price)
        self.digit_history.append(last_digit)
        self.parity_history.append(last_digit % 2)
    
    def detect_streaks(self) -> Dict[str, Any]:
        """Detect various types of streaks"""
        if len(self.parity_history) < 5:
            return {}
        
        parity_list = list(self.parity_history)
        
        # Current parity streak
        current_parity = parity_list[-1]
        parity_streak = 1
        for i in range(len(parity_list) - 2, -1, -1):
            if parity_list[i] == current_parity:
                parity_streak += 1
            else:
                break
        
        # Price direction streak
        if len(self.price_history) >= 3:
            price_list = list(self.price_history)
            price_changes = [1 if price_list[i] > price_list[i-1] else 0 
                           for i in range(1, len(price_list))]
            
            current_direction = price_changes[-1]
            direction_streak = 1
            for i in range(len(price_changes) - 2, -1, -1):
                if price_changes[i] == current_direction:
                    direction_streak += 1
                else:
                    break
        else:
            direction_streak = 0
        
        # Alternating pattern detection
        alternating_score = self._detect_alternating_pattern()
        
        return {
            'parity_streak': parity_streak,
            'current_parity': current_parity,
            'direction_streak': direction_streak,
            'alternating_score': alternating_score,
            'streak_momentum': min(10, parity_streak) / 10.0  # Normalized
        }
    
    def _detect_alternating_pattern(self) -> float:
        """Detect alternating patterns (ABAB...)"""
        if len(self.parity_history) < 6:
            return 0.0
        
        parity_list = list(self.parity_history)[-10:]  # Look at last 10
        alternations = 0
        total_comparisons = len(parity_list) - 1
        
        for i in range(1, len(parity_list)):
            if parity_list[i] != parity_list[i-1]:
                alternations += 1
        
        return alternations / total_comparisons if total_comparisons > 0 else 0.0
    
    def detect_micro_patterns(self) -> Dict[str, float]:
        """Detect micro-patterns in recent data"""
        if len(self.digit_history) < 10:
            return {}
        
        recent_digits = list(self.digit_history)[-10:]
        
        # Ascending/descending trends
        ascending_score = self._calculate_trend_score(recent_digits, ascending=True)
        descending_score = self._calculate_trend_score(recent_digits, ascending=False)
        
        # Variance patterns
        variance_score = np.var(recent_digits) / 8.25  # Normalized by max variance
        
        # Range patterns
        digit_range = max(recent_digits) - min(recent_digits)
        range_score = digit_range / 9.0  # Normalized
        
        # Clustering score (how clustered are the digits)
        clustering_score = self._calculate_clustering_score(recent_digits)
        
        return {
            'ascending_trend': ascending_score,
            'descending_trend': descending_score,
            'variance_pattern': variance_score,
            'range_pattern': range_score,
            'clustering_pattern': clustering_score
        }
    
    def _calculate_trend_score(self, digits: List[int], ascending: bool = True) -> float:
        """Calculate trend score for digit sequence"""
        if len(digits) < 3:
            return 0.0
        
        trend_score = 0
        comparisons = 0
        
        for i in range(1, len(digits)):
            if ascending and digits[i] > digits[i-1]:
                trend_score += 1
            elif not ascending and digits[i] < digits[i-1]:
                trend_score += 1
            comparisons += 1
        
        return trend_score / comparisons if comparisons > 0 else 0.0
    
    def _calculate_clustering_score(self, digits: List[int]) -> float:
        """Calculate how clustered the digits are"""
        if len(digits) < 3:
            return 0.0
        
        # Calculate average distance between consecutive digits
        distances = [abs(digits[i] - digits[i-1]) for i in range(1, len(digits))]
        avg_distance = np.mean(distances)
        
        # Normalize (max average distance is 4.5 for random digits)
        return 1.0 - min(1.0, avg_distance / 4.5)

class HistogramPatternDetector:
    """Histogram-based pattern detection"""
    
    def __init__(self, windows: List[int] = [10, 20, 50, 100]):
        self.windows = windows
        self.digit_history = deque(maxlen=max(windows))
        
    def update(self, last_digit: int):
        """Update with new digit"""
        self.digit_history.append(last_digit)
    
    def get_histogram_features(self) -> Dict[str, float]:
        """Get histogram-based features for different windows"""
        features = {}
        
        for window in self.windows:
            if len(self.digit_history) < window:
                continue
            
            recent_digits = list(self.digit_history)[-window:]
            
            # Digit frequency distribution
            digit_counts = Counter(recent_digits)
            
            # Even/odd distribution
            even_count = sum(1 for d in recent_digits if d % 2 == 0)
            odd_count = window - even_count
            
            # High/low distribution (>5 vs <=5)
            high_count = sum(1 for d in recent_digits if d > 5)
            low_count = window - high_count
            
            # Statistical measures
            digit_array = np.array(recent_digits)
            mean_digit = np.mean(digit_array)
            std_digit = np.std(digit_array)
            skewness = stats.skew(digit_array)
            kurtosis = stats.kurtosis(digit_array)
            
            # Entropy (measure of randomness)
            entropy = self._calculate_entropy(digit_counts, window)
            
            # Chi-square test for uniformity
            expected = window / 10
            observed = [digit_counts.get(i, 0) for i in range(10)]
            chi2_stat = sum((obs - expected) ** 2 / expected for obs in observed if expected > 0)
            
            prefix = f"hist_{window}_"
            features.update({
                f"{prefix}even_ratio": even_count / window,
                f"{prefix}odd_ratio": odd_count / window,
                f"{prefix}high_ratio": high_count / window,
                f"{prefix}low_ratio": low_count / window,
                f"{prefix}mean": mean_digit / 9.0,  # Normalized
                f"{prefix}std": std_digit / 3.0,   # Normalized
                f"{prefix}skewness": np.clip(skewness / 2.0, -1, 1),  # Normalized
                f"{prefix}kurtosis": np.clip(kurtosis / 5.0, -1, 1),  # Normalized
                f"{prefix}entropy": entropy,
                f"{prefix}chi2_uniformity": min(1.0, chi2_stat / 50.0)  # Normalized
            })
        
        return features
    
    def _calculate_entropy(self, digit_counts: Counter, total: int) -> float:
        """Calculate entropy of digit distribution"""
        entropy = 0
        for count in digit_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        # Normalize by maximum entropy (log2(10))
        return entropy / np.log2(10)
    
    def detect_bias_patterns(self) -> Dict[str, float]:
        """Detect bias patterns in digit distribution"""
        if len(self.digit_history) < 20:
            return {}
        
        recent_digits = list(self.digit_history)[-50:]  # Use last 50 digits
        
        # Even/odd bias
        even_count = sum(1 for d in recent_digits if d % 2 == 0)
        even_bias = abs(even_count / len(recent_digits) - 0.5)
        
        # High/low bias
        high_count = sum(1 for d in recent_digits if d > 5)
        high_bias = abs(high_count / len(recent_digits) - 0.4)  # Expected: 4/10 = 0.4
        
        # Individual digit biases
        digit_counts = Counter(recent_digits)
        expected_count = len(recent_digits) / 10
        
        max_digit_bias = 0
        for digit in range(10):
            actual_count = digit_counts.get(digit, 0)
            bias = abs(actual_count - expected_count) / expected_count
            max_digit_bias = max(max_digit_bias, bias)
        
        # Sequential bias (preference for consecutive digits)
        sequential_pairs = 0
        for i in range(1, len(recent_digits)):
            if abs(recent_digits[i] - recent_digits[i-1]) == 1:
                sequential_pairs += 1
        
        sequential_bias = sequential_pairs / (len(recent_digits) - 1) if len(recent_digits) > 1 else 0
        
        return {
            'even_odd_bias': even_bias,
            'high_low_bias': high_bias,
            'max_digit_bias': min(1.0, max_digit_bias),
            'sequential_bias': sequential_bias
        }

class AdvancedPatternEngine:
    """Comprehensive pattern detection engine"""
    
    def __init__(self):
        self.ngram_detector = NGramPatternDetector()
        self.sequence_detector = SequencePatternDetector()
        self.histogram_detector = HistogramPatternDetector()
        self.feature_scaler = StandardScaler()
        self.is_scaler_fitted = False
        self.init_database()
        
    def init_database(self):
        """Initialize pattern tracking database"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            # Pattern features table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pattern_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    symbol TEXT,
                    last_digit INTEGER,
                    ngram_features TEXT,
                    sequence_features TEXT,
                    histogram_features TEXT,
                    pattern_strength REAL,
                    prediction_signal TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing pattern database: {e}")
    
    def update_patterns(self, price: float, last_digit: int, symbol: str) -> Dict[str, Any]:
        """Update all pattern detectors and return comprehensive features"""
        try:
            # Update all detectors
            self.ngram_detector.update(last_digit)
            self.sequence_detector.update(price, last_digit)
            self.histogram_detector.update(last_digit)
            
            # Get features from all detectors
            features = {}
            
            # N-gram features
            ngram_probs = self.ngram_detector.get_next_digit_probabilities()
            pattern_strength = self.ngram_detector.get_pattern_strength()
            repeating_patterns = self.ngram_detector.detect_repeating_patterns()
            
            features['ngram_even_prob'] = sum(ngram_probs.get(i, 0) for i in [0, 2, 4, 6, 8])
            features['ngram_odd_prob'] = sum(ngram_probs.get(i, 0) for i in [1, 3, 5, 7, 9])
            features['ngram_over5_prob'] = sum(ngram_probs.get(i, 0) for i in [6, 7, 8, 9])
            features['ngram_under5_prob'] = sum(ngram_probs.get(i, 0) for i in [0, 1, 2, 3, 4])
            features['ngram_pattern_strength'] = pattern_strength
            features['ngram_repeating_patterns'] = len(repeating_patterns)
            
            # Sequence features
            streak_features = self.sequence_detector.detect_streaks()
            micro_features = self.sequence_detector.detect_micro_patterns()
            
            features.update(streak_features)
            features.update(micro_features)
            
            # Histogram features
            hist_features = self.histogram_detector.get_histogram_features()
            bias_features = self.histogram_detector.detect_bias_patterns()
            
            features.update(hist_features)
            features.update(bias_features)
            
            # Generate trading signals
            signals = self._generate_pattern_signals(features)
            features.update(signals)
            
            # Store in database
            self._store_pattern_features(symbol, last_digit, features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error updating patterns: {e}")
            return {}
    
    def _generate_pattern_signals(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Generate trading signals based on pattern features"""
        signals = {}
        
        try:
            # Even/Odd signal based on multiple indicators
            even_indicators = [
                features.get('ngram_even_prob', 0.5) - 0.5,  # Deviation from expected
                0.5 - features.get('hist_50_even_ratio', 0.5),  # Reversion signal
                -features.get('even_odd_bias', 0) if features.get('hist_50_even_ratio', 0.5) < 0.5 else features.get('even_odd_bias', 0)
            ]
            
            signals['pattern_even_signal'] = np.mean(even_indicators)
            signals['pattern_odd_signal'] = -signals['pattern_even_signal']
            
            # Over/Under signal
            over_indicators = [
                features.get('ngram_over5_prob', 0.4) - 0.4,
                0.4 - features.get('hist_50_high_ratio', 0.4),
                -features.get('high_low_bias', 0) if features.get('hist_50_high_ratio', 0.4) < 0.4 else features.get('high_low_bias', 0)
            ]
            
            signals['pattern_over_signal'] = np.mean(over_indicators)
            signals['pattern_under_signal'] = -signals['pattern_over_signal']
            
            # Momentum signal based on streaks and trends
            momentum_indicators = [
                features.get('streak_momentum', 0) * (1 if features.get('current_parity', 0) == 1 else -1),
                features.get('ascending_trend', 0) - features.get('descending_trend', 0),
                features.get('direction_streak', 0) / 10.0  # Normalized
            ]
            
            signals['pattern_momentum_signal'] = np.mean(momentum_indicators)
            
            # Overall pattern confidence
            confidence_factors = [
                features.get('ngram_pattern_strength', 0),
                features.get('even_odd_bias', 0),
                features.get('high_low_bias', 0),
                1.0 - features.get('hist_50_entropy', 1.0)  # Lower entropy = higher confidence
            ]
            
            signals['pattern_confidence'] = np.mean(confidence_factors)
            
        except Exception as e:
            logger.error(f"Error generating pattern signals: {e}")
            signals = {
                'pattern_even_signal': 0.0,
                'pattern_odd_signal': 0.0,
                'pattern_over_signal': 0.0,
                'pattern_under_signal': 0.0,
                'pattern_momentum_signal': 0.0,
                'pattern_confidence': 0.0
            }
        
        return signals
    
    def _store_pattern_features(self, symbol: str, last_digit: int, features: Dict[str, Any]):
        """Store pattern features in database"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            # Separate features by type for storage
            ngram_features = {k: v for k, v in features.items() if k.startswith('ngram_')}
            sequence_features = {k: v for k, v in features.items() if k.startswith(('parity_', 'direction_', 'alternating', 'streak_', 'ascending', 'descending', 'variance', 'range', 'clustering'))}
            histogram_features = {k: v for k, v in features.items() if k.startswith('hist_') or k.endswith('_bias')}
            
            cursor.execute("""
                INSERT INTO pattern_features 
                (timestamp, symbol, last_digit, ngram_features, sequence_features, 
                 histogram_features, pattern_strength, prediction_signal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(time.time()), symbol, last_digit,
                json.dumps(ngram_features), json.dumps(sequence_features),
                json.dumps(histogram_features),
                features.get('pattern_confidence', 0),
                json.dumps({k: v for k, v in features.items() if k.startswith('pattern_')})
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing pattern features: {e}")
    
    def get_recent_pattern_features(self, symbol: str, count: int = 100) -> pd.DataFrame:
        """Get recent pattern features for analysis"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            
            df = pd.read_sql_query("""
                SELECT * FROM pattern_features 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, conn, params=(symbol, count))
            
            conn.close()
            return df
            
        except Exception as e:
            logger.error(f"Error getting recent pattern features: {e}")
            return pd.DataFrame()
    
    def get_pattern_performance(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """Analyze pattern detection performance"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            
            cutoff_time = int(time.time()) - (days * 24 * 3600)
            
            # Try to get performance data, but handle missing trades table gracefully
            try:
                df = pd.read_sql_query("""
                    SELECT pf.*, t.win, t.profit_loss
                    FROM pattern_features pf
                    LEFT JOIN trades t ON pf.timestamp BETWEEN t.entry_time - 5 AND t.entry_time + 5
                    WHERE pf.symbol = ? AND pf.timestamp > ?
                    ORDER BY pf.timestamp DESC
                """, conn, params=(symbol, cutoff_time))
            except sqlite3.OperationalError as e:
                if "no such table: trades" in str(e):
                    # Fallback: get pattern features without trade results
                    logger.warning("Trades table not found, analyzing patterns without performance data")
                    df = pd.read_sql_query("""
                        SELECT pf.*
                        FROM pattern_features pf
                        WHERE pf.symbol = ? AND pf.timestamp > ?
                        ORDER BY pf.timestamp DESC
                    """, conn, params=(symbol, cutoff_time))
                    df['win'] = None
                    df['profit_loss'] = None
                else:
                    raise e
            
            conn.close()
            
            if len(df) == 0:
                return {}
            
            # Analyze pattern signal accuracy
            performance = {
                'total_patterns': len(df),
                'patterns_with_trades': len(df[df['win'].notna()]),
                'avg_pattern_strength': df['pattern_strength'].mean(),
            }
            
            # Analyze signal accuracy for trades
            trades_df = df[df['win'].notna()]
            if len(trades_df) > 0:
                performance['pattern_win_rate'] = trades_df['win'].mean()
                performance['avg_profit'] = trades_df['profit_loss'].mean()
                
                # Analyze performance by pattern strength
                high_strength = trades_df[trades_df['pattern_strength'] > 0.7]
                if len(high_strength) > 0:
                    performance['high_strength_win_rate'] = high_strength['win'].mean()
                    performance['high_strength_trades'] = len(high_strength)
            
            return performance
            
        except Exception as e:
            logger.error(f"Error analyzing pattern performance: {e}")
            return {}

# Global pattern engine instance
pattern_engine = AdvancedPatternEngine()

if __name__ == "__main__":
    # Test pattern detection system
    engine = AdvancedPatternEngine()
    
    print("Testing pattern detection system...")
    
    # Simulate some market data
    np.random.seed(42)
    prices = [1000.0]
    
    for i in range(100):
        # Simulate price movement
        change = np.random.normal(0, 1)
        new_price = prices[-1] + change
        prices.append(new_price)
        
        # Extract last digit
        last_digit = int(str(new_price).replace('.', '')[-1])
        
        # Update patterns
        features = engine.update_patterns(new_price, last_digit, "R_100")
        
        if i % 20 == 0 and i > 0:
            print(f"\nStep {i}:")
            print(f"Price: {new_price:.2f}, Last digit: {last_digit}")
            print(f"Pattern confidence: {features.get('pattern_confidence', 0):.3f}")
            print(f"Even signal: {features.get('pattern_even_signal', 0):.3f}")
            print(f"Over signal: {features.get('pattern_over_signal', 0):.3f}")
            print(f"Pattern strength: {features.get('ngram_pattern_strength', 0):.3f}")
    
    # Get performance analysis
    performance = engine.get_pattern_performance("R_100")
    print(f"\nPattern Performance: {performance}")
