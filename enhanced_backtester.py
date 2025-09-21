# enhanced_backtester.py - Advanced Backtesting with Statistical Significance and Calibration
import numpy as np
import pandas as pd
import sqlite3
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, confusion_matrix
# Optional plotting dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    logger.warning("Matplotlib/seaborn not available - plotting features disabled")
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from config import config
from pattern_calibration import ProbabilityCalibrator
from pattern_detectors import AdvancedPatternEngine
from meta_controller import MetaController
from martingale_system import MartingaleRecoverySystem
from online_learning import AdaptiveLearningSystem

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Enhanced backtest configuration"""
    start_date: str = None
    end_date: str = None
    initial_balance: float = 1000.0
    symbols: List[str] = None
    use_pattern_detection: bool = True
    use_calibration: bool = True
    use_meta_controller: bool = True
    use_martingale: bool = False
    use_online_learning: bool = True
    min_confidence: float = 0.6
    max_daily_trades: int = 50
    statistical_significance_level: float = 0.05
    bootstrap_samples: int = 1000
    walk_forward_windows: int = 10

@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    config: BacktestConfig
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    
    # Statistical significance metrics
    statistical_significance: Dict[str, Any]
    
    # Calibration metrics
    calibration_metrics: Dict[str, Any]
    
    # Pattern analysis
    pattern_analysis: Dict[str, Any]
    
    # Risk metrics
    risk_metrics: Dict[str, Any]
    
    # Trade analysis
    trade_analysis: Dict[str, Any]
    
    # Performance by time periods
    temporal_analysis: Dict[str, Any]

class EnhancedBacktester:
    """Advanced backtesting framework with statistical rigor"""
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.calibrator = ProbabilityCalibrator()
        self.pattern_engine = AdvancedPatternEngine()
        self.meta_controller = MetaController()
        self.martingale_system = MartingaleRecoverySystem()
        self.online_system = AdaptiveLearningSystem()
        self.results_cache = {}
        
    def run_comprehensive_backtest(self, data_source: str = "database") -> BacktestResults:
        """Run comprehensive backtest with all enhancements"""
        logger.info("Starting comprehensive backtest...")
        
        # Load data
        if data_source == "database":
            data = self._load_data_from_database()
        else:
            data = self._load_data_from_file(data_source)
        
        if data.empty:
            raise ValueError("No data available for backtesting")
        
        # Initialize systems
        self._initialize_systems()
        
        # Run walk-forward analysis if configured
        if self.config.walk_forward_windows > 1:
            results = self._run_walk_forward_backtest(data)
        else:
            results = self._run_single_backtest(data)
        
        # Calculate statistical significance
        results.statistical_significance = self._calculate_statistical_significance(results)
        
        # Analyze calibration quality
        results.calibration_metrics = self._analyze_calibration_quality(results)
        
        # Pattern analysis
        results.pattern_analysis = self._analyze_pattern_performance(results)
        
        # Risk analysis
        results.risk_metrics = self._calculate_risk_metrics(results)
        
        # Trade analysis
        results.trade_analysis = self._analyze_trade_characteristics(results)
        
        # Temporal analysis
        results.temporal_analysis = self._analyze_temporal_patterns(results)
        
        logger.info(f"Backtest completed: {results.total_trades} trades, "
                   f"{results.win_rate:.2%} win rate, {results.total_return:.2%} return")
        
        return results
    
    def _load_data_from_database(self) -> pd.DataFrame:
        """Load historical data from database"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            
            # Build query
            query = """
                SELECT t.*, tf.* FROM ticks t
                LEFT JOIN tick_features tf ON t.epoch = tf.timestamp AND t.symbol = tf.symbol
                WHERE 1=1
            """
            params = []
            
            if self.config.symbols:
                placeholders = ','.join(['?' for _ in self.config.symbols])
                query += f" AND t.symbol IN ({placeholders})"
                params.extend(self.config.symbols)
            
            if self.config.start_date:
                query += " AND t.timestamp >= ?"
                params.append(int(pd.Timestamp(self.config.start_date).timestamp()))
            
            if self.config.end_date:
                query += " AND t.timestamp <= ?"
                params.append(int(pd.Timestamp(self.config.end_date).timestamp()))
            
            query += " ORDER BY t.symbol, t.epoch"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            logger.info(f"Loaded {len(df)} historical records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            return pd.DataFrame()
    
    def _initialize_systems(self):
        """Initialize all trading systems"""
        # Set initial balance for systems that need it
        self.martingale_system.update_balance(self.config.initial_balance)
        
        # Initialize pattern engine with some historical data
        logger.info("Initializing trading systems...")
    
    def _run_single_backtest(self, data: pd.DataFrame) -> BacktestResults:
        """Run single-period backtest"""
        trades = []
        balance_history = [self.config.initial_balance]
        current_balance = self.config.initial_balance
        
        # Group by symbol for processing
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].reset_index(drop=True)
            
            logger.info(f"Processing {len(symbol_data)} ticks for {symbol}")
            
            # Process each tick
            for i in range(50, len(symbol_data) - 1):  # Start after enough history
                current_tick = symbol_data.iloc[i]
                next_tick = symbol_data.iloc[i + 1]
                
                # Skip if missing required data
                if pd.isna(current_tick.get('last_digit')):
                    continue
                
                # Prepare features
                features = self._extract_features(current_tick, symbol_data.iloc[:i+1])
                
                # Update pattern engine
                if self.config.use_pattern_detection:
                    pattern_features = self.pattern_engine.update_patterns(
                        current_tick['quote'], int(current_tick['last_digit']), symbol
                    )
                    features.update(pattern_features)
                
                # Get model predictions (simplified for backtest)
                model_predictions = self._simulate_model_predictions(features)
                
                # Use meta-controller for decision making
                if self.config.use_meta_controller:
                    decision = self.meta_controller.analyze_trading_opportunity(
                        symbol, current_tick['quote'], int(current_tick['last_digit']),
                        model_predictions
                    )
                else:
                    decision = self._simple_decision_logic(features, model_predictions)
                
                if decision is None:
                    continue
                
                # Check daily trade limit
                today_trades = len([t for t in trades if 
                                  pd.Timestamp(t['datetime']).date() == current_tick['datetime'].date()])
                
                if today_trades >= self.config.max_daily_trades:
                    continue
                
                # Simulate trade execution
                trade_result = self._simulate_trade(
                    decision, current_tick, next_tick, current_balance
                )
                
                if trade_result:
                    trades.append(trade_result)
                    current_balance += trade_result['profit_loss']
                    balance_history.append(current_balance)
                    
                    # Update systems with trade result
                    self._update_systems_with_result(decision, trade_result)
        
        # Create results object
        results = self._create_results_object(trades, balance_history)
        return results
    
    def _run_walk_forward_backtest(self, data: pd.DataFrame) -> BacktestResults:
        """Run walk-forward analysis"""
        logger.info(f"Running walk-forward analysis with {self.config.walk_forward_windows} windows")
        
        # Split data into windows
        data_length = len(data)
        window_size = data_length // self.config.walk_forward_windows
        
        all_trades = []
        all_balances = []
        window_results = []
        
        for window_idx in range(self.config.walk_forward_windows):
            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, data_length)
            
            if end_idx - start_idx < 100:  # Skip small windows
                continue
            
            window_data = data.iloc[start_idx:end_idx].reset_index(drop=True)
            
            logger.info(f"Processing window {window_idx + 1}/{self.config.walk_forward_windows}")
            
            # Run backtest on window
            window_config = self.config
            window_config.initial_balance = all_balances[-1] if all_balances else self.config.initial_balance
            
            window_backtest = EnhancedBacktester(window_config)
            window_result = window_backtest._run_single_backtest(window_data)
            
            window_results.append(window_result)
            all_trades.extend(window_result.trade_analysis['trades'])
            
            if all_balances:
                # Continue from previous window's final balance
                new_balances = [b + all_balances[-1] - window_config.initial_balance 
                              for b in window_result.trade_analysis['balance_history']]
                all_balances.extend(new_balances[1:])  # Skip first duplicate
            else:
                all_balances.extend(window_result.trade_analysis['balance_history'])
        
        # Combine results
        combined_results = self._combine_window_results(window_results, all_trades, all_balances)
        combined_results.temporal_analysis['walk_forward_windows'] = window_results
        
        return combined_results
    
    def _extract_features(self, current_tick: pd.Series, history: pd.DataFrame) -> Dict[str, Any]:
        """Extract features for prediction"""
        features = {
            'last_digit': int(current_tick.get('last_digit', 0)),
            'price_change': float(current_tick.get('price_change', 0)),
            'volatility_5': float(current_tick.get('volatility_5', 0)),
            'volatility_20': float(current_tick.get('volatility_20', 0)),
            'streak_length': int(current_tick.get('streak_length', 0)),
            'streak_direction': int(current_tick.get('streak_direction', 0)),
            'momentum_score': float(current_tick.get('momentum_score', 0)),
            'digit_parity': current_tick.get('digit_parity', 'even')
        }
        
        # Add derived features
        features['digit_freq_even'] = 0.5  # Placeholder
        features['digit_freq_odd'] = 0.5   # Placeholder
        features['price_trend_5'] = 0.0    # Placeholder
        features['price_trend_20'] = 0.0   # Placeholder
        
        return features
    
    def _simulate_model_predictions(self, features: Dict[str, Any]) -> Dict[str, Dict]:
        """Simulate model predictions for backtesting"""
        # Simplified model simulation - in real backtest, use actual trained models
        base_prob = 0.5
        
        # Add some feature-based adjustments
        if features['last_digit'] % 2 == 0:
            even_prob = base_prob + 0.05 * np.random.normal(0, 0.1)
        else:
            even_prob = base_prob - 0.05 * np.random.normal(0, 0.1)
        
        even_prob = np.clip(even_prob, 0.1, 0.9)
        
        return {
            'xgboost': {
                'probability': even_prob,
                'confidence': 0.6 + 0.2 * abs(even_prob - 0.5)
            },
            'random_forest': {
                'probability': even_prob + np.random.normal(0, 0.05),
                'confidence': 0.5 + 0.3 * abs(even_prob - 0.5)
            }
        }
    
    def _simple_decision_logic(self, features: Dict, model_predictions: Dict) -> Optional[Dict]:
        """Simple decision logic for when meta-controller is disabled"""
        # Get best prediction
        best_model = max(model_predictions.items(), 
                        key=lambda x: x[1]['confidence'])
        
        model_name, pred_data = best_model
        probability = pred_data['probability']
        confidence = pred_data['confidence']
        
        if confidence < self.config.min_confidence:
            return None
        
        # Determine contract type
        if probability > 0.55:
            contract_type = 'DIGITEVEN'
        elif probability < 0.45:
            contract_type = 'DIGITODD'
        else:
            return None
        
        return {
            'contract_type': contract_type,
            'probability': probability,
            'calibrated_probability': probability,  # No calibration in simple mode
            'confidence': confidence,
            'stake': 2.0,  # Fixed stake
            'expected_value': 0.05,  # Placeholder
            'reasoning': f"Simple logic: {model_name} prediction"
        }
    
    def _simulate_trade(self, decision: Dict, current_tick: pd.Series, 
                       next_tick: pd.Series, current_balance: float) -> Optional[Dict]:
        """Simulate trade execution and outcome"""
        try:
            contract_type = decision['contract_type']
            stake = min(decision['stake'], current_balance * 0.1)  # Risk limit
            
            if stake < 0.35:  # Minimum stake
                return None
            
            # Determine actual outcome
            next_digit = int(str(next_tick['quote']).replace('.', '')[-1])
            
            if contract_type == 'DIGITEVEN':
                win = (next_digit % 2 == 0)
            elif contract_type == 'DIGITODD':
                win = (next_digit % 2 == 1)
            elif contract_type == 'DIGITOVER':
                win = (next_digit > 5)
            elif contract_type == 'DIGITUNDER':
                win = (next_digit < 5)
            else:
                return None
            
            # Calculate profit/loss
            if win:
                payout_ratio = 0.8  # Typical Deriv payout
                profit = stake * payout_ratio
            else:
                profit = -stake
            
            return {
                'timestamp': current_tick['timestamp'],
                'datetime': current_tick['datetime'],
                'symbol': current_tick['symbol'],
                'contract_type': contract_type,
                'stake': stake,
                'predicted_probability': decision['probability'],
                'calibrated_probability': decision.get('calibrated_probability', decision['probability']),
                'confidence': decision['confidence'],
                'actual_outcome': int(win),
                'profit_loss': profit,
                'win': win,
                'entry_quote': current_tick['quote'],
                'exit_quote': next_tick['quote'],
                'exit_digit': next_digit,
                'reasoning': decision.get('reasoning', ''),
                'expected_value': decision.get('expected_value', 0)
            }
            
        except Exception as e:
            logger.error(f"Error simulating trade: {e}")
            return None
    
    def _update_systems_with_result(self, decision: Dict, trade_result: Dict):
        """Update systems with trade result"""
        if self.config.use_calibration:
            # Update calibration system
            model_key = f"{trade_result['symbol']}_{trade_result['contract_type']}"
            self.calibrator.collect_calibration_data(
                model_key, 
                trade_result['predicted_probability'],
                trade_result['win'],
                trade_result['symbol'],
                trade_result['contract_type']
            )
        
        if self.config.use_online_learning:
            # Update online learning system
            features = {'last_digit': trade_result['exit_digit']}  # Simplified
            model_key = f"{trade_result['symbol']}_{trade_result['contract_type']}"
            
            self.online_system.add_sample(
                model_key, features, trade_result['actual_outcome'],
                trade_result['predicted_probability'], trade_result['confidence']
            )
    
    def _create_results_object(self, trades: List[Dict], balance_history: List[float]) -> BacktestResults:
        """Create comprehensive results object"""
        if not trades:
            # Return empty results
            return BacktestResults(
                config=self.config,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                profit_factor=0.0,
                statistical_significance={},
                calibration_metrics={},
                pattern_analysis={},
                risk_metrics={},
                trade_analysis={'trades': [], 'balance_history': balance_history},
                temporal_analysis={}
            )
        
        df_trades = pd.DataFrame(trades)
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len(df_trades[df_trades['win'] == True])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades
        
        # Return metrics
        initial_balance = self.config.initial_balance
        final_balance = balance_history[-1]
        total_return = (final_balance - initial_balance) / initial_balance
        
        # Risk metrics
        returns = np.diff(balance_history) / balance_history[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Drawdown
        peak = np.maximum.accumulate(balance_history)
        drawdown = (np.array(balance_history) - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Profit factor
        gross_profit = df_trades[df_trades['profit_loss'] > 0]['profit_loss'].sum()
        gross_loss = abs(df_trades[df_trades['profit_loss'] < 0]['profit_loss'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return BacktestResults(
            config=self.config,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            statistical_significance={},
            calibration_metrics={},
            pattern_analysis={},
            risk_metrics={},
            trade_analysis={'trades': trades, 'balance_history': balance_history},
            temporal_analysis={}
        )
    
    def _calculate_statistical_significance(self, results: BacktestResults) -> Dict[str, Any]:
        """Calculate statistical significance of results"""
        if results.total_trades < 30:
            return {'error': 'insufficient_trades', 'min_required': 30}
        
        trades = results.trade_analysis['trades']
        df_trades = pd.DataFrame(trades)
        
        # Test 1: Win rate vs random (50%)
        wins = results.winning_trades
        total = results.total_trades
        
        # Binomial test
        p_value_winrate = stats.binom_test(wins, total, 0.5, alternative='two-sided')
        
        # Test 2: Mean return vs zero
        returns = df_trades['profit_loss'] / df_trades['stake']  # Normalized returns
        t_stat, p_value_returns = stats.ttest_1samp(returns, 0)
        
        # Test 3: Bootstrap confidence intervals
        bootstrap_returns = []
        np.random.seed(42)
        
        for _ in range(self.config.bootstrap_samples):
            sample_indices = np.random.choice(len(returns), size=len(returns), replace=True)
            sample_returns = returns.iloc[sample_indices]
            bootstrap_returns.append(sample_returns.mean())
        
        bootstrap_returns = np.array(bootstrap_returns)
        ci_lower = np.percentile(bootstrap_returns, 2.5)
        ci_upper = np.percentile(bootstrap_returns, 97.5)
        
        # Test 4: Sharpe ratio significance
        sharpe_pvalue = None
        if results.sharpe_ratio != 0:
            # Approximate test for Sharpe ratio
            n = len(returns)
            sharpe_stat = results.sharpe_ratio * np.sqrt(n)
            sharpe_pvalue = 2 * (1 - stats.norm.cdf(abs(sharpe_stat)))
        
        return {
            'win_rate_test': {
                'p_value': p_value_winrate,
                'significant': p_value_winrate < self.config.statistical_significance_level,
                'null_hypothesis': 'win_rate = 50%'
            },
            'return_test': {
                'p_value': p_value_returns,
                'significant': p_value_returns < self.config.statistical_significance_level,
                'null_hypothesis': 'mean_return = 0'
            },
            'bootstrap_ci': {
                'lower': ci_lower,
                'upper': ci_upper,
                'mean': np.mean(bootstrap_returns),
                'contains_zero': ci_lower <= 0 <= ci_upper
            },
            'sharpe_test': {
                'p_value': sharpe_pvalue,
                'significant': sharpe_pvalue < self.config.statistical_significance_level if sharpe_pvalue else False,
                'null_hypothesis': 'sharpe_ratio = 0'
            },
            'overall_significance': (
                p_value_winrate < self.config.statistical_significance_level and
                p_value_returns < self.config.statistical_significance_level
            )
        }
    
    def _analyze_calibration_quality(self, results: BacktestResults) -> Dict[str, Any]:
        """Analyze prediction calibration quality"""
        if not results.trade_analysis['trades']:
            return {'error': 'no_trades'}
        
        df_trades = pd.DataFrame(results.trade_analysis['trades'])
        
        predicted_probs = df_trades['calibrated_probability'].values
        actual_outcomes = df_trades['actual_outcome'].values
        
        # Brier Score
        brier_score = brier_score_loss(actual_outcomes, predicted_probs)
        
        # Log Loss
        log_loss_score = log_loss(actual_outcomes, predicted_probs)
        
        # Reliability (calibration) curve
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_curve = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
            
            if np.sum(in_bin) > 0:
                bin_accuracy = np.mean(actual_outcomes[in_bin])
                bin_confidence = np.mean(predicted_probs[in_bin])
                bin_count = np.sum(in_bin)
                
                calibration_curve.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'bin_confidence': bin_confidence,
                    'bin_accuracy': bin_accuracy,
                    'bin_count': bin_count,
                    'calibration_error': abs(bin_confidence - bin_accuracy)
                })
        
        # Expected Calibration Error (ECE)
        if calibration_curve:
            total_samples = len(predicted_probs)
            ece = sum(
                (curve['bin_count'] / total_samples) * curve['calibration_error']
                for curve in calibration_curve
            )
        else:
            ece = 0.0
        
        return {
            'brier_score': brier_score,
            'log_loss': log_loss_score,
            'expected_calibration_error': ece,
            'calibration_curve': calibration_curve,
            'reliability_score': 1.0 - ece,  # Higher is better
            'sharpness': np.std(predicted_probs)  # Spread of predictions
        }
    
    def _analyze_pattern_performance(self, results: BacktestResults) -> Dict[str, Any]:
        """Analyze pattern detection performance"""
        if not self.config.use_pattern_detection or not results.trade_analysis['trades']:
            return {'enabled': False}
        
        # This would analyze how well patterns predicted outcomes
        # For now, return placeholder analysis
        return {
            'enabled': True,
            'pattern_accuracy': 0.55,  # Placeholder
            'high_confidence_patterns': 0.65,  # Placeholder
            'pattern_contribution': 0.15  # Placeholder
        }
    
    def _calculate_risk_metrics(self, results: BacktestResults) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        if not results.trade_analysis['trades']:
            return {}
        
        df_trades = pd.DataFrame(results.trade_analysis['trades'])
        balance_history = results.trade_analysis['balance_history']
        
        # Value at Risk (VaR)
        returns = df_trades['profit_loss'] / df_trades['stake']
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Maximum consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        for trade in results.trade_analysis['trades']:
            if not trade['win']:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        # Calmar ratio (return / max drawdown)
        calmar_ratio = abs(results.total_return / results.max_drawdown) if results.max_drawdown < 0 else 0
        
        # Sortino ratio (return / downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = np.mean(returns) / downside_deviation if downside_deviation > 0 else 0
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'max_consecutive_losses': max_consecutive_losses,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'volatility': np.std(returns),
            'downside_deviation': downside_deviation
        }
    
    def _analyze_trade_characteristics(self, results: BacktestResults) -> Dict[str, Any]:
        """Analyze trade characteristics and patterns"""
        if not results.trade_analysis['trades']:
            return results.trade_analysis
        
        df_trades = pd.DataFrame(results.trade_analysis['trades'])
        
        # Add trade characteristics analysis
        analysis = results.trade_analysis.copy()
        
        # Performance by contract type
        contract_performance = {}
        for contract_type in df_trades['contract_type'].unique():
            contract_trades = df_trades[df_trades['contract_type'] == contract_type]
            contract_performance[contract_type] = {
                'trades': len(contract_trades),
                'win_rate': contract_trades['win'].mean(),
                'avg_profit': contract_trades['profit_loss'].mean(),
                'total_profit': contract_trades['profit_loss'].sum()
            }
        
        analysis['contract_performance'] = contract_performance
        
        # Performance by confidence level
        confidence_bins = pd.cut(df_trades['confidence'], bins=5, labels=['Low', 'Low-Med', 'Medium', 'Med-High', 'High'])
        confidence_performance = {}
        
        for bin_name in confidence_bins.categories:
            bin_trades = df_trades[confidence_bins == bin_name]
            if len(bin_trades) > 0:
                confidence_performance[bin_name] = {
                    'trades': len(bin_trades),
                    'win_rate': bin_trades['win'].mean(),
                    'avg_profit': bin_trades['profit_loss'].mean()
                }
        
        analysis['confidence_performance'] = confidence_performance
        
        return analysis
    
    def _analyze_temporal_patterns(self, results: BacktestResults) -> Dict[str, Any]:
        """Analyze temporal patterns in performance"""
        if not results.trade_analysis['trades']:
            return {}
        
        df_trades = pd.DataFrame(results.trade_analysis['trades'])
        df_trades['datetime'] = pd.to_datetime(df_trades['datetime'])
        
        # Daily performance
        daily_stats = df_trades.groupby(df_trades['datetime'].dt.date).agg({
            'profit_loss': ['sum', 'count'],
            'win': 'mean'
        }).round(4)
        
        # Hourly performance
        hourly_stats = df_trades.groupby(df_trades['datetime'].dt.hour).agg({
            'profit_loss': ['sum', 'count'],
            'win': 'mean'
        }).round(4)
        
        # Day of week performance
        dow_stats = df_trades.groupby(df_trades['datetime'].dt.dayofweek).agg({
            'profit_loss': ['sum', 'count'],
            'win': 'mean'
        }).round(4)
        
        return {
            'daily_performance': daily_stats.to_dict(),
            'hourly_performance': hourly_stats.to_dict(),
            'day_of_week_performance': dow_stats.to_dict(),
            'trading_period': {
                'start': df_trades['datetime'].min().isoformat(),
                'end': df_trades['datetime'].max().isoformat(),
                'duration_days': (df_trades['datetime'].max() - df_trades['datetime'].min()).days
            }
        }
    
    def _combine_window_results(self, window_results: List[BacktestResults], 
                               all_trades: List[Dict], all_balances: List[float]) -> BacktestResults:
        """Combine results from multiple walk-forward windows"""
        if not window_results:
            return self._create_results_object([], [self.config.initial_balance])
        
        # Aggregate metrics
        total_trades = sum(r.total_trades for r in window_results)
        total_winning = sum(r.winning_trades for r in window_results)
        
        # Calculate combined metrics
        combined_results = self._create_results_object(all_trades, all_balances)
        
        return combined_results
    
    def generate_report(self, results: BacktestResults) -> str:
        """Generate comprehensive backtest report"""
        report = []
        
        report.append("=" * 80)
        report.append("ENHANCED BACKTEST REPORT")
        report.append("=" * 80)
        
        # Configuration
        report.append("\n📋 CONFIGURATION:")
        report.append(f"Period: {results.config.start_date or 'All'} to {results.config.end_date or 'All'}")
        report.append(f"Initial Balance: ${results.config.initial_balance:,.2f}")
        report.append(f"Symbols: {results.config.symbols or 'All'}")
        report.append(f"Pattern Detection: {results.config.use_pattern_detection}")
        report.append(f"Calibration: {results.config.use_calibration}")
        report.append(f"Meta-Controller: {results.config.use_meta_controller}")
        report.append(f"Online Learning: {results.config.use_online_learning}")
        
        # Performance Summary
        report.append("\n📊 PERFORMANCE SUMMARY:")
        report.append(f"Total Trades: {results.total_trades:,}")
        report.append(f"Win Rate: {results.win_rate:.2%}")
        report.append(f"Total Return: {results.total_return:.2%}")
        report.append(f"Sharpe Ratio: {results.sharpe_ratio:.3f}")
        report.append(f"Max Drawdown: {results.max_drawdown:.2%}")
        report.append(f"Profit Factor: {results.profit_factor:.2f}")
        
        # Statistical Significance
        if results.statistical_significance:
            sig = results.statistical_significance
            report.append("\n🔬 STATISTICAL SIGNIFICANCE:")
            report.append(f"Win Rate Test: {'✅ Significant' if sig['win_rate_test']['significant'] else '❌ Not Significant'} (p={sig['win_rate_test']['p_value']:.4f})")
            report.append(f"Return Test: {'✅ Significant' if sig['return_test']['significant'] else '❌ Not Significant'} (p={sig['return_test']['p_value']:.4f})")
            report.append(f"Bootstrap CI: [{sig['bootstrap_ci']['lower']:.4f}, {sig['bootstrap_ci']['upper']:.4f}]")
            report.append(f"Overall Significant: {'✅ Yes' if sig['overall_significance'] else '❌ No'}")
        
        # Calibration Quality
        if results.calibration_metrics:
            cal = results.calibration_metrics
            if 'error' not in cal:
                report.append("\n🎯 CALIBRATION QUALITY:")
                report.append(f"Brier Score: {cal['brier_score']:.4f} (lower is better)")
                report.append(f"Log Loss: {cal['log_loss']:.4f} (lower is better)")
                report.append(f"Expected Calibration Error: {cal['expected_calibration_error']:.4f}")
                report.append(f"Reliability Score: {cal['reliability_score']:.4f} (higher is better)")
        
        # Risk Metrics
        if results.risk_metrics:
            risk = results.risk_metrics
            report.append("\n⚠️ RISK ANALYSIS:")
            report.append(f"VaR (95%): {risk['var_95']:.4f}")
            report.append(f"CVaR (95%): {risk['cvar_95']:.4f}")
            report.append(f"Max Consecutive Losses: {risk['max_consecutive_losses']}")
            report.append(f"Sortino Ratio: {risk['sortino_ratio']:.3f}")
            report.append(f"Calmar Ratio: {risk['calmar_ratio']:.3f}")
        
        # Recommendations
        report.append("\n💡 RECOMMENDATIONS:")
        
        if results.statistical_significance.get('overall_significance', False):
            report.append("✅ Strategy shows statistically significant edge")
        else:
            report.append("⚠️ Strategy lacks statistical significance - more data needed")
        
        if results.calibration_metrics.get('reliability_score', 0) > 0.8:
            report.append("✅ Good prediction calibration quality")
        else:
            report.append("⚠️ Prediction calibration needs improvement")
        
        if results.sharpe_ratio > 1.0:
            report.append("✅ Strong risk-adjusted returns")
        elif results.sharpe_ratio > 0.5:
            report.append("⚠️ Moderate risk-adjusted returns")
        else:
            report.append("❌ Poor risk-adjusted returns")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, results: BacktestResults, filename: str = None):
        """Save backtest results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_backtest_{timestamp}.json"
        
        try:
            # Convert to serializable format
            results_dict = {
                'config': results.config.__dict__,
                'metrics': {
                    'total_trades': results.total_trades,
                    'win_rate': results.win_rate,
                    'total_return': results.total_return,
                    'sharpe_ratio': results.sharpe_ratio,
                    'max_drawdown': results.max_drawdown,
                    'profit_factor': results.profit_factor
                },
                'statistical_significance': results.statistical_significance,
                'calibration_metrics': results.calibration_metrics,
                'pattern_analysis': results.pattern_analysis,
                'risk_metrics': results.risk_metrics,
                'temporal_analysis': results.temporal_analysis,
                'trade_count': len(results.trade_analysis['trades'])
            }
            
            with open(filename, 'w') as f:
                json.dump(results_dict, f, indent=2, default=str)
            
            logger.info(f"Backtest results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")

if __name__ == "__main__":
    # Test enhanced backtester
    config = BacktestConfig(
        initial_balance=1000.0,
        symbols=['R_100'],
        use_pattern_detection=True,
        use_calibration=True,
        use_meta_controller=True,
        walk_forward_windows=1
    )
    
    backtester = EnhancedBacktester(config)
    
    print("Enhanced Backtester initialized")
    print("Note: Run with actual historical data for meaningful results")
    
    # Example of how to use:
    # results = backtester.run_comprehensive_backtest()
    # report = backtester.generate_report(results)
    # print(report)
    # backtester.save_results(results)
