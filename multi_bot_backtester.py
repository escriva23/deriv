# multi_bot_backtester.py - Backtesting framework for multi-bot system
import sqlite3
import pandas as pd
import numpy as np
import logging
import json
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from shared_config import config, BOT_IDS

logger = logging.getLogger(__name__)

class MultiBotBacktester:
    def __init__(self, start_date: str = None, end_date: str = None):
        self.start_date = start_date
        self.end_date = end_date
        
        # Simulated probe bots
        self.probe_a = ProbeASimulator()
        self.probe_b = ProbeBSimulator()
        self.probe_c = ProbeCSimulator()
        self.coordinator = CoordinatorSimulator()
        
        # Results tracking
        self.results = {
            'probe_a': [],
            'probe_b': [],
            'probe_c': [],
            'coordinator': []
        }
        
        # Signal exchange simulation
        self.signal_queue = deque()
        self.performance_tracking = defaultdict(dict)
        
    def load_historical_data(self, symbol: str, limit: int = None) -> pd.DataFrame:
        """Load historical tick data for backtesting"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            
            query = """
                SELECT epoch, symbol, quote, timestamp 
                FROM ticks 
                WHERE symbol = ?
            """
            params = [symbol]
            
            if self.start_date:
                query += " AND timestamp >= ?"
                params.append(int(pd.Timestamp(self.start_date).timestamp()))
            
            if self.end_date:
                query += " AND timestamp <= ?"
                params.append(int(pd.Timestamp(self.end_date).timestamp()))
            
            query += " ORDER BY epoch"
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if len(df) == 0:
                logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
            
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            logger.info(f"Loaded {len(df)} historical records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return pd.DataFrame()
    
    def simulate_contract_outcome(self, contract_type: str, entry_quote: float, 
                                exit_quote: float, stake: float) -> Tuple[float, bool]:
        """Simulate contract outcome"""
        try:
            if contract_type == "DIGITEVEN":
                last_digit = int(str(exit_quote).split('.')[-1][-1])
                win = (last_digit % 2 == 0)
                
            elif contract_type == "DIGITODD":
                last_digit = int(str(exit_quote).split('.')[-1][-1])
                win = (last_digit % 2 == 1)
                
            elif contract_type == "DIGITOVER":
                last_digit = int(str(exit_quote).split('.')[-1][-1])
                win = (last_digit > 5)
                
            elif contract_type == "DIGITUNDER":
                last_digit = int(str(exit_quote).split('.')[-1][-1])
                win = (last_digit < 5)
                
            elif contract_type == "CALL":
                win = (exit_quote > entry_quote)
                
            elif contract_type == "PUT":
                win = (exit_quote < entry_quote)
                
            else:
                return 0.0, False
            
            # Simulate payout (typical Deriv payout ratio)
            payout_ratio = 0.8
            profit = stake * payout_ratio if win else -stake
            
            return profit, win
            
        except Exception as e:
            logger.error(f"Error simulating contract outcome: {e}")
            return 0.0, False
    
    def run_backtest(self, symbol: str = "R_100", initial_balance: float = 1000.0) -> Dict:
        """Run complete multi-bot backtest"""
        logger.info(f"Starting multi-bot backtest for {symbol}")
        
        # Load historical data
        df = self.load_historical_data(symbol, limit=10000)  # Limit for testing
        if len(df) < 100:
            logger.error("Insufficient historical data")
            return {}
        
        # Initialize balances
        balances = {
            'probe_a': initial_balance,
            'probe_b': initial_balance,
            'probe_c': initial_balance,
            'coordinator': initial_balance * 10  # Real account has more capital
        }
        
        # Process each tick
        for i in range(50, len(df) - 1):  # Start after enough data for analysis
            try:
                current_tick = df.iloc[i]
                next_tick = df.iloc[i + 1]
                
                current_time = current_tick['timestamp']
                current_quote = float(current_tick['quote'])
                next_quote = float(next_tick['quote'])
                
                # Get recent data for analysis
                recent_data = df.iloc[max(0, i-100):i+1]
                
                # Generate probe signals
                signals = []
                
                # Probe A signal (digit parity)
                probe_a_signal = self.probe_a.generate_signal(recent_data, current_time)
                if probe_a_signal:
                    signals.append(probe_a_signal)
                
                # Probe B signal (digit over/under - opposite tendency)
                probe_b_signal = self.probe_b.generate_signal(recent_data, current_time)
                if probe_b_signal:
                    signals.append(probe_b_signal)
                
                # Probe C signal (momentum)
                probe_c_signal = self.probe_c.generate_signal(recent_data, current_time)
                if probe_c_signal:
                    signals.append(probe_c_signal)
                
                # Execute probe trades
                for signal in signals:
                    bot_id = signal['bot_id']
                    contract_type = signal['contract_type']
                    stake = signal['stake']
                    
                    profit, win = self.simulate_contract_outcome(
                        contract_type, current_quote, next_quote, stake
                    )
                    
                    # Update balance
                    balances[bot_id] += profit
                    
                    # Record result
                    result = {
                        'timestamp': current_time,
                        'bot_id': bot_id,
                        'contract_type': contract_type,
                        'stake': stake,
                        'profit': profit,
                        'win': win,
                        'balance': balances[bot_id],
                        'entry_quote': current_quote,
                        'exit_quote': next_quote,
                        'confidence': signal['confidence']
                    }
                    
                    self.results[bot_id].append(result)
                    
                    # Update performance tracking
                    self.update_performance_tracking(bot_id, result)
                
                # Coordinator decision
                if len(signals) >= 2:  # Need multiple signals
                    coordinator_decision = self.coordinator.make_decision(
                        signals, self.performance_tracking, current_time
                    )
                    
                    if coordinator_decision:
                        contract_type = coordinator_decision['contract_type']
                        stake = coordinator_decision['stake']
                        
                        profit, win = self.simulate_contract_outcome(
                            contract_type, current_quote, next_quote, stake
                        )
                        
                        balances['coordinator'] += profit
                        
                        result = {
                            'timestamp': current_time,
                            'bot_id': 'coordinator',
                            'contract_type': contract_type,
                            'stake': stake,
                            'profit': profit,
                            'win': win,
                            'balance': balances['coordinator'],
                            'entry_quote': current_quote,
                            'exit_quote': next_quote,
                            'confidence': coordinator_decision['confidence'],
                            'reasoning': coordinator_decision.get('reasoning', '')
                        }
                        
                        self.results['coordinator'].append(result)
                        self.update_performance_tracking('coordinator', result)
                
            except Exception as e:
                logger.error(f"Error processing tick {i}: {e}")
                continue
        
        # Calculate final metrics
        metrics = self.calculate_metrics(balances, initial_balance)
        
        return {
            'results': self.results,
            'metrics': metrics,
            'balances': balances
        }
    
    def update_performance_tracking(self, bot_id: str, result: Dict):
        """Update performance tracking for bots"""
        if bot_id not in self.performance_tracking:
            self.performance_tracking[bot_id] = {
                'total_trades': 0,
                'winning_trades': 0,
                'total_profit': 0.0,
                'recent_results': deque(maxlen=config.PERFORMANCE_WINDOW)
            }
        
        perf = self.performance_tracking[bot_id]
        perf['total_trades'] += 1
        perf['total_profit'] += result['profit']
        perf['recent_results'].append(result['win'])
        
        if result['win']:
            perf['winning_trades'] += 1
        
        # Calculate recent performance
        if len(perf['recent_results']) > 0:
            perf['recent_performance'] = sum(perf['recent_results']) / len(perf['recent_results'])
        else:
            perf['recent_performance'] = 0.0
    
    def calculate_metrics(self, balances: Dict, initial_balance: float) -> Dict:
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        for bot_id, results in self.results.items():
            if not results:
                continue
            
            df_results = pd.DataFrame(results)
            
            total_trades = len(results)
            winning_trades = len(df_results[df_results['win'] == True])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_profit = df_results['profit'].sum()
            final_balance = balances[bot_id]
            
            if bot_id == 'coordinator':
                initial = initial_balance * 10
            else:
                initial = initial_balance
            
            total_return = (final_balance - initial) / initial
            
            # Risk metrics
            returns = df_results['profit'] / initial
            volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
            
            # Drawdown
            balance_series = pd.Series([initial] + df_results['balance'].tolist())
            peak = balance_series.expanding().max()
            drawdown = (balance_series - peak) / peak
            max_drawdown = drawdown.min()
            
            # Sharpe ratio
            if volatility > 0:
                sharpe_ratio = (total_return * 252) / volatility
            else:
                sharpe_ratio = 0
            
            metrics[bot_id] = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'total_return': total_return,
                'final_balance': final_balance,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'volatility': volatility
            }
        
        return metrics
    
    def generate_report(self, results: Dict) -> str:
        """Generate comprehensive backtest report"""
        report = []
        report.append("=" * 80)
        report.append("MULTI-BOT TRADING SYSTEM BACKTEST REPORT")
        report.append("=" * 80)
        
        metrics = results['metrics']
        balances = results['balances']
        
        # Overall summary
        report.append("\n📊 OVERALL PERFORMANCE:")
        report.append("-" * 40)
        
        for bot_id, bot_metrics in metrics.items():
            bot_name = bot_id.replace('_', ' ').title()
            report.append(f"\n{bot_name}:")
            report.append(f"  Trades: {bot_metrics['total_trades']}")
            report.append(f"  Win Rate: {bot_metrics['win_rate']:.2%}")
            report.append(f"  Total Return: {bot_metrics['total_return']:.2%}")
            report.append(f"  Final Balance: ${bot_metrics['final_balance']:.2f}")
            report.append(f"  Max Drawdown: {bot_metrics['max_drawdown']:.2%}")
            report.append(f"  Sharpe Ratio: {bot_metrics['sharpe_ratio']:.2f}")
        
        # Strategy analysis
        report.append("\n🎯 STRATEGY ANALYSIS:")
        report.append("-" * 40)
        
        # Compare probe performances
        probe_performances = {
            'probe_a': metrics.get('probe_a', {}).get('win_rate', 0),
            'probe_b': metrics.get('probe_b', {}).get('win_rate', 0),
            'probe_c': metrics.get('probe_c', {}).get('win_rate', 0)
        }
        
        best_probe = max(probe_performances.items(), key=lambda x: x[1])
        report.append(f"Best Performing Probe: {best_probe[0]} ({best_probe[1]:.2%} win rate)")
        
        # Coordinator effectiveness
        if 'coordinator' in metrics:
            coord_metrics = metrics['coordinator']
            report.append(f"Coordinator Win Rate: {coord_metrics['win_rate']:.2%}")
            report.append(f"Coordinator Return: {coord_metrics['total_return']:.2%}")
            
            # Compare to best probe
            if coord_metrics['win_rate'] > best_probe[1]:
                report.append("✅ Coordinator outperformed best individual probe")
            else:
                report.append("❌ Coordinator underperformed best individual probe")
        
        # Risk assessment
        report.append("\n⚠️ RISK ASSESSMENT:")
        report.append("-" * 40)
        
        for bot_id, bot_metrics in metrics.items():
            if bot_metrics['max_drawdown'] < -0.2:
                report.append(f"🔴 {bot_id}: High drawdown risk ({bot_metrics['max_drawdown']:.2%})")
            elif bot_metrics['max_drawdown'] < -0.1:
                report.append(f"🟡 {bot_id}: Moderate drawdown risk ({bot_metrics['max_drawdown']:.2%})")
            else:
                report.append(f"🟢 {bot_id}: Low drawdown risk ({bot_metrics['max_drawdown']:.2%})")
        
        # Recommendations
        report.append("\n💡 RECOMMENDATIONS:")
        report.append("-" * 40)
        
        if 'coordinator' in metrics:
            coord_return = metrics['coordinator']['total_return']
            if coord_return > 0.1:
                report.append("✅ System shows strong potential for live trading")
            elif coord_return > 0.05:
                report.append("🟡 System shows moderate potential - consider optimization")
            else:
                report.append("🔴 System needs significant improvement before live trading")
        
        report.append(f"Minimum recommended trades before live: {config.BACKTEST_MIN_TRADES}")
        report.append(f"Minimum win rate threshold: {config.BACKTEST_MIN_WIN_RATE:.2%}")
        
        return "\n".join(report)

# Simplified probe simulators for backtesting
class ProbeASimulator:
    def __init__(self):
        self.digit_history = []
        self.parity_history = []
    
    def generate_signal(self, data: pd.DataFrame, timestamp: float) -> Optional[Dict]:
        """Generate Probe A signal (digit parity)"""
        if len(data) < 20:
            return None
        
        # Extract last digits
        quotes = data['quote'].values
        digits = [int(str(float(q)).split('.')[-1][-1]) for q in quotes]
        
        self.digit_history.extend(digits[-5:])  # Add recent digits
        if len(self.digit_history) > 50:
            self.digit_history = self.digit_history[-50:]
        
        if len(self.digit_history) < 20:
            return None
        
        # Calculate parity bias
        even_count = sum(1 for d in self.digit_history if d % 2 == 0)
        odd_count = len(self.digit_history) - even_count
        
        even_freq = even_count / len(self.digit_history)
        odd_freq = odd_count / len(self.digit_history)
        
        # Predict underrepresented parity
        if even_freq < 0.45:
            return {
                'bot_id': 'probe_a',
                'contract_type': 'DIGITEVEN',
                'stake': config.PROBE_STAKE,
                'confidence': 0.5 + (0.5 - even_freq),
                'timestamp': timestamp
            }
        elif odd_freq < 0.45:
            return {
                'bot_id': 'probe_a',
                'contract_type': 'DIGITODD',
                'stake': config.PROBE_STAKE,
                'confidence': 0.5 + (0.5 - odd_freq),
                'timestamp': timestamp
            }
        
        return None

class ProbeBSimulator:
    def __init__(self):
        self.digit_history = []
    
    def generate_signal(self, data: pd.DataFrame, timestamp: float) -> Optional[Dict]:
        """Generate Probe B signal (digit over/under - opposite strategy)"""
        if len(data) < 20:
            return None
        
        quotes = data['quote'].values
        digits = [int(str(float(q)).split('.')[-1][-1]) for q in quotes]
        
        self.digit_history.extend(digits[-5:])
        if len(self.digit_history) > 100:
            self.digit_history = self.digit_history[-100:]
        
        if len(self.digit_history) < 20:
            return None
        
        # Calculate over/under bias
        over_count = sum(1 for d in self.digit_history if d > 5)
        under_count = sum(1 for d in self.digit_history if d < 5)
        total = len(self.digit_history)
        
        over_freq = over_count / total
        under_freq = under_count / total
        
        # Predict underrepresented category
        if over_freq < 0.35:  # Expected ~40%
            return {
                'bot_id': 'probe_b',
                'contract_type': 'DIGITOVER',
                'stake': config.PROBE_STAKE,
                'confidence': 0.5 + (0.4 - over_freq),
                'timestamp': timestamp
            }
        elif under_freq < 0.45:  # Expected ~50%
            return {
                'bot_id': 'probe_b',
                'contract_type': 'DIGITUNDER',
                'stake': config.PROBE_STAKE,
                'confidence': 0.5 + (0.5 - under_freq),
                'timestamp': timestamp
            }
        
        return None

class ProbeCSimulator:
    def __init__(self):
        self.price_history = []
    
    def generate_signal(self, data: pd.DataFrame, timestamp: float) -> Optional[Dict]:
        """Generate Probe C signal (momentum)"""
        if len(data) < 10:
            return None
        
        quotes = data['quote'].values.astype(float)
        self.price_history.extend(quotes[-5:])
        
        if len(self.price_history) > 20:
            self.price_history = self.price_history[-20:]
        
        if len(self.price_history) < 10:
            return None
        
        # Calculate momentum
        recent_changes = []
        for i in range(1, len(self.price_history)):
            change = (self.price_history[i] - self.price_history[i-1]) / self.price_history[i-1]
            recent_changes.append(change)
        
        if len(recent_changes) < 5:
            return None
        
        # Weight recent changes more
        weights = np.exp(np.linspace(0, 1, len(recent_changes)))
        momentum = np.average(recent_changes, weights=weights)
        
        if momentum > 0.0001:  # Upward momentum
            return {
                'bot_id': 'probe_c',
                'contract_type': 'CALL',
                'stake': config.PROBE_STAKE,
                'confidence': min(0.9, 0.5 + abs(momentum) * 1000),
                'timestamp': timestamp
            }
        elif momentum < -0.0001:  # Downward momentum
            return {
                'bot_id': 'probe_c',
                'contract_type': 'PUT',
                'stake': config.PROBE_STAKE,
                'confidence': min(0.9, 0.5 + abs(momentum) * 1000),
                'timestamp': timestamp
            }
        
        return None

class CoordinatorSimulator:
    def __init__(self):
        self.consecutive_losses = 0
        self.last_trade_time = 0
    
    def make_decision(self, signals: List[Dict], performance_tracking: Dict, 
                     timestamp: float) -> Optional[Dict]:
        """Make coordinator trading decision"""
        if timestamp - self.last_trade_time < 2.0:  # Minimum time between trades
            return None
        
        if len(signals) < 2:
            return None
        
        # Find best performing probe
        best_probe = None
        best_performance = 0
        
        for bot_id, perf in performance_tracking.items():
            if bot_id == 'coordinator':
                continue
            
            recent_perf = perf.get('recent_performance', 0)
            total_trades = perf.get('total_trades', 0)
            
            if total_trades >= 5 and recent_perf > best_performance:
                best_performance = recent_perf
                best_probe = bot_id
        
        # Look for signals from best probe
        best_signal = None
        for signal in signals:
            if signal['bot_id'] == best_probe and signal['confidence'] > config.MIN_PROBABILITY:
                best_signal = signal
                break
        
        # If no best probe signal, use consensus
        if not best_signal:
            # Group by contract type
            contract_groups = defaultdict(list)
            for signal in signals:
                if signal['confidence'] > config.MIN_PROBABILITY:
                    contract_groups[signal['contract_type']].append(signal)
            
            # Find strongest consensus
            best_contract = None
            best_avg_confidence = 0
            
            for contract_type, type_signals in contract_groups.items():
                avg_confidence = np.mean([s['confidence'] for s in type_signals])
                if avg_confidence > best_avg_confidence:
                    best_avg_confidence = avg_confidence
                    best_contract = contract_type
            
            if best_contract and best_avg_confidence > config.MIN_PROBABILITY:
                best_signal = {
                    'contract_type': best_contract,
                    'confidence': best_avg_confidence
                }
        
        if not best_signal:
            return None
        
        # Calculate position size (larger for coordinator)
        base_stake = config.PROBE_STAKE * 5  # 5x probe stake
        confidence_multiplier = max(0.5, min(2.0, best_signal['confidence'] * 1.5))
        stake = base_stake * confidence_multiplier
        
        # Apply consecutive loss reduction
        if self.consecutive_losses > 0:
            stake *= (0.8 ** self.consecutive_losses)
        
        stake = max(config.MIN_STAKE, min(config.MAX_STAKE, stake))
        
        self.last_trade_time = timestamp
        
        return {
            'contract_type': best_signal['contract_type'],
            'stake': stake,
            'confidence': best_signal['confidence'],
            'reasoning': f"Following {'best probe' if best_probe else 'consensus'}"
        }

def main():
    """Main backtesting function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Bot System Backtester')
    parser.add_argument('--symbol', type=str, default='R_100', help='Symbol to backtest')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--balance', type=float, default=1000.0, help='Initial balance')
    parser.add_argument('--save', action='store_true', help='Save results to file')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create backtester
    backtester = MultiBotBacktester(args.start_date, args.end_date)
    
    # Run backtest
    results = backtester.run_backtest(args.symbol, args.balance)
    
    if results:
        # Generate and print report
        report = backtester.generate_report(results)
        print(report)
        
        if args.save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multi_bot_backtest_{timestamp}.json"
            
            # Save results
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save report
            report_filename = f"multi_bot_report_{timestamp}.txt"
            with open(report_filename, 'w') as f:
                f.write(report)
            
            print(f"\nResults saved to {filename}")
            print(f"Report saved to {report_filename}")

if __name__ == "__main__":
    main()
