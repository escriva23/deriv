# backtester.py - Comprehensive backtesting framework
import sqlite3
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
from config import config
from ai_predictor import AIPredictor
from risk_manager import RiskManager

logger = logging.getLogger(__name__)

class DerivBacktester:
    def __init__(self, start_date: str = None, end_date: str = None):
        self.start_date = start_date
        self.end_date = end_date
        self.ai_predictor = AIPredictor()
        self.results = []
        self.performance_metrics = {}
        
    def load_historical_data(self, symbol: str, limit: int = None) -> pd.DataFrame:
        """Load historical tick data for backtesting"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            
            query = """
                SELECT t.*, tf.* FROM ticks t
                LEFT JOIN tick_features tf ON t.epoch = tf.timestamp AND t.symbol = tf.symbol
                WHERE t.symbol = ?
            """
            params = [symbol]
            
            if self.start_date:
                query += " AND t.timestamp >= ?"
                params.append(int(pd.Timestamp(self.start_date).timestamp()))
            
            if self.end_date:
                query += " AND t.timestamp <= ?"
                params.append(int(pd.Timestamp(self.end_date).timestamp()))
            
            query += " ORDER BY t.epoch"
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if len(df) == 0:
                logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            logger.info(f"Loaded {len(df)} historical records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return pd.DataFrame()
    
    def simulate_contract_outcome(self, contract_type: str, entry_tick: Dict, 
                                exit_tick: Dict, stake: float) -> Tuple[float, bool]:
        """Simulate contract outcome based on entry and exit ticks"""
        try:
            entry_quote = float(entry_tick['quote'])
            exit_quote = float(exit_tick['quote'])
            
            # Simulate different contract types
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
                logger.warning(f"Unknown contract type: {contract_type}")
                return 0.0, False
            
            # Calculate profit/loss
            # Assume payout ratio of 0.8 (typical for Deriv)
            payout_ratio = 0.8
            
            if win:
                profit = stake * payout_ratio
            else:
                profit = -stake
            
            return profit, win
            
        except Exception as e:
            logger.error(f"Error simulating contract outcome: {e}")
            return 0.0, False
    
    def backtest_symbol(self, symbol: str, initial_balance: float = 1000.0) -> Dict:
        """Run backtest for a specific symbol"""
        logger.info(f"Starting backtest for {symbol}")
        
        # Load historical data
        df = self.load_historical_data(symbol)
        if len(df) < 100:
            logger.warning(f"Insufficient data for {symbol}")
            return {}
        
        # Initialize risk manager
        risk_manager = RiskManager(initial_balance)
        
        # Prepare results tracking
        trades = []
        balance_history = [initial_balance]
        
        # Simulate trading
        for i in range(50, len(df) - 1):  # Start after enough data for features
            try:
                current_row = df.iloc[i]
                next_row = df.iloc[i + 1]  # Exit tick
                
                # Skip if missing features
                if pd.isna(current_row.get('last_digit')):
                    continue
                
                # Prepare features for AI prediction
                features = {
                    'last_digit': current_row.get('last_digit', 0),
                    'digit_parity': current_row.get('digit_parity', 'even'),
                    'price_change': current_row.get('price_change', 0),
                    'volatility_5': current_row.get('volatility_5', 0),
                    'volatility_20': current_row.get('volatility_20', 0),
                    'streak_length': current_row.get('streak_length', 0),
                    'streak_direction': current_row.get('streak_direction', 0),
                    'momentum_score': current_row.get('momentum_score', 0)
                }
                
                # Get AI predictions
                predictions = self.ai_predictor.get_ensemble_prediction(symbol, features)
                
                # Check if we should trade
                should_trade, contract_type, confidence = self.ai_predictor.should_trade(predictions)
                
                if not should_trade:
                    continue
                
                # Check risk management
                allowed, reason = risk_manager.check_trading_allowed()
                if not allowed:
                    continue
                
                # Calculate position size
                stake = risk_manager.calculate_position_size(confidence, symbol)
                
                # Simulate contract outcome
                entry_tick = {'quote': current_row['quote']}
                exit_tick = {'quote': next_row['quote']}
                
                profit, win = self.simulate_contract_outcome(
                    contract_type, entry_tick, exit_tick, stake
                )
                
                # Update risk manager
                risk_update = risk_manager.update_balance(profit, contract_type, stake)
                
                # Record trade
                trade_record = {
                    'timestamp': current_row['timestamp'],
                    'datetime': current_row['datetime'],
                    'symbol': symbol,
                    'contract_type': contract_type,
                    'stake': stake,
                    'confidence': confidence,
                    'profit': profit,
                    'win': win,
                    'balance': risk_update['new_balance'],
                    'entry_quote': current_row['quote'],
                    'exit_quote': next_row['quote'],
                    'features': features
                }
                
                trades.append(trade_record)
                balance_history.append(risk_update['new_balance'])
                
                # Stop if balance too low
                if risk_update['new_balance'] < initial_balance * 0.1:
                    logger.warning(f"Stopping backtest - balance too low: ${risk_update['new_balance']:.2f}")
                    break
                    
            except Exception as e:
                logger.error(f"Error in backtest iteration {i}: {e}")
                continue
        
        # Calculate performance metrics
        if len(trades) > 0:
            metrics = self.calculate_performance_metrics(trades, initial_balance)
            
            result = {
                'symbol': symbol,
                'trades': trades,
                'metrics': metrics,
                'balance_history': balance_history
            }
            
            logger.info(f"Backtest completed for {symbol}: {len(trades)} trades, "
                       f"Final balance: ${balance_history[-1]:.2f}")
            
            return result
        else:
            logger.warning(f"No trades executed for {symbol}")
            return {}
    
    def calculate_performance_metrics(self, trades: List[Dict], initial_balance: float) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return {}
        
        df_trades = pd.DataFrame(trades)
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len(df_trades[df_trades['win'] == True])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_profit = df_trades['profit'].sum()
        gross_profit = df_trades[df_trades['profit'] > 0]['profit'].sum()
        gross_loss = abs(df_trades[df_trades['profit'] < 0]['profit'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Return metrics
        final_balance = df_trades['balance'].iloc[-1]
        total_return = (final_balance - initial_balance) / initial_balance
        
        # Risk metrics
        returns = df_trades['profit'] / initial_balance  # Approximate returns
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Drawdown calculation
        balance_series = pd.Series([initial_balance] + df_trades['balance'].tolist())
        peak = balance_series.expanding().max()
        drawdown = (balance_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (simplified)
        if volatility > 0:
            sharpe_ratio = (total_return * 252) / volatility  # Assuming daily trading
        else:
            sharpe_ratio = 0
        
        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        current_win_streak = 0
        current_loss_streak = 0
        
        for trade in trades:
            if trade['win']:
                current_win_streak += 1
                current_loss_streak = 0
                max_consecutive_wins = max(max_consecutive_wins, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                max_consecutive_losses = max(max_consecutive_losses, current_loss_streak)
        
        # Average trade metrics
        avg_win = df_trades[df_trades['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['profit'] < 0]['profit'].mean() if losing_trades > 0 else 0
        avg_trade = df_trades['profit'].mean()
        
        # Confidence metrics
        avg_confidence = df_trades['confidence'].mean()
        high_conf_trades = len(df_trades[df_trades['confidence'] > 0.7])
        high_conf_win_rate = len(df_trades[(df_trades['confidence'] > 0.7) & (df_trades['win'] == True)]) / high_conf_trades if high_conf_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_return': total_return,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'avg_confidence': avg_confidence,
            'high_conf_win_rate': high_conf_win_rate,
            'final_balance': final_balance,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    def run_full_backtest(self, symbols: List[str] = None, initial_balance: float = 1000.0) -> Dict:
        """Run backtest across multiple symbols"""
        if symbols is None:
            symbols = config.SYMBOLS
        
        logger.info(f"Starting full backtest for symbols: {symbols}")
        
        all_results = {}
        combined_trades = []
        
        for symbol in symbols:
            try:
                # Train models if needed (using historical data)
                logger.info(f"Training models for {symbol}")
                self.ai_predictor.train_models(symbol, 'digit_parity')
                self.ai_predictor.train_models(symbol, 'digit_over_under')
                
                # Run backtest for symbol
                result = self.backtest_symbol(symbol, initial_balance)
                
                if result:
                    all_results[symbol] = result
                    combined_trades.extend(result['trades'])
                    
            except Exception as e:
                logger.error(f"Error backtesting {symbol}: {e}")
        
        # Calculate combined metrics
        if combined_trades:
            combined_metrics = self.calculate_performance_metrics(combined_trades, initial_balance)
            
            # Generate summary report
            summary = self.generate_summary_report(all_results, combined_metrics)
            
            return {
                'individual_results': all_results,
                'combined_metrics': combined_metrics,
                'combined_trades': combined_trades,
                'summary': summary
            }
        else:
            logger.warning("No trades found in backtest")
            return {}
    
    def generate_summary_report(self, results: Dict, combined_metrics: Dict) -> str:
        """Generate a summary report of backtest results"""
        report = []
        report.append("=" * 60)
        report.append("BACKTEST SUMMARY REPORT")
        report.append("=" * 60)
        
        if combined_metrics:
            report.append(f"Total Trades: {combined_metrics['total_trades']}")
            report.append(f"Win Rate: {combined_metrics['win_rate']:.2%}")
            report.append(f"Total Return: {combined_metrics['total_return']:.2%}")
            report.append(f"Profit Factor: {combined_metrics['profit_factor']:.2f}")
            report.append(f"Max Drawdown: {combined_metrics['max_drawdown']:.2%}")
            report.append(f"Sharpe Ratio: {combined_metrics['sharpe_ratio']:.2f}")
            report.append(f"Final Balance: ${combined_metrics['final_balance']:.2f}")
            report.append("")
            
            report.append("INDIVIDUAL SYMBOL PERFORMANCE:")
            report.append("-" * 40)
            
            for symbol, result in results.items():
                metrics = result['metrics']
                report.append(f"{symbol}:")
                report.append(f"  Trades: {metrics['total_trades']}")
                report.append(f"  Win Rate: {metrics['win_rate']:.2%}")
                report.append(f"  Return: {metrics['total_return']:.2%}")
                report.append(f"  Max DD: {metrics['max_drawdown']:.2%}")
                report.append("")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict, filename: str = None):
        """Save backtest results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.json"
        
        try:
            # Convert datetime objects to strings for JSON serialization
            json_results = {}
            
            for key, value in results.items():
                if key == 'combined_trades':
                    # Convert datetime objects in trades
                    json_trades = []
                    for trade in value:
                        json_trade = trade.copy()
                        if 'datetime' in json_trade:
                            json_trade['datetime'] = json_trade['datetime'].isoformat()
                        json_trades.append(json_trade)
                    json_results[key] = json_trades
                else:
                    json_results[key] = value
            
            with open(filename, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            logger.info(f"Backtest results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")

def main():
    """Main function for running backtests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deriv AI Trading Bot Backtester')
    parser.add_argument('--symbol', type=str, help='Symbol to backtest (default: all)')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--balance', type=float, default=1000.0, help='Initial balance')
    parser.add_argument('--save', action='store_true', help='Save results to file')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create backtester
    backtester = DerivBacktester(args.start_date, args.end_date)
    
    # Run backtest
    if args.symbol:
        results = backtester.backtest_symbol(args.symbol, args.balance)
        if results:
            print(backtester.generate_summary_report({args.symbol: results}, results['metrics']))
    else:
        results = backtester.run_full_backtest(initial_balance=args.balance)
        if results:
            print(results['summary'])
            
            if args.save:
                backtester.save_results(results)

if __name__ == "__main__":
    main()
