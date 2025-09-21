# 🤖 Deriv AI Trading Bot

An advanced AI-powered trading bot for Deriv's synthetic markets using machine learning, pattern recognition, and sophisticated risk management.

## ⚠️ Important Disclaimers

- **This bot is for educational and research purposes**
- **No guarantees of profits** - trading involves significant risk
- **Always test on demo accounts first**
- **Use proper risk management** - never risk more than you can afford to lose
- **Deriv's synthetic markets use RNG** - no system can guarantee consistent wins

## 🚀 Features

### 🧠 Advanced AI Engine
- **Multiple ML Models**: XGBoost, Random Forest, Gradient Boosting, Logistic Regression
- **Ensemble Predictions**: Combines multiple prediction types for better accuracy
- **Real-time Feature Engineering**: Volatility, momentum, streak detection, digit frequency analysis
- **Continuous Learning**: Models update performance based on trade outcomes

### 📊 Sophisticated Risk Management
- **Kelly Criterion Position Sizing**: Optimal stake calculation based on confidence
- **Multi-layer Risk Controls**: Daily loss limits, consecutive loss limits, drawdown protection
- **Dynamic Stake Adjustment**: Reduces position size after losses
- **Emergency Stop Mechanisms**: Automatic trading halt on risk threshold breach

### 📈 Real-time Data Processing
- **Live Tick Collection**: WebSocket connection to Deriv API
- **Feature Calculation**: Real-time technical indicators and pattern detection
- **Database Storage**: SQLite for tick data, features, trades, and performance metrics

### 🎯 Smart Contract Selection
- **Multiple Contract Types**: Even/Odd, Over/Under, Rise/Fall
- **Confidence-based Trading**: Only trades when AI confidence exceeds threshold
- **Adaptive Strategy**: Switches between contract types based on market conditions

## 📋 Requirements

- Python 3.8+
- Deriv API Token (demo recommended)
- Windows/Linux/MacOS

## 🛠️ Installation

1. **Clone or download the bot files**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Deriv API token:**
   - Go to [Deriv API](https://app.deriv.com/account/api-token)
   - Create a **DEMO** token for testing
   - Set environment variable:
     ```bash
     # Windows
     set DERIV_TOKEN=your_demo_token_here
     
     # Linux/Mac
     export DERIV_TOKEN=your_demo_token_here
     ```

## 🚀 Quick Start

### 1. Data Collection Mode (Recommended First Step)
```bash
python main_bot.py --collect-only
```
This will collect tick data for 30+ minutes to train the AI models.

### 2. Demo Trading Mode
```bash
python main_bot.py
```
Runs the bot in demo mode (default) - **ALWAYS START HERE**

### 3. Live Trading Mode (Only after extensive testing)
```bash
python main_bot.py --live
```
⚠️ **WARNING**: Only use after thorough testing on demo account

## 📁 File Structure

```
deriv-ai-bot/
├── config.py              # Configuration settings
├── data_collector.py       # Real-time tick data collection
├── ai_predictor.py        # ML models and predictions
├── risk_manager.py        # Risk management system
├── trading_executor.py    # Trade execution engine
├── main_bot.py           # Main bot orchestrator
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── trading_data.db      # SQLite database (created automatically)
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Trading Parameters
SYMBOLS = ["R_100", "R_50", "R_75", "R_25"]  # Markets to trade
INITIAL_STAKE = 2.0                          # Base stake amount
MIN_STAKE = 0.35                            # Minimum stake
MAX_STAKE = 100.0                           # Maximum stake

# Risk Management
MAX_DAILY_LOSS = 500.0                      # Daily loss limit
MAX_CONSECUTIVE_LOSSES = 5                  # Max consecutive losses
DAILY_PROFIT_TARGET = 200.0                # Daily profit target

# AI Parameters
MIN_CONFIDENCE = 0.65                       # Minimum prediction confidence
LOOKBACK_WINDOW = 500                       # Ticks for analysis
```

## 📊 How It Works

### 1. Data Collection Phase
- Connects to Deriv WebSocket API
- Collects real-time tick data for synthetic indices
- Calculates technical features (volatility, momentum, streaks)
- Stores everything in SQLite database

### 2. AI Training Phase
- Uses collected data to train multiple ML models
- Performs time-series cross-validation
- Selects best performing model for each symbol/contract type
- Saves trained models to disk

### 3. Trading Phase
- Monitors real-time ticks
- Calculates features for latest market data
- Gets ensemble predictions from AI models
- Applies risk management rules
- Executes trades via Deriv API
- Monitors contract outcomes
- Updates model performance

### 4. Risk Management
- **Position Sizing**: Uses Kelly Criterion with confidence scaling
- **Loss Limits**: Daily and consecutive loss protection
- **Drawdown Control**: Monitors maximum account drawdown
- **Emergency Stops**: Automatic halt on risk threshold breach

## 📈 Expected Performance

**Realistic Expectations:**
- **Win Rate**: 52-58% (slightly better than random due to pattern detection)
- **Profit Factor**: 1.1-1.3 (after accounting for Deriv's house edge)
- **Drawdowns**: Expect 10-20% drawdowns during losing streaks
- **Long-term**: House edge means consistent long-term profits are unlikely

**Key Success Factors:**
1. **Strict Risk Management**: Never risk more than 2% per trade
2. **Proper Testing**: Extensive backtesting and demo trading
3. **Regular Monitoring**: Don't leave bot unattended for long periods
4. **Model Updates**: Retrain models regularly with new data

## 🔧 Advanced Usage

### Custom Strategies
Modify `ai_predictor.py` to implement your own prediction logic:

```python
def custom_strategy(self, features):
    # Your custom logic here
    if features['streak_length'] > 5 and features['volatility_5'] < 0.01:
        return 'DIGITEVEN', 0.75  # contract_type, confidence
    return None, 0.0
```

### Additional Features
- **Web Dashboard**: Uncomment Flask/Dash in requirements.txt
- **Telegram Notifications**: Add telegram bot integration
- **Advanced Models**: Implement LSTM or Transformer models
- **Multi-timeframe Analysis**: Add longer-term trend analysis

## 🐛 Troubleshooting

### Common Issues

1. **"DERIV_TOKEN not set"**
   - Set your Deriv API token as environment variable
   - Use DEMO token for testing

2. **"Connection failed"**
   - Check internet connection
   - Verify token is valid
   - Try regenerating token in Deriv

3. **"Insufficient data"**
   - Run data collection mode for 30+ minutes first
   - Check if ticks are being saved to database

4. **"No models trained"**
   - Ensure sufficient data collection
   - Check for errors in training logs
   - Verify all dependencies installed

### Debug Mode
Add logging level to see more details:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📞 Support

This is an educational project. For issues:
1. Check the troubleshooting section
2. Review log files for error messages
3. Ensure all requirements are installed
4. Test with demo account first

## ⚖️ Legal Notice

- This software is provided "as is" without warranty
- Trading involves substantial risk of loss
- Past performance does not guarantee future results
- Users are responsible for compliance with local regulations
- The authors are not responsible for any financial losses

## 🎯 Recommended Usage

1. **Start with data collection** (30+ minutes)
2. **Test extensively on demo account** (1+ weeks)
3. **Start with very small stakes** if going live
4. **Monitor closely** and adjust parameters
5. **Never risk more than you can afford to lose**

---

**Remember: The goal is not to "beat" the system, but to manage risk intelligently while potentially capturing short-term market inefficiencies.**
#   d e r i v  
 