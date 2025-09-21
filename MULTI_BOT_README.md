# 🤖 Multi-Bot Deriv Trading System

An advanced multi-bot intelligence system for Deriv binary options trading that uses opposing probe bots to gather market intelligence and a coordinator bot to make informed real-money trades.

## 🎯 System Overview

This system implements a sophisticated **probe-and-coordinate** strategy:

### 🔍 **3 Probe Bots (Demo Accounts)**
- **Probe A**: Digit parity analysis (EVEN/ODD) using rolling frequency windows
- **Probe B**: Over/Under analysis (DIGITOVER/DIGITUNDER) - takes OPPOSITE strategy to Probe A
- **Probe C**: Momentum-based analysis (CALL/PUT) using tick streaks and micro-volatility

### 🎛️ **Coordinator Bot (Real Account)**
- Analyzes probe signals in real-time via Redis
- Filters stale signals (<0.6s freshness requirement)
- Calculates Expected Value (EV) and probability thresholds
- Makes informed trading decisions based on probe performance
- **Mirrors winning probe trades simultaneously**

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Probe A   │    │   Probe B   │    │   Probe C   │
│ (Parity)    │    │(Over/Under) │    │ (Momentum)  │
│ Demo Acct   │    │ Demo Acct   │    │ Demo Acct   │
│ EVEN/ODD    │    │ OVER/UNDER  │    │ CALL/PUT    │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │
                    ┌─────▼─────┐
                    │   Redis   │
                    │  Message  │
                    │   Queue   │
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │Coordinator│
                    │Real Money │
                    │  Trading  │
                    │ + Mirror  │
                    └───────────┘
```

## 🚀 Quick Start

### Prerequisites

1. **Redis Server** (for real-time signal exchange)
   ```bash
   # Windows (using Chocolatey)
   choco install redis-64
   redis-server
   
   # Or download from: https://redis.io/download
   ```

2. **Python Dependencies**
   ```bash
   pip install -r multi_bot_requirements.txt
   ```

3. **Deriv API Tokens** (Get from [Deriv API](https://app.deriv.com/account/api-token))
   - 3 Demo account tokens (for probe bots)
   - 1 Real account token (for coordinator)

### Environment Setup

Create a `.env` file or set environment variables:

```bash
# Demo account tokens (for probe bots)
export PROBE_A_TOKEN="your_demo_token_1"
export PROBE_B_TOKEN="your_demo_token_2" 
export PROBE_C_TOKEN="your_demo_token_3"

# Real account token (for coordinator)
export COORDINATOR_TOKEN="your_real_token"
```

### Launch System

```bash
# Check prerequisites
python multi_bot_launcher.py --check-only

# Start all bots (full system)
python multi_bot_launcher.py --mode full

# Start only probe bots (testing mode)
python multi_bot_launcher.py --mode probes_only

# Interactive mode with commands
python multi_bot_launcher.py --interactive
```

## 📊 Strategy Details

### Probe Bot Strategies

#### 🅰️ **Probe A - Digit Parity (EVEN/ODD)**
- Analyzes last digit frequency (even vs odd)
- Uses 50-tick rolling window
- Predicts underrepresented parity
- Confidence based on frequency deviation
- **Contract Types**: DIGITEVEN, DIGITODD

#### 🅱️ **Probe B - Over/Under (OPPOSITE)**
- Analyzes digit distribution (>5 vs <5)
- Uses 100-tick histogram analysis
- Takes **OPPOSITE** approach to Probe A
- Includes skewness and entropy factors
- **Contract Types**: DIGITOVER, DIGITUNDER

#### ©️ **Probe C - Momentum (CALL/PUT)**
- Analyzes price momentum and acceleration
- Uses 20-tick momentum window
- Detects trend strength via linear regression
- Factors in micro-volatility and tick streaks
- **Contract Types**: CALL, PUT

### 🎯 Coordinator Intelligence

The coordinator makes trading decisions based on:

1. **Signal Freshness**: Only considers signals <0.6 seconds old
2. **Probe Performance**: Tracks recent win rates and switches to best performer
3. **Expected Value**: Calculates EV = (probability × payout) - stake
4. **Consensus Analysis**: Aggregates multiple probe signals
5. **Risk Management**: Position sizing, daily limits, consecutive loss protection
6. **Mirror Trading**: When trading, mirrors the winning probe simultaneously

## ⚙️ Configuration

Edit `shared_config.py` to customize:

```python
# Signal processing
MAX_SIGNAL_AGE = 0.6  # seconds
MIN_PROBABILITY = 0.65  # minimum confidence to trade
MIN_EV_THRESHOLD = 0.05  # minimum expected value

# Risk management
MAX_STAKE_PCT = 0.02  # 2% of balance per trade
DAILY_LOSS_LIMIT_PCT = 0.05  # 5% daily loss limit
MAX_CONSECUTIVE_LOSSES = 7  # cooldown trigger
PROBE_STAKE = 1.0  # probe bot stake amount
```

## 📈 Backtesting

Test the system with historical data:

```bash
# Run backtest
python multi_bot_backtester.py --symbol R_100 --balance 1000 --save

# Specific date range
python multi_bot_backtester.py --start-date 2024-01-01 --end-date 2024-01-31
```

Sample backtest output:
```
================================================================================
MULTI-BOT TRADING SYSTEM BACKTEST REPORT
================================================================================

📊 OVERALL PERFORMANCE:
----------------------------------------

Probe A:
  Trades: 245
  Win Rate: 52.24%
  Total Return: 8.45%
  Final Balance: $1084.50
  Max Drawdown: -12.30%
  Sharpe Ratio: 1.23

Coordinator:
  Trades: 89
  Win Rate: 58.43%
  Total Return: 15.67%
  Final Balance: $11567.00
  Max Drawdown: -8.90%
  Sharpe Ratio: 2.15

🎯 STRATEGY ANALYSIS:
----------------------------------------
Best Performing Probe: probe_a (52.24% win rate)
Coordinator Win Rate: 58.43%
Coordinator Return: 15.67%
✅ Coordinator outperformed best individual probe

💡 RECOMMENDATIONS:
----------------------------------------
✅ System shows strong potential for live trading
```

## 🛡️ Risk Management

### Built-in Safety Features

1. **Position Sizing**: Kelly Criterion-based with confidence scaling
2. **Daily Limits**: Automatic shutdown at 5% daily loss
3. **Consecutive Loss Protection**: Cooldown after 7 consecutive losses
4. **Signal Validation**: Rejects stale or low-confidence signals
5. **Balance Monitoring**: Real-time balance tracking and alerts

### Emergency Controls

```bash
# Stop all bots immediately
python multi_bot_launcher.py --interactive
> stop coordinator
> stop probe_a
> stop probe_b  
> stop probe_c
```

## 📋 Monitoring & Management

### Interactive Commands

```bash
python multi_bot_launcher.py --interactive

Available commands:
> status          # Show system status
> start <bot>     # Start specific bot
> stop <bot>      # Stop specific bot
> restart <bot>   # Restart specific bot
> quit            # Exit launcher
```

### System Status

```
============================================================
MULTI-BOT SYSTEM STATUS
============================================================
Running Bots: 4/4

🟢 Probe A (Parity)
   PID: 12345
   Runtime: 02:15:30

🟢 Probe B (Over/Under)
   PID: 12346
   Runtime: 02:15:28

🟢 Probe C (Momentum)
   PID: 12347
   Runtime: 02:15:26

🟢 Coordinator (Real)
   PID: 12348
   Runtime: 02:05:20
```

## 📁 Complete File Structure

```
j:\esc\
├── shared_config.py           # ✅ System configuration
├── signal_manager.py          # ✅ Redis signal exchange
├── probe_a.py                 # ✅ Probe A (parity bot)
├── probe_b.py                 # ✅ Probe B (over/under bot)
├── probe_c.py                 # ✅ Probe C (momentum bot)
├── coordinator.py             # ✅ Main coordinator bot
├── multi_bot_launcher.py      # ✅ System launcher
├── multi_bot_backtester.py    # ✅ Backtesting framework
├── multi_bot_requirements.txt # ✅ Dependencies
├── MULTI_BOT_README.md        # ✅ This documentation
├── config.py                  # 📄 Original single bot config
├── data_collector.py          # 📄 Original data collector
├── ai_predictor.py            # 📄 Original AI predictor
├── risk_manager.py            # 📄 Original risk manager
├── trading_executor.py        # 📄 Original trading executor
├── main_bot.py                # 📄 Original main bot
├── backtester.py              # 📄 Original backtester
└── README.md                  # 📄 Original documentation
```

## 🔧 Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   ```bash
   # Start Redis server
   redis-server
   # Or check if running: redis-cli ping
   ```

2. **Missing API Tokens**
   ```bash
   # Check environment variables
   echo $PROBE_A_TOKEN
   echo $COORDINATOR_TOKEN
   ```

3. **Bot Crashes**
   ```bash
   # Check logs for specific bot
   python multi_bot_launcher.py --interactive
   > restart probe_a
   ```

4. **No Trading Activity**
   - Verify minimum confidence thresholds in `shared_config.py`
   - Check signal freshness requirements (MAX_SIGNAL_AGE)
   - Ensure sufficient historical data for analysis

### Debug Mode

Enable detailed logging:
```python
# In shared_config.py
LOG_LEVEL = "DEBUG"
```

## 🎯 Key Strategy Features

### **Opposing Bot Intelligence**
- **Probe A & B**: Take opposite positions on digit-based contracts
- **Market Intelligence**: Learn which strategy works in current conditions
- **Adaptive Strategy**: Coordinator follows the winning approach

### **Real-time Performance Analysis**
- **Dynamic Switching**: Automatically follows best-performing probe
- **Confidence Weighting**: Higher stakes when probe confidence is high
- **Performance Tracking**: 100-trade rolling window for recent performance

### **Mirror Trading**
- **Simultaneous Execution**: When coordinator trades, it mirrors the winning probe
- **Risk Alignment**: Real account follows demo account success
- **Continuous Learning**: System improves by tracking probe outcomes

## ⚠️ Important Disclaimers

### Trading Risks
- **No Guaranteed Profits**: This system cannot guarantee profits
- **Market Risk**: Binary options trading involves substantial risk
- **Capital Loss**: You may lose your entire investment
- **Demo Testing**: Always test thoroughly on demo accounts first

### Technical Limitations
- **RNG Markets**: Deriv uses random number generation for synthetic markets
- **Pattern Detection**: Historical patterns may not predict future results
- **System Failures**: Technical issues can cause unexpected losses
- **Latency**: Network delays can affect signal timing

### Recommended Usage Protocol
1. **Start Small**: Begin with minimal stakes ($0.35-$1.00)
2. **Demo First**: Run extensive demo testing (50,000+ trades)
3. **Monitor Closely**: Never leave system unattended
4. **Set Limits**: Use strict daily and total loss limits
5. **Regular Review**: Analyze performance and adjust parameters
6. **Gradual Scaling**: Only increase stakes after proven success

## 📞 Support & Next Steps

### Pre-Launch Checklist
- [ ] Redis server installed and running
- [ ] All environment variables set
- [ ] Demo account tokens tested
- [ ] Real account token verified
- [ ] Backtesting completed successfully
- [ ] Risk limits configured appropriately

### Launch Sequence
1. **Prerequisites Check**: `python multi_bot_launcher.py --check-only`
2. **Demo Testing**: `python multi_bot_launcher.py --mode probes_only`
3. **Full System**: `python multi_bot_launcher.py --mode full --interactive`

## 📜 License

This software is provided for educational purposes only. Use at your own risk.

---

**🚨 CRITICAL REMINDER: Trading involves substantial risk. Never trade with money you cannot afford to lose. Always start with demo accounts and minimal stakes.**
