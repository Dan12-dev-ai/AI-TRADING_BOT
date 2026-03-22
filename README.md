# 🤖 AI-TRADING_BOT (Medallion-X)

An automated trading system designed for institutional strategies, integrating AI-driven analysis and algorithmic execution.

## 🚀 Features
- **Institutional Logic:** Focused on Smart Money Concepts (SMC) and Inner Circle Trader (ICT) methods.
- **Market Structure Analysis:** Automated detection of BOS (Break of Structure) and CHoCH (Change of Character).
- **Risk Management:** Built-in position sizing and stop-loss logic.
- **Deployment Ready:** Configured for Replit and cloud-based environments.

## 🛠 Tech Stack
- **Language:** Python 3.x
- **Environment:** Docker / Replit
- **Libraries:** Pandas, CCXT (for exchange connectivity), NumPy

## 📂 Project Structure
- `test_ai.py`: Main entry point for testing AI logic.
- `settings.py`: Configuration and environment variable management.
- `requirements.txt`: List of necessary Python packages.
- `.env`: (Local only) Sensitive API keys and secrets.

## ⚙️ Installation & Setup

### 1. Replit Setup
1. Import this repository into **Replit**.
2. Go to the **Secrets** (Padlock icon) tool in Replit.
3. Add the following keys:
   - `BINANCE_API_KEY`: Your exchange API key.
   - `BINANCE_SECRET_KEY`: Your exchange secret key.

### 2. Run the Bot
To start the bot in the Replit console, run:
```bash
python test_ai.py
