# 📈 Stock Markov Analysis Platform

> **Advanced Markov Chain-based Stock Market Analysis & Forecasting System**

A production-ready, AI-powered platform for stock market analysis using Markov chain models, Monte Carlo simulations, and machine learning. Built for traders, quantitative analysts, and researchers.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)

---

## ✨ Features

### 🎯 Core Capabilities

- **🔮 Markov Chain Modeling**
  - First, second, and third-order Markov chains
  - Ensemble models combining multiple orders
  - Volatility-adjusted state assignment
  - Stationary distribution analysis

- **🎲 Monte Carlo Simulation**
  - Parallel processing with multi-core support
  - Numba JIT compilation (10-100x faster)
  - Configurable confidence intervals
  - Risk metrics (VaR, CVaR, drawdown)

- **📊 Advanced Backtesting**
  - Walk-forward analysis
  - Time-series cross-validation
  - Out-of-sample testing
  - Statistical significance testing

- **🤖 AI-Powered Insights**
  - **Groq Integration** (Recommended): Mixtral-8x7B, Llama 3.1, blazing fast
  - **Gemini Support**: Google's latest models
  - **OpenAI Compatible**: GPT models
  - Context-aware financial analysis

- **💼 Portfolio Management**
  - Real-time position tracking
  - Automated allocation rules
  - Transaction history & P/L tracking
  - Risk-based position sizing

- **📈 Comprehensive Metrics**
  - 27+ performance metrics (Sharpe, Sortino, Calmar, etc.)
  - Maximum drawdown & duration
  - Win/loss statistics
  - Alpha & Beta calculation

- **🧮 Financial Calculators**
  - Kelly Criterion
  - Position sizing
  - Risk/Reward analysis
  - Compound interest
  - Optimal F

- **🎨 Interactive Dashboard**
  - Real-time data visualization
  - Dark/Light theme
  - Multi-ticker comparison
  - Responsive design
  - Export reports (JSON, Markdown, CSV)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip package manager
- (Optional) API keys for AI features

### Installation

```bash
# Clone the repository
git clone https://github.com/EmmaAdjei007/markov-stock-analysis.git
cd markov-stock-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install Numba for 10-100x faster simulations
pip install numba

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Running the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Quick Example

```python
from src.core.data import StockDataFetcher, DataPreprocessor
from src.core.models import MarkovChain
from src.core.simulation import MonteCarloSimulator

# Fetch data
fetcher = StockDataFetcher()
data = fetcher.fetch(["AAPL"], start_date="2020-01-01")

# Preprocess
preprocessor = DataPreprocessor(n_states=5)
processed = preprocessor.process(data)

# Train Markov model
model = MarkovChain(n_states=5)
model.fit(processed["AAPL"]["State"].values)

# Run simulation
simulator = MonteCarloSimulator(n_simulations=1000, n_days=30)
returns_by_state = preprocessor.get_returns_by_state(processed["AAPL"])

result = simulator.run(
    initial_price=data["AAPL"]["Close"][-1],
    transition_matrix=model.get_transition_matrix(),
    returns_by_state=returns_by_state
)

print(f"Median price in 30 days: ${result.median_path[-1]:.2f}")
print(f"95% confidence interval: ${result.percentiles[5][-1]:.2f} - ${result.percentiles[95][-1]:.2f}")
```

---

## 📁 Project Structure

```
stock-markov-analysis/
├── src/
│   ├── core/                      # Core engine modules
│   │   ├── data/                  # Data fetching & preprocessing
│   │   │   ├── fetcher.py        # Yahoo Finance integration
│   │   │   └── preprocessor.py   # State assignment & indicators
│   │   ├── models/                # Markov chain models
│   │   │   └── markov.py         # 1st, 2nd, 3rd order + ensemble
│   │   ├── simulation/            # Monte Carlo engine
│   │   │   └── simulator.py      # Parallel, Numba-optimized
│   │   └── metrics/               # Performance & backtesting
│   │       ├── performance.py    # 27+ metrics
│   │       └── backtesting.py    # Walk-forward, CV, testing
│   ├── calculators/               # Financial calculators
│   │   └── financial_calculators.py
│   ├── config/                    # Configuration management
│   │   └── config_loader.py      # YAML + env vars
│   ├── utils/                     # Utilities
│   │   └── ai_integration.py     # Multi-provider AI
│   └── dashboard/                 # Streamlit components
├── config/
│   └── config.yaml               # Application configuration
├── tests/                        # Comprehensive test suite
├── examples/                     # Usage examples
├── docs/                         # Documentation
├── data/
│   ├── raw/                      # Cached market data
│   └── processed/                # Preprocessed data
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
└── README.md                     # This file
```

---

## 🎯 Usage Examples

### Example 1: Basic Markov Chain Analysis

```python
from src.core import StockDataFetcher, DataPreprocessor, MarkovChain

# Fetch and process data
fetcher = StockDataFetcher()
data = fetcher.fetch(["MSFT"], start_date="2022-01-01")

preprocessor = DataPreprocessor(n_states=5, use_log_returns=False)
processed = preprocessor.process(data, method="quantile")

# Train model
model = MarkovChain(n_states=5, alpha=0.001)
model.fit(processed["MSFT"]["State"].values)

# Analyze model
info = model.get_model_info()
print(f"Most stable state: {info.most_stable_state}")
print(f"Average entropy: {info.entropy_mean:.3f}")

# Predict next state probabilities
current_state = processed["MSFT"]["State"].iloc[-1]
next_probs = model.predict(current_state)
print(f"Next state probabilities: {next_probs}")
```

### Example 2: Advanced Backtesting

```python
from src.core.models import HigherOrderMarkov
from src.core.metrics import Backtester

# Create higher-order model
model = HigherOrderMarkov(n_states=5, order=2)

# Set up backtester
backtester = Backtester(
    model=model,
    n_states=5,
    train_size=0.8
)

# Run walk-forward analysis
result = backtester.run_walk_forward(
    states=states,
    window_size=90,
    step_size=30,
    mode='rolling'
)

# Print results
print(f"Average Hit Rate: {result.overall_metrics['hit_rate']:.2%}")
print(f"RMSE: {result.overall_metrics['rmse']:.2f}")
```

### Example 3: Portfolio Optimization with Kelly Criterion

```python
from src.calculators import kelly_criterion, position_sizing

# Calculate Kelly Criterion
kelly = kelly_criterion(
    win_rate=0.58,
    avg_win=250,
    avg_loss=120
)

print(f"Kelly Percentage: {kelly.kelly_percentage:.1f}%")
print(f"Recommended (half-Kelly): {kelly.recommended_percentage:.1f}%")

# Calculate position size
position = position_sizing(
    account_size=100000,
    risk_per_trade=kelly.recommended_fraction,
    entry_price=150.0,
    stop_loss_price=145.0
)

print(f"Position size: {position.shares} shares")
print(f"Risk amount: ${position.risk_amount:.2f}")
```

### Example 4: AI-Powered Analysis

```python
from src.utils import StockAnalysisAgent

# Initialize agent (Groq recommended for free tier)
agent = StockAnalysisAgent(provider="groq")

# Provide context
context = {
    "ticker": "AAPL",
    "current_price": 185.50,
    "sharpe_ratio": 1.85,
    "max_drawdown": -18.5,
    "var_95": -3.2,
    "prediction": "neutral_to_bullish"
}

# Get analysis
analysis = agent.analyze(
    "Based on these metrics, what's the risk profile and should I adjust my position?",
    context=context
)

print(analysis)
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# AI API Keys
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here

# Application Settings
DEBUG=False
LOG_LEVEL=INFO
```

### Configuration File

Edit `config/config.yaml` to customize:

- Markov model parameters (states, smoothing, order)
- Simulation settings (paths, days, volatility)
- Backtesting configuration (windows, metrics)
- Visualization preferences
- Portfolio rules

---

## 📊 Dashboard Features

### 1. Market Overview Tab
- Historical price charts
- Normalized multi-ticker comparison
- Volatility analysis
- Current market state

### 2. Simulation Tab
- Monte Carlo price projections
- Confidence intervals (90%, 95%, 99%)
- Risk metrics dashboard
- Downloadable results

### 3. Portfolio Management Tab
- **Holdings**: View and manage positions
- **Allocation Rules**: Automated rebalancing
- **Transactions**: Complete audit trail

### 4. AI Analyst Tab
- Context-aware financial analysis
- Conversation history
- Voice synthesis (text-to-speech)
- Export chat history

### 5. Tools & Calculators
- **Kelly Criterion** calculator
- **Position Sizing** tool
- **Risk/Reward** analyzer
- **Compound Interest** calculator

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_markov_model.py -v
```

---

## 📈 Performance

### Benchmarks (Apple M1, 1000 simulations × 30 days)

| Configuration | Time | Speedup |
|---------------|------|---------|
| Pure Python | 12.5s | 1x |
| NumPy Optimized | 2.1s | 6x |
| **Numba JIT** | **0.15s** | **83x** |
| Numba + Parallel | 0.08s | 156x |

### Memory Usage

- Typical: 50-200 MB
- Large simulations (10k paths): ~500 MB
- Scales linearly with simulation count

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 mypy pytest-cov

# Run code formatters
black src/ tests/
flake8 src/ tests/

# Type checking
mypy src/
```

---

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. It is not financial advice and should not be used as the sole basis for investment decisions.

- Past performance does not guarantee future results
- Markov models have limitations and assumptions
- Always conduct your own due diligence
- Consult with qualified financial advisors
- Risk of capital loss exists in all investments

The authors and contributors are not responsible for any financial losses incurred through the use of this software.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Acknowledgments

- **Yahoo Finance** for market data (via yfinance)
- **Groq** for fast, free AI inference
- **Streamlit** for the amazing web framework
- **NumPy/SciPy** community for scientific computing tools
- All open-source contributors

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/EmmaAdjei007/stock-markov-analysis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/EmmaAdjei007/stock-markov-analysis/discussions)

---

## 🗺️ Roadmap

- [ ] Real-time data streaming via WebSocket
- [ ] Mobile-responsive dashboard
- [ ] REST API with FastAPI
- [ ] Database persistence (PostgreSQL)
- [ ] Advanced ML models (LSTM, Transformer)
- [ ] Multi-timeframe analysis
- [ ] Automated strategy backtesting
- [ ] Cloud deployment templates (AWS, GCP, Azure)

---

<p align="center">
  Made with ❤️ for the quantitative finance community
</p>

<p align="center">
  <a href="#-stock-markov-analysis-platform">Back to Top ⬆️</a>
</p>


