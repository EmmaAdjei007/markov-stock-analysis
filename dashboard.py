import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict
from streamlit.components.v1 import html
from uuid import uuid4
import google.generativeai as genai

# Setup logging
logging.basicConfig(level=logging.INFO)

# ------------------ LOCAL MODULE IMPORTS ------------------
from src.core.data.fetcher import get_stock_data
from src.core.data.preprocessor import assign_states
from src.core.models.markov import get_transition_matrix
from src.core.simulation.simulator import simulate_price_path

# ------------------ CONFIGURATION ------------------
try:
    GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except:
    GOOGLE_API_KEY = None

# ------------------ MODERN DARK THEME ------------------
def apply_modern_theme():
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* Root Variables */
        :root {
            --primary-bg: #0a0e27;
            --secondary-bg: #1a1f3a;
            --card-bg: #1e2542;
            --accent-primary: #00d4ff;
            --accent-secondary: #7c3aed;
            --accent-success: #10b981;
            --accent-warning: #f59e0b;
            --accent-danger: #ef4444;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --border-color: rgba(148, 163, 184, 0.1);
            --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --gradient-secondary: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --gradient-success: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }

        /* Main App Background */
        .stApp {
            background: var(--primary-bg);
            background-image:
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.1) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.1) 0px, transparent 50%);
            font-family: 'Inter', sans-serif;
        }

        /* Main container */
        .main {
            background: transparent;
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }

        h1 {
            font-size: 2.5rem !important;
            background: linear-gradient(135deg, #667eea 0%, #00d4ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* All text elements */
        p, span, div, label, li, td, th {
            color: var(--text-primary) !important;
        }

        /* Markdown text */
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] span,
        [data-testid="stMarkdownContainer"] li {
            color: var(--text-primary) !important;
        }

        /* Labels for inputs */
        label {
            color: var(--text-primary) !important;
            font-weight: 500 !important;
        }

        /* Selectbox and input text */
        .stSelectbox label,
        .stMultiSelect label,
        .stTextInput label,
        .stNumberInput label,
        .stSlider label,
        .stDateInput label {
            color: var(--text-primary) !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: var(--secondary-bg) !important;
            border-right: 1px solid var(--border-color);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--text-primary) !important;
        }

        /* Sidebar labels and text */
        [data-testid="stSidebar"] label {
            color: var(--text-primary) !important;
        }

        [data-testid="stSidebar"] .stMarkdown {
            color: var(--text-primary) !important;
        }

        /* Sidebar help icons - make them very visible */
        [data-testid="stSidebar"] [data-testid="stTooltipIcon"] {
            color: var(--accent-primary) !important;
            opacity: 1 !important;
            font-size: 1.2rem !important;
        }

        [data-testid="stSidebar"] [data-testid="stTooltipIcon"]:hover {
            color: var(--accent-success) !important;
        }

        /* Sidebar metric values */
        [data-testid="stSidebar"] [data-testid="stMetricValue"] {
            color: var(--accent-primary) !important;
        }

        [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
            color: var(--text-primary) !important;
        }

        /* Cards and containers */
        .metric-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            border-color: var(--accent-primary);
            box-shadow: 0 8px 32px rgba(0, 212, 255, 0.15);
        }

        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--gradient-success);
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: var(--secondary-bg);
            padding: 0.5rem;
            border-radius: 12px;
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: var(--text-secondary);
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            border: none;
            transition: all 0.3s ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: var(--card-bg);
            color: var(--text-primary);
        }

        .stTabs [aria-selected="true"] {
            background: var(--gradient-primary) !important;
            color: white !important;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            color: var(--accent-primary) !important;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
        }

        [data-testid="stMetricLabel"] {
            color: var(--text-secondary) !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Buttons */
        .stButton > button {
            background: var(--gradient-primary);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }

        /* Inputs */
        .stSelectbox, .stMultiSelect, .stTextInput, .stNumberInput {
            background: var(--card-bg);
        }

        .stSelectbox > div > div,
        .stMultiSelect > div > div,
        .stTextInput > div > div,
        .stNumberInput > div > div {
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color);
            border-radius: 10px;
            color: var(--text-primary) !important;
        }

        /* Input text and values */
        input, textarea, select {
            color: var(--text-primary) !important;
            background: var(--card-bg) !important;
        }

        /* Dropdown options */
        [role="option"] {
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
        }

        [role="option"]:hover {
            background: var(--accent-primary) !important;
            color: white !important;
        }

        /* Multi-select tags */
        [data-baseweb="tag"] {
            background: var(--accent-primary) !important;
            color: white !important;
        }

        /* Date input */
        .stDateInput > div > div > input {
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color);
            border-radius: 10px;
            color: var(--text-primary) !important;
        }

        /* Sliders */
        .stSlider > div > div > div {
            background: var(--gradient-primary);
        }

        .stSlider label {
            color: var(--text-primary) !important;
        }

        .stSlider [data-testid="stTickBarMin"],
        .stSlider [data-testid="stTickBarMax"],
        .stSlider [data-baseweb="slider"] div {
            color: var(--text-primary) !important;
        }

        /* Dataframes and Tables */
        [data-testid="stDataFrame"] {
            background: var(--card-bg) !important;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }

        [data-testid="stDataFrame"] table {
            color: var(--text-primary) !important;
        }

        [data-testid="stDataFrame"] th {
            background: var(--secondary-bg) !important;
            color: var(--accent-primary) !important;
            font-weight: 600 !important;
        }

        [data-testid="stDataFrame"] td {
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
        }

        /* Table elements */
        table {
            color: var(--text-primary) !important;
        }

        th {
            background: var(--secondary-bg) !important;
            color: var(--accent-primary) !important;
            font-weight: 600 !important;
        }

        td {
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
        }

        /* Streamlit's styled dataframe */
        .dataframe {
            color: var(--text-primary) !important;
        }

        .dataframe th {
            background: var(--secondary-bg) !important;
            color: var(--accent-primary) !important;
        }

        .dataframe td {
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
        }

        /* Expander */
        .streamlit-expanderHeader {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            color: var(--text-primary);
            font-weight: 600;
        }

        /* Chat messages */
        [data-testid="stChatMessage"] {
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color);
            border-radius: 12px;
        }

        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] span,
        [data-testid="stChatMessage"] div {
            color: var(--text-primary) !important;
        }

        /* Chat input */
        [data-testid="stChatInput"] textarea {
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color);
        }

        /* Spinner */
        .stSpinner > div {
            border-top-color: var(--accent-primary) !important;
        }

        /* Success/Info/Warning/Error boxes */
        .stSuccess {
            background: rgba(16, 185, 129, 0.1);
            border-left: 4px solid var(--accent-success);
            color: var(--accent-success);
        }

        .stInfo {
            background: rgba(0, 212, 255, 0.1);
            border-left: 4px solid var(--accent-primary);
            color: var(--accent-primary);
        }

        .stWarning {
            background: rgba(245, 158, 11, 0.1);
            border-left: 4px solid var(--accent-warning);
            color: var(--accent-warning);
        }

        .stError {
            background: rgba(239, 68, 68, 0.1);
            border-left: 4px solid var(--accent-danger);
            color: var(--accent-danger);
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }

        ::-webkit-scrollbar-track {
            background: var(--secondary-bg);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--card-bg);
            border-radius: 5px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent-primary);
        }

        /* Sample question cards */
        .sample-question {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            color: var(--text-secondary);
            font-size: 0.9rem;
            text-align: center;
        }

        .sample-question:hover {
            background: var(--gradient-primary);
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }

        /* Code blocks */
        code {
            background: var(--secondary-bg) !important;
            color: var(--accent-primary) !important;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
        }

        pre {
            background: var(--secondary-bg) !important;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
        }

        pre code {
            color: var(--text-primary) !important;
        }

        /* Links */
        a {
            color: var(--accent-primary) !important;
            text-decoration: none;
        }

        a:hover {
            color: var(--accent-secondary) !important;
            text-decoration: underline;
        }

        /* Captions and small text */
        .caption, small {
            color: var(--text-secondary) !important;
        }

        /* Tooltips and help icons - multiple selectors for maximum visibility */
        [data-testid="stTooltipIcon"],
        .stTooltipIcon,
        [data-baseweb="tooltip"] button,
        button[aria-label*="help"] {
            color: var(--accent-primary) !important;
            opacity: 1 !important;
            visibility: visible !important;
            display: inline-flex !important;
        }

        [data-testid="stTooltipIcon"]:hover,
        .stTooltipIcon:hover,
        [data-baseweb="tooltip"] button:hover {
            color: var(--accent-success) !important;
            opacity: 1 !important;
            transform: scale(1.1);
        }

        /* Help icon SVG */
        [data-testid="stTooltipIcon"] svg,
        .stTooltipIcon svg {
            fill: var(--accent-primary) !important;
            color: var(--accent-primary) !important;
        }

        /* Tooltip content/popover - comprehensive styling */
        [role="tooltip"],
        [data-baseweb="popover"],
        .stTooltip,
        div[data-baseweb="popover"] > div {
            background: var(--card-bg) !important;
            background-color: var(--card-bg) !important;
            border: 1px solid var(--accent-primary) !important;
            color: var(--text-primary) !important;
            border-radius: 8px !important;
            padding: 0.75rem !important;
            box-shadow: 0 4px 20px rgba(0, 212, 255, 0.3) !important;
            z-index: 9999 !important;
        }

        /* All text inside tooltips */
        [role="tooltip"] *,
        [data-baseweb="popover"] *,
        .stTooltip *,
        div[data-baseweb="popover"] * {
            color: var(--text-primary) !important;
            background: transparent !important;
        }

        [role="tooltip"] p,
        [role="tooltip"] span,
        [role="tooltip"] div,
        [data-baseweb="popover"] p,
        [data-baseweb="popover"] span,
        [data-baseweb="popover"] div {
            color: var(--text-primary) !important;
            font-size: 0.9rem !important;
            line-height: 1.5 !important;
        }

        /* Tooltip arrow */
        [data-baseweb="popover"] [data-popper-arrow] {
            display: none !important;
        }

        /* Radio buttons and checkboxes */
        [data-testid="stRadio"] label,
        [data-testid="stCheckbox"] label {
            color: var(--text-primary) !important;
        }

        /* File uploader */
        [data-testid="stFileUploader"] label {
            color: var(--text-primary) !important;
        }

        /* Download button */
        [data-testid="stDownloadButton"] button {
            background: var(--gradient-success) !important;
            color: white !important;
        }

        /* Widget labels */
        .stRadio > label,
        .stCheckbox > label,
        .stFileUploader > label {
            color: var(--text-primary) !important;
            font-weight: 500 !important;
        }

        /* All divs inside main content */
        .main div {
            color: var(--text-primary);
        }

        /* Ensure all text in containers is visible */
        .element-container div,
        .element-container span,
        .element-container p {
            color: var(--text-primary) !important;
        }
    </style>
    """, unsafe_allow_html=True)

# ------------------ UTILITY FUNCTIONS ------------------
def calculate_performance_metrics(simulations=None, df=None, current_price=None, forecast_days=100, **kwargs):
    """Calculate performance metrics for simulations."""
    metrics = {}
    summary_stats = []

    if simulations is not None and df is not None and current_price is not None:
        final_prices = simulations[:, -1]
        returns = (final_prices - current_price) / current_price
        historical_vol = df['Daily_Return'].std() * np.sqrt(252) * 100
        sim_vol = np.std(simulations) / np.mean(simulations) * 100

        metrics['Expected Return'] = np.mean(returns) * 100
        metrics['Volatility'] = np.std(returns) * 100
        metrics['VaR (95%)'] = np.percentile(returns, 5) * 100
        metrics['Best Case (5%)'] = np.percentile(returns, 95) * 100

        summary_stats = [
            ['Current Price', f"${current_price:.2f}"],
            ['Forecast Horizon', f"{forecast_days} days"],
            ['Mean Predicted Price', f"${np.mean(final_prices):.2f}"],
            ['Price Range', f"${np.min(final_prices):.2f} - ${np.max(final_prices):.2f}"],
            ['Historical Volatility', f"{historical_vol:.2f}%"],
            ['Simulated Volatility', f"{sim_vol:.2f}%"],
            ['90% CI', f"${np.percentile(final_prices, 5):.2f} - ${np.percentile(final_prices, 95):.2f}"],
            ['95% CI', f"${np.percentile(final_prices, 2.5):.2f} - ${np.percentile(final_prices, 97.5):.2f}"]
        ]

    return metrics, summary_stats

def get_realtime_price(ticker: str) -> float:
    """Get real-time price with fallback and caching."""
    try:
        # Use history method which is more reliable and less rate-limited
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])

        # Fallback to info (may be rate limited)
        info = stock.info
        return float(info.get('currentPrice', 0) or info.get('regularMarketPrice', 0) or info.get('previousClose', 0))
    except Exception:
        # Silently return 0 to avoid cluttering UI with warnings
        return 0.0

def load_history():
    HISTORY_FILE = "chat_history.json"
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_history(history):
    HISTORY_FILE = "chat_history.json"
    with open(HISTORY_FILE, "w") as f:
        json.dump(history[-100:], f)

def get_gemini_response(prompt: str) -> str:
    if not GOOGLE_API_KEY:
        return "⚠️ Gemini API key not configured. Please add GEMINI_API_KEY to your secrets."
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"AI Error: {e}")
        return "Failed to get a response from Gemini API."

def stock_analysis_agent(query: str, context: str) -> str:
    system_prompt = f"""**Role**: Senior Quantitative Analyst
**Context**: {context}
**Task**: Analyze: {query}

Provide comprehensive analysis with:
- Key insights and patterns
- Risk assessment
- Market regime analysis
- Actionable recommendations

Keep response concise and data-driven."""
    return get_gemini_response(system_prompt)

# ------------------ MAIN APP ------------------
def main():
    st.set_page_config(
        page_title="Stock Markov Analytics",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    apply_modern_theme()

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = load_history()
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        st.markdown("##### 📊 Asset Selection")
        tickers = st.multiselect(
            "Select Assets",
            ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"],
            default=["AAPL", "MSFT"],
            help="Choose stocks to analyze"
        )

        st.markdown("##### 📅 Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", pd.to_datetime("2020-01-01"))
        with col2:
            end_date = st.date_input("End", pd.to_datetime(datetime.today()))

        st.markdown("##### 🔧 Model Parameters")
        n_states = st.slider("Market Regimes", 3, 7, 5, help="Number of market states")
        n_simulations = st.slider("Simulation Paths", 100, 5000, 1000, step=100)
        sim_duration = st.slider("Forecast Days", 10, 90, 30, step=5)

        st.divider()

        st.markdown("##### 💹 Real-Time Prices")
        if tickers:
            for ticker in tickers[:4]:  # Limit to 4 for space
                try:
                    price = get_realtime_price(ticker)
                    st.metric(ticker, f"${price:.2f}", delta=None)
                except:
                    pass

    # Header
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='margin: 0;'>📊 Stock Markov Analytics</h1>
        <p style='color: var(--text-secondary); font-size: 1.1rem; margin-top: 0.5rem;'>
            Advanced market analysis powered by Markov Chain models & Monte Carlo simulations
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Analysis context
    analysis_context = f"""
    Assets: {', '.join(tickers) if tickers else 'None'}
    Period: {start_date} to {end_date}
    Regimes: {n_states} | Simulations: {n_simulations} | Forecast: {sim_duration} days
    """

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Market Overview",
        "🎲 Monte Carlo Simulation",
        "💬 AI Analyst",
        "🧮 Financial Tools"
    ])

    # TAB 1: Market Overview
    with tab1:
        if not tickers:
            st.info("👈 Select assets from the sidebar to begin analysis")
        else:
            try:
                with st.spinner("Loading market data..."):
                    raw_data = get_stock_data(tickers, start_date, end_date)
                    processed_data = assign_states(raw_data, n_states)

                st.markdown("### 📊 Market Metrics")
                cols = st.columns(len(tickers))
                for idx, ticker in enumerate(tickers):
                    with cols[idx]:
                        latest = raw_data[ticker]['Close'].iloc[-1]
                        volatility = processed_data[ticker]['Daily_Return'].std() * np.sqrt(252)
                        avg_return = processed_data[ticker]['Daily_Return'].mean() * 252

                        st.markdown(f"""
                        <div class='metric-card'>
                            <h3 style='margin: 0 0 1rem 0; color: var(--accent-primary);'>{ticker}</h3>
                            <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                                <span style='color: var(--text-secondary);'>Price</span>
                                <span style='color: var(--text-primary); font-weight: 600;'>${latest:.2f}</span>
                            </div>
                            <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                                <span style='color: var(--text-secondary);'>Volatility</span>
                                <span style='color: var(--accent-warning); font-weight: 600;'>{volatility*100:.2f}%</span>
                            </div>
                            <div style='display: flex; justify-content: space-between;'>
                                <span style='color: var(--text-secondary);'>Avg Return</span>
                                <span style='color: {"var(--accent-success)" if avg_return > 0 else "var(--accent-danger)"}; font-weight: 600;'>{avg_return*100:.2f}%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("### 📉 Price Comparison")
                # Create Plotly chart
                fig = go.Figure()
                for ticker in tickers:
                    norm_price = processed_data[ticker]['Close'] / processed_data[ticker]['Close'].iloc[0] * 100
                    fig.add_trace(go.Scatter(
                        x=processed_data[ticker].index,
                        y=norm_price,
                        name=ticker,
                        mode='lines',
                        line=dict(width=2)
                    ))

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(30,37,66,0.5)',
                    xaxis_title="Date",
                    yaxis_title="Normalized Price (Base = 100)",
                    hovermode='x unified',
                    height=500,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        bgcolor='rgba(30,37,66,0.8)',
                        bordercolor='rgba(148, 163, 184, 0.3)',
                        borderwidth=1,
                        font=dict(
                            color='#e2e8f0',
                            size=12
                        )
                    ),
                    font=dict(
                        color='#e2e8f0'
                    ),
                    xaxis=dict(
                        title_font=dict(color='#e2e8f0'),
                        tickfont=dict(color='#e2e8f0'),
                        gridcolor='rgba(148, 163, 184, 0.1)'
                    ),
                    yaxis=dict(
                        title_font=dict(color='#e2e8f0'),
                        tickfont=dict(color='#e2e8f0'),
                        gridcolor='rgba(148, 163, 184, 0.1)'
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Data loading failed: {str(e)}")

    # TAB 2: Monte Carlo Simulation
    with tab2:
        if not tickers:
            st.info("👈 Select assets from the sidebar to begin simulation")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_ticker = st.selectbox("Select Asset for Simulation", tickers)
            with col2:
                st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
                run_sim = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

            if run_sim:
                try:
                    with st.spinner("Running Monte Carlo simulation..."):
                        raw_data = get_stock_data(tickers, start_date, end_date)
                        processed_data = assign_states(raw_data, n_states)
                        transition_matrix, _ = get_transition_matrix(
                            processed_data[selected_ticker]['State'].values, n_states
                        )

                        simulation_results = []
                        for i in range(n_simulations):
                            sim_result = simulate_price_path(
                                processed_data[selected_ticker]['Close'].iloc[-1],
                                transition_matrix,
                                [processed_data[selected_ticker][processed_data[selected_ticker]['State'] == s]['Daily_Return'].values
                                 for s in range(n_states)],
                                sim_duration
                            )
                            if isinstance(sim_result, tuple):
                                sim_result = sim_result[0]
                            simulation_results.append(np.array(sim_result))

                        simulations = np.stack(simulation_results, axis=0)

                        # Calculate percentiles
                        percentiles = {
                            'p5': np.percentile(simulations, 5, axis=0),
                            'p25': np.percentile(simulations, 25, axis=0),
                            'p50': np.percentile(simulations, 50, axis=0),
                            'p75': np.percentile(simulations, 75, axis=0),
                            'p95': np.percentile(simulations, 95, axis=0),
                        }

                        # Create visualization
                        fig = go.Figure()

                        # Add confidence interval bands
                        days = np.arange(sim_duration + 1)
                        fig.add_trace(go.Scatter(
                            x=np.concatenate([days, days[::-1]]),
                            y=np.concatenate([percentiles['p95'], percentiles['p5'][::-1]]),
                            fill='toself',
                            fillcolor='rgba(0, 212, 255, 0.1)',
                            line=dict(color='rgba(0, 212, 255, 0)'),
                            name='90% CI',
                            showlegend=True
                        ))

                        fig.add_trace(go.Scatter(
                            x=np.concatenate([days, days[::-1]]),
                            y=np.concatenate([percentiles['p75'], percentiles['p25'][::-1]]),
                            fill='toself',
                            fillcolor='rgba(0, 212, 255, 0.2)',
                            line=dict(color='rgba(0, 212, 255, 0)'),
                            name='50% CI',
                            showlegend=True
                        ))

                        # Add median line
                        fig.add_trace(go.Scatter(
                            x=days,
                            y=percentiles['p50'],
                            mode='lines',
                            line=dict(color='#00d4ff', width=3),
                            name='Median Forecast'
                        ))

                        # Add sample paths (light)
                        for i in range(min(50, n_simulations)):
                            fig.add_trace(go.Scatter(
                                x=days,
                                y=simulations[i],
                                mode='lines',
                                line=dict(color='rgba(124, 58, 237, 0.05)', width=1),
                                showlegend=False,
                                hoverinfo='skip'
                            ))

                        fig.update_layout(
                            title=f"{selected_ticker} - {sim_duration} Day Price Forecast",
                            template="plotly_dark",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(30,37,66,0.5)',
                            xaxis_title="Trading Days",
                            yaxis_title="Price ($)",
                            hovermode='x unified',
                            height=600,
                            legend=dict(
                                bgcolor='rgba(30,37,66,0.8)',
                                bordercolor='rgba(148, 163, 184, 0.3)',
                                borderwidth=1,
                                font=dict(
                                    color='#e2e8f0',
                                    size=12
                                )
                            ),
                            font=dict(
                                color='#e2e8f0'
                            ),
                            title_font=dict(
                                color='#e2e8f0',
                                size=18
                            ),
                            xaxis=dict(
                                title_font=dict(color='#e2e8f0'),
                                tickfont=dict(color='#e2e8f0'),
                                gridcolor='rgba(148, 163, 184, 0.1)'
                            ),
                            yaxis=dict(
                                title_font=dict(color='#e2e8f0'),
                                tickfont=dict(color='#e2e8f0'),
                                gridcolor='rgba(148, 163, 184, 0.1)'
                            )
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Calculate metrics
                        metrics, summary_stats = calculate_performance_metrics(
                            simulations=simulations,
                            df=processed_data[selected_ticker],
                            current_price=processed_data[selected_ticker]['Close'].iloc[-1],
                            forecast_days=sim_duration
                        )
                        st.session_state.metrics = metrics

                        # Display metrics
                        st.markdown("### 📊 Risk Metrics")
                        cols = st.columns(4)
                        metric_data = [
                            ("Expected Return", metrics.get('Expected Return', 0), "📈"),
                            ("Volatility", metrics.get('Volatility', 0), "📊"),
                            ("VaR (95%)", metrics.get('VaR (95%)', 0), "⚠️"),
                            ("Best Case", metrics.get('Best Case (5%)', 0), "🎯")
                        ]

                        for idx, (label, value, icon) in enumerate(metric_data):
                            with cols[idx]:
                                st.metric(f"{icon} {label}", f"{value:.2f}%")

                        # Summary table
                        if summary_stats:
                            st.markdown("### 📋 Simulation Summary")
                            summary_df = pd.DataFrame(summary_stats, columns=["Metric", "Value"])
                            st.dataframe(summary_df, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"❌ Simulation failed: {str(e)}")

    # TAB 3: AI Analyst
    with tab3:
        st.markdown("### 💬 AI Financial Analyst")

        if not GOOGLE_API_KEY:
            st.warning("⚠️ Gemini API key not configured. Please add GEMINI_API_KEY to your Streamlit secrets.")

        # Sample questions
        st.markdown("##### Quick Questions")
        sample_questions = [
            "Analyze current market volatility",
            "Compare risk profiles of selected assets",
            f"What's the outlook for {tickers[0] if tickers else 'AAPL'}?",
            "Explain the current market regime",
            "What are key risk factors?",
            "Suggest portfolio adjustments"
        ]

        cols = st.columns(3)
        for idx, question in enumerate(sample_questions[:6]):
            with cols[idx % 3]:
                if st.button(question, use_container_width=True, key=f"q_{idx}"):
                    st.session_state.selected_question = question
                    st.experimental_rerun()

        st.divider()

        # Chat history
        st.markdown("##### Conversation")
        for msg in st.session_state.chat_history[-10:]:  # Show last 10 messages
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # TAB 4: Financial Tools
    with tab4:
        st.markdown("### 🧮 Financial Calculators")

        calc_tab1, calc_tab2, calc_tab3 = st.tabs([
            "💰 Compound Interest",
            "📊 Position Sizing",
            "⚖️ Risk/Reward"
        ])

        with calc_tab1:
            st.markdown("##### Calculate Investment Growth")
            col1, col2 = st.columns(2)
            with col1:
                principal = st.number_input("Initial Investment ($)", value=10000.0, step=1000.0)
                rate = st.number_input("Annual Return (%)", value=8.0, step=0.5)
                years = st.number_input("Time Period (Years)", value=10, step=1)
            with col2:
                compound_freq = st.selectbox(
                    "Compound Frequency",
                    ["Annually", "Semi-annually", "Quarterly", "Monthly", "Daily"]
                )
                freq_map = {"Annually": 1, "Semi-annually": 2, "Quarterly": 4, "Monthly": 12, "Daily": 365}
                n = freq_map[compound_freq]

            if st.button("Calculate", key="calc_compound"):
                result = principal * (1 + (rate / 100) / n) ** (n * years)
                total_return = result - principal
                roi = (total_return / principal) * 100

                col1, col2, col3 = st.columns(3)
                col1.metric("Future Value", f"${result:,.2f}")
                col2.metric("Total Return", f"${total_return:,.2f}")
                col3.metric("ROI", f"{roi:.2f}%")

        with calc_tab2:
            st.markdown("##### Calculate Optimal Position Size")
            col1, col2 = st.columns(2)
            with col1:
                account_size = st.number_input("Account Size ($)", value=100000.0, step=5000.0)
                risk_percent = st.number_input("Risk Per Trade (%)", value=1.0, step=0.1, max_value=5.0)
            with col2:
                entry_price = st.number_input("Entry Price ($)", value=100.0, step=1.0)
                stop_price = st.number_input("Stop Loss ($)", value=95.0, step=1.0)

            if st.button("Calculate Position", key="calc_position"):
                risk_amount = account_size * (risk_percent / 100)
                risk_per_share = abs(entry_price - stop_price)
                shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
                position_value = shares * entry_price

                col1, col2, col3 = st.columns(3)
                col1.metric("Shares to Buy", f"{shares}")
                col2.metric("Position Value", f"${position_value:,.2f}")
                col3.metric("Max Risk", f"${risk_amount:,.2f}")

        with calc_tab3:
            st.markdown("##### Evaluate Trade Risk/Reward")
            col1, col2 = st.columns(2)
            with col1:
                entry = st.number_input("Entry Price ($)", value=100.0, step=1.0, key="rr_entry")
                stop_loss = st.number_input("Stop Loss ($)", value=95.0, step=1.0, key="rr_stop")
            with col2:
                take_profit = st.number_input("Take Profit ($)", value=115.0, step=1.0, key="rr_target")

            if st.button("Calculate R/R", key="calc_rr"):
                risk = abs(entry - stop_loss)
                reward = abs(take_profit - entry)
                rr_ratio = reward / risk if risk > 0 else 0
                breakeven_wr = (1 / (1 + rr_ratio)) * 100 if rr_ratio > 0 else 0

                col1, col2, col3 = st.columns(3)
                col1.metric("Risk/Reward Ratio", f"1:{rr_ratio:.2f}")
                col2.metric("Potential Reward", f"${reward:.2f}")
                col3.metric("Breakeven Win Rate", f"{breakeven_wr:.1f}%")

    # Chat input (must be outside tabs)
    prompt = st.chat_input("Ask the AI analyst...")
    if hasattr(st.session_state, 'selected_question') and st.session_state.get('selected_question'):
        prompt = st.session_state.selected_question
        st.session_state.selected_question = None

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        try:
            response = stock_analysis_agent(prompt, analysis_context)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            save_history(st.session_state.chat_history)
            st.experimental_rerun()
        except Exception as e:
            st.error(f"❌ AI Analysis Failed: {str(e)}")
            st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})

if __name__ == "__main__":
    main()
