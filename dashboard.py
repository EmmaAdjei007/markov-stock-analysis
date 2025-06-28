import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import altair as alt
import seaborn as sns
import requests
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict
from streamlit.components.v1 import html
from uuid import uuid4
import google.generativeai as genai

# Setup logging
logging.basicConfig(level=logging.INFO)

# ------------------ LOCAL MODULE IMPORTS ------------------
from src.data_fetcher import get_stock_data
from src.data_preprocessor import assign_states
from src.markov_model import get_transition_matrix
from src.simulation import simulate_price_path, calculate_performance_metrics

# ------------------ GOOGLE GEMINI CONFIGURATION ------------------
GOOGLE_API_KEY = os.getenv("API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# ------------------ THEME CONFIGURATION ------------------
DARK_THEME = {
    "primary": "#00ff88",
    "secondary": "#00b4d8",
    "background": "#121212",
    "surface": "#1e1e1e",
    "text": "#ffffff",
    "accent": "#ff6b6b"
}

LIGHT_THEME = {
    "primary": "#2a9d8f",
    "secondary": "#264653",
    "background": "#f8f9fa",
    "surface": "#ffffff",
    "text": "#2b2d42",
    "accent": "#ef233c"
}

def apply_theme(theme: Dict):
    st.markdown(f"""
    <style>
        :root {{
            --primary: {theme['primary']};
            --secondary: {theme['secondary']};
            --background: {theme['background']};
            --surface: {theme['surface']};
            --text: {theme['text']};
            --accent: {theme['accent']};
        }}
        .main {{ background-color: var(--background); color: var(--text); }}
        .metric-card {{ background: var(--surface); border: 1px solid rgba(255,255,255,0.1); }}
        .stSelectbox, .stTextInput, .stDateInput {{ border: 1px solid var(--secondary) !important; }}
        h1, h2, h3 {{ color: var(--primary) !important; }}
        [data-testid="stCheckbox"] {{
            opacity: 1 !important;
        }}
        [data-testid="stCheckbox"] > label {{
            color: {theme['text']} !important;
            opacity: 1 !important;
            visibility: visible !important;
        }}
    </style>
    """, unsafe_allow_html=True)

# ------------------ UTILITY FUNCTIONS ------------------
def get_realtime_price(ticker: str) -> float:
    stock = yf.Ticker(ticker)
    return stock.fast_info["lastPrice"]

def speak_button(text: str):
    button_id = "speak-btn-" + str(uuid4()).replace("-", "")
    clean_text = text.replace('`', '\\`').replace('"', '\\"').replace('\n', ' ')
    script = f"""
    <button id="{button_id}" 
            style="background-color: #555; color: white; border: none; 
                   padding: 0.5rem 1rem; margin-top: 5px; 
                   border-radius: 6px; cursor: pointer;">
      🔊 Speak
    </button>
    <script>
        var btn = document.getElementById("{button_id}");
        btn.onclick = function() {{
            var utterance = new SpeechSynthesisUtterance(`{clean_text}`);
            utterance.rate = 1.0;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(utterance);
        }};
    </script>
    """
    return html(script, height=40)

def custom_voice_controls():
    if 'voice_enabled' not in st.session_state:
        st.session_state.voice_enabled = False
    if 'auto_speak' not in st.session_state:
        st.session_state.auto_speak = False

    voice_enabled_id = f"voice-enabled-{str(uuid4())[:8]}"
    auto_speak_id = f"auto-speak-{str(uuid4())[:8]}"

    html_code = f"""
    <div style="margin: 1rem 0; padding: 1rem; background: rgba(0,0,0,0.1); border-radius: 8px;">
        <h3 style="margin-bottom: 1rem; color: white;">Voice Settings</h3>
        <div style="display: flex; gap: 2rem;">
            <div style="display: flex; align-items: center;">
                <input type="checkbox" id="{voice_enabled_id}" style="margin-right: 8px; width: 18px; height: 18px;"
                {"checked" if st.session_state.voice_enabled else ""}>
                <label for="{voice_enabled_id}" style="color: white; font-size: 16px;">🔊 Enable Voice</label>
            </div>
            <div style="display: flex; align-items: center;">
                <input type="checkbox" id="{auto_speak_id}" style="margin-right: 8px; width: 18px; height: 18px;"
                {"checked" if st.session_state.auto_speak else ""}>
                <label for="{auto_speak_id}" style="color: white; font-size: 16px;">🗣️ Auto-speak</label>
            </div>
        </div>
    </div>
    <script>
    document.getElementById('{voice_enabled_id}').addEventListener('change', function(e) {{
        window.parent.postMessage({{
            type: 'streamlit:setComponentValue',
            value: e.target.checked,
            key: 'voice_enabled'
        }}, '*');
    }});
    document.getElementById('{auto_speak_id}').addEventListener('change', function(e) {{
        window.parent.postMessage({{
            type: 'streamlit:setComponentValue',
            value: e.target.checked,
            key: 'auto_speak'
        }}, '*');
    }});
    </script>
    """
    st.components.v1.html(html_code, height=150)
    return st.session_state.voice_enabled, st.session_state.auto_speak

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


def generate_report(history, context, metrics=None):
    report = [
        "# QuantPro Analysis Report\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        "## Analysis Context\n",
        context.replace("**", "") + "\n",
        "## Simulation Metrics\n"
    ]
    if metrics:
        report.extend([
            f"- **Expected Return:** {metrics.get('Simulated Mean Return', 0):.2f}%",
            f"- **Simulated Volatility:** {metrics.get('Simulated Volatility', 0):.2f}%",
            f"- **Value at Risk (95%):** {metrics.get('Simulated VaR (95%)', 0):.2f}%",
            f"- **Best Case (5%):** {metrics.get('Simulated Best Case (5%)', 0):.2f}%\n"
        ])
    report.append("## Conversation History\n")
    for msg in history:
        report.append(f"**{msg['role'].title()}**: {msg['content']}\n")
    return "\n".join(report)


def get_gemini_response(prompt: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"AI Error: {e}")
        logging.error(f"AI Error: {e}")
        return "Failed to get a response from Gemini API."

def stock_analysis_agent(query: str, context: str) -> str:
    system_prompt = f"""**Role**: Senior Financial Analyst at JP Morgan  
**Context**: {context}  
**Task**: Analyze this query: {query}  
Provide institutional-grade analysis with:
- Technical pattern recognition  
- Risk/reward assessment  
- Volatility analysis  
- Comparative market analysis  

**Response Guidelines**:
1. Start with key takeaways in bold  
2. Use bullet points for complex analysis  
3. Highlight risk factors in _italic_  
4. Maintain SEC compliance standards  
5. Reference current market conditions  
"""
    return get_gemini_response(system_prompt)

def init_ai_analyst_tab():
    st.markdown("""
    <style>
    #ai-analyst-tab [data-testid="stCheckbox"] *,
    #ai-analyst-tab .st-emotion-cache-ue6h4q p, 
    #ai-analyst-tab .st-emotion-cache-916qix p, 
    #ai-analyst-tab .st-emotion-cache-1qg75dn p,
    #ai-analyst-tab [data-testid="stCheckbox"] label,
    #ai-analyst-tab [data-testid="stCheckbox"] span,
    #ai-analyst-tab [data-testid="stCheckbox"] p,
    #ai-analyst-tab [data-testid="stCheckbox"] div {
        color: white !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.components.v1.html("""
    <div id="checkbox-fixer"></div>
    <script>
    function fixCheckboxes() {
        const checkboxes = document.querySelectorAll('[data-testid="stCheckbox"]');
        checkboxes.forEach(checkbox => {
            const elements = checkbox.querySelectorAll('*');
            elements.forEach(el => {
                el.style.setProperty('color', 'white', 'important');
                el.style.setProperty('opacity', '1', 'important');
                el.style.setProperty('visibility', 'visible', 'important');
            });
        });
    }
    fixCheckboxes();
    setTimeout(fixCheckboxes, 500);
    setInterval(fixCheckboxes, 2000);
    </script>
    """, height=0)

# ------------------ PORTFOLIO MANAGEMENT FUNCTIONS ------------------
def create_portfolio_tab():
    st.header("Portfolio Management")
    
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {
            'holdings': {},
            'cash': 10000.0,
            'transactions': [],
            'performance': [],
            'allocation_rules': {}
        }
    
    port_tab1, port_tab2, port_tab3 = st.tabs(["Summary", "Allocation Rules", "Transaction History"])
    
    with port_tab1:
        display_portfolio_summary()
    
    with port_tab2:
        configure_allocation_rules()
    
    with port_tab3:
        display_transaction_history()

def display_portfolio_summary():
    st.subheader("Portfolio Summary")
    
    portfolio = st.session_state.portfolio
    holdings = portfolio['holdings']
    cash = portfolio['cash']
    total_value = cash
    holdings_data = []
    
    for ticker, data in holdings.items():
        try:
            current_price = get_realtime_price(ticker)
            shares = data['shares']
            avg_price = data['avg_price']
            position_value = shares * current_price
            total_value += position_value
            profit_loss = position_value - (shares * avg_price)
            profit_loss_pct = (profit_loss / (shares * avg_price)) * 100 if shares > 0 else 0
            
            holdings_data.append({
                'Ticker': ticker,
                'Shares': shares,
                'Avg Price': f"${avg_price:.2f}",
                'Current Price': f"${current_price:.2f}",
                'Value': f"${position_value:.2f}",
                'P/L': f"${profit_loss:.2f} ({profit_loss_pct:.2f}%)"
            })
        except Exception as e:
            st.warning(f"Could not update {ticker}: {str(e)}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Portfolio Value", f"${total_value:.2f}")
    col2.metric("Cash Available", f"${cash:.2f}")
    col3.metric("Invested Amount", f"${(total_value - cash):.2f}")
    
    if holdings_data:
        st.dataframe(pd.DataFrame(holdings_data))
    else:
        st.info("No holdings in portfolio. Use the Allocation Rules tab to set up automatic investments.")
    
    st.subheader("Manual Transactions")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Buy Assets")
        buy_ticker = st.selectbox("Select Asset to Buy", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"], key="buy_ticker")
        buy_amount = st.number_input("Investment Amount ($)", min_value=0.0, max_value=cash, step=100.0, key="buy_amount")
        if st.button("Execute Buy", key="buy_btn"):
            if 0 < buy_amount <= cash:
                execute_transaction(buy_ticker, buy_amount, transaction_type="BUY")
                st.success(f"Successfully purchased {buy_ticker}")
                st.experimental_rerun()
            else:
                st.error("Invalid amount. Please check your cash balance.")
    
    with col2:
        st.markdown("#### Sell Assets")
        sell_options = list(holdings.keys())
        if sell_options:
            sell_ticker = st.selectbox("Select Asset to Sell", sell_options, key="sell_ticker")
            max_shares = holdings[sell_ticker]['shares'] if sell_ticker in holdings else 0
            sell_shares = st.number_input("Shares to Sell", min_value=0.0, max_value=max_shares, step=1.0, key="sell_shares")
            if st.button("Execute Sell", key="sell_btn"):
                if 0 < sell_shares <= max_shares:
                    current_price = get_realtime_price(sell_ticker)
                    sell_amount = sell_shares * current_price
                    execute_transaction(sell_ticker, sell_amount, transaction_type="SELL", shares=sell_shares)
                    st.success(f"Successfully sold {sell_shares} shares of {sell_ticker}")
                    st.experimental_rerun()
                else:
                    st.error("Invalid share amount.")
        else:
            st.info("No assets to sell.")

def configure_allocation_rules():
    st.subheader("Allocation Rules")
    st.markdown("""
    Set up rules for automatic portfolio adjustments based on market regimes and risk metrics.
    The system will use Markov Chain predictions to make optimal allocation decisions.
    """)
    
    all_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
    current_rules = st.session_state.portfolio.get('allocation_rules', {})
    
    st.markdown("### Create New Rule")
    rule_col1, rule_col2 = st.columns(2)
    with rule_col1:
        rule_ticker = st.selectbox("Asset", all_tickers, key="rule_ticker")
        rule_condition = st.selectbox("Condition", [
            "Market Regime Changes To",
            "Volatility Above",
            "Volatility Below",
            "Expected Return Above",
            "Expected Return Below",
            "VaR Below",
            "Price Below MA(20)",
            "Price Above MA(20)"
        ], key="rule_condition")
        if "Market Regime Changes To" in rule_condition:
            rule_value = st.selectbox("Regime", ["Bullish", "Bearish", "Sideways", "Volatile", "Stable"], key="rule_value")
        elif "Volatility" in rule_condition or "Expected Return" in rule_condition or "VaR" in rule_condition:
            rule_value = st.slider("Threshold Value (%)", 0.0, 100.0, 20.0, key="rule_slider")
        else:
            rule_value = True
    with rule_col2:
        rule_action = st.selectbox("Action", ["Buy", "Sell", "Rebalance"], key="rule_action")
        if rule_action == "Buy":
            action_amount = st.number_input("Amount to Buy ($)", min_value=100.0, step=100.0, key="action_amount")
            action_param = action_amount
        elif rule_action == "Sell":
            action_percent = st.slider("Percentage of Position to Sell", 0, 100, 50, key="action_percent")
            action_param = action_percent
        else:
            action_target = st.slider("Target Allocation (%)", 0, 100, 20, key="action_target")
            action_param = action_target
    
    if st.button("Add Rule", key="add_rule"):
        rule_id = f"{rule_ticker}_{rule_condition}_{str(rule_value)}".replace(" ", "_")
        current_rules[rule_id] = {
            "ticker": rule_ticker,
            "condition": rule_condition,
            "value": rule_value,
            "action": rule_action,
            "param": action_param,
            "active": True,
            "last_triggered": None
        }
        st.session_state.portfolio['allocation_rules'] = current_rules
        st.success("Rule added successfully!")
    
    st.markdown("### Current Rules")
    if current_rules:
        rules_data = []
        for rule_id, rule in current_rules.items():
            rules_data.append({
                "Ticker": rule["ticker"],
                "Condition": f"{rule['condition']} {rule['value']}",
                "Action": f"{rule['action']} {rule['param']}",
                "Status": "Active" if rule["active"] else "Inactive",
                "Last Triggered": rule["last_triggered"] or "Never"
            })
        st.dataframe(pd.DataFrame(rules_data))
        
        selected_rule = st.selectbox("Select Rule to Manage", list(current_rules.keys()), format_func=lambda x: f"{current_rules[x]['ticker']} - {current_rules[x]['condition']}", key="selected_rule")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Toggle Active Status", key="toggle_rule"):
                current_rules[selected_rule]["active"] = not current_rules[selected_rule]["active"]
                st.session_state.portfolio['allocation_rules'] = current_rules
                st.success(f"Rule is now {'active' if current_rules[selected_rule]['active'] else 'inactive'}")
        with col2:
            if st.button("Delete Rule", key="delete_rule"):
                del current_rules[selected_rule]
                st.session_state.portfolio['allocation_rules'] = current_rules
                st.success("Rule deleted")
        with col3:
            if st.button("Test Rule Now", key="test_rule"):
                result = evaluate_rule(current_rules[selected_rule])
                if result:
                    st.success("Rule conditions are currently met!")
                else:
                    st.info("Rule conditions are not currently met.")
    else:
        st.info("No allocation rules defined yet.")

def display_transaction_history():
    st.subheader("Transaction History")
    transactions = st.session_state.portfolio.get('transactions', [])
    if transactions:
        df = pd.DataFrame(transactions)
        st.dataframe(df)
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Transaction History",
            data=csv,
            file_name="portfolio_transactions.csv",
            mime="text/csv"
        )
    else:
        st.info("No transactions recorded yet.")

def execute_transaction(ticker, amount, transaction_type="BUY", shares=None):
    portfolio = st.session_state.portfolio
    holdings = portfolio['holdings']
    cash = portfolio['cash']
    timestamp = datetime.now()
    current_price = get_realtime_price(ticker)
    
    if transaction_type == "BUY":
        shares_to_buy = amount / current_price
        portfolio['cash'] -= amount
        if ticker in holdings:
            total_shares = holdings[ticker]['shares'] + shares_to_buy
            total_cost = (holdings[ticker]['shares'] * holdings[ticker]['avg_price']) + amount
            holdings[ticker]['shares'] = total_shares
            holdings[ticker]['avg_price'] = total_cost / total_shares
            holdings[ticker]['last_updated'] = timestamp
        else:
            holdings[ticker] = {
                'shares': shares_to_buy,
                'avg_price': current_price,
                'last_updated': timestamp
            }
        portfolio['transactions'].append({
            'Date': timestamp,
            'Type': 'BUY',
            'Ticker': ticker,
            'Shares': shares_to_buy,
            'Price': current_price,
            'Amount': amount,
            'Trigger': 'Manual'
        })
    
    elif transaction_type == "SELL":
        if ticker in holdings and shares <= holdings[ticker]['shares']:
            portfolio['cash'] += amount
            holdings[ticker]['shares'] -= shares
            if holdings[ticker]['shares'] <= 0:
                del holdings[ticker]
            portfolio['transactions'].append({
                'Date': timestamp,
                'Type': 'SELL',
                'Ticker': ticker,
                'Shares': shares,
                'Price': current_price,
                'Amount': amount,
                'Trigger': 'Manual'
            })
    
    st.session_state.portfolio = portfolio

def evaluate_rule(rule):
    ticker = rule["ticker"]
    condition = rule["condition"]
    value = rule["value"]
    
    try:
        current_price = get_realtime_price(ticker)
        if "Market Regime Changes To" in condition:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            raw_data = get_stock_data([ticker], start_date, end_date)
            processed_data = assign_states(raw_data, 5)
            current_regime = processed_data[ticker]['State'].iloc[-1]
            regime_names = {0: "Bearish", 1: "Sideways", 2: "Bullish", 3: "Volatile", 4: "Stable"}
            current_regime_name = regime_names.get(current_regime, "Unknown")
            return current_regime_name == value
        
        elif "Volatility Above" in condition:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            raw_data = get_stock_data([ticker], start_date, end_date)
            volatility = raw_data[ticker]['Close'].pct_change().std() * 100 * (252 ** 0.5)
            return volatility > value
        
        elif "Volatility Below" in condition:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            raw_data = get_stock_data([ticker], start_date, end_date)
            volatility = raw_data[ticker]['Close'].pct_change().std() * 100 * (252 ** 0.5)
            return volatility < value
        
        elif "Expected Return Above" in condition or "Expected Return Below" in condition:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            raw_data = get_stock_data([ticker], start_date, end_date)
            processed_data = assign_states(raw_data, 5)
            transition_matrix, _ = get_transition_matrix(processed_data[ticker]['State'].values, 5)
            simulation_results = []
            for i in range(100):
                sim_result = simulate_price_path(
                    processed_data[ticker]['Close'].iloc[-1],
                    transition_matrix,
                    [processed_data[ticker][processed_data[ticker]['State'] == s]['Daily_Return'].values for s in range(5)],
                    30
                )
                if isinstance(sim_result, tuple):
                    sim_result = sim_result[0]
                simulation_results.append(np.array(sim_result))
            simulations = np.stack(simulation_results, axis=0)
            expected_return = ((simulations[:, -1] / simulations[:, 0]) - 1).mean() * 100
            if "Above" in condition:
                return expected_return > value
            else:
                return expected_return < value
        
        elif "VaR Below" in condition:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            raw_data = get_stock_data([ticker], start_date, end_date)
            processed_data = assign_states(raw_data, 5)
            transition_matrix, _ = get_transition_matrix(processed_data[ticker]['State'].values, 5)
            simulation_results = []
            for i in range(100):
                sim_result = simulate_price_path(
                    processed_data[ticker]['Close'].iloc[-1],
                    transition_matrix,
                    [processed_data[ticker][processed_data[ticker]['State'] == s]['Daily_Return'].values for s in range(5)],
                    30
                )
                if isinstance(sim_result, tuple):
                    sim_result = sim_result[0]
                simulation_results.append(np.array(sim_result))
            simulations = np.stack(simulation_results, axis=0)
            returns = (simulations[:, -1] / simulations[:, 0]) - 1
            var_95 = np.percentile(returns, 5) * -100
            return var_95 < value
        
        elif "Price Below MA(20)" in condition or "Price Above MA(20)" in condition:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            raw_data = get_stock_data([ticker], start_date, end_date)
            ma_20 = raw_data[ticker]['Close'].rolling(window=20).mean().iloc[-1]
            if "Below" in condition:
                return current_price < ma_20
            else:
                return current_price > ma_20
        return False
    except Exception as e:
        logging.error(f"Error evaluating rule: {str(e)}")
        return False

def check_portfolio_rules():
    portfolio = st.session_state.portfolio
    rules = portfolio.get('allocation_rules', {})
    
    for rule_id, rule in rules.items():
        if rule["active"]:
            if evaluate_rule(rule):
                execute_rule_action(rule)
                rule["last_triggered"] = datetime.now()
                portfolio['allocation_rules'][rule_id] = rule

def execute_rule_action(rule):
    ticker = rule["ticker"]
    action = rule["action"]
    param = rule["param"]
    portfolio = st.session_state.portfolio
    
    try:
        if action == "Buy":
            if param <= portfolio['cash']:
                execute_transaction(ticker, param, transaction_type="BUY")
                logging.info(f"Rule triggered: Bought ${param} of {ticker}")
            else:
                logging.warning(f"Rule trigger failed: Not enough cash to buy ${param} of {ticker}")
        elif action == "Sell":
            if ticker in portfolio['holdings']:
                current_price = get_realtime_price(ticker)
                shares = portfolio['holdings'][ticker]['shares']
                shares_to_sell = shares * (param / 100)
                sell_amount = shares_to_sell * current_price
                execute_transaction(ticker, sell_amount, transaction_type="SELL", shares=shares_to_sell)
                logging.info(f"Rule triggered: Sold {shares_to_sell} shares ({param}%) of {ticker}")
        elif action == "Rebalance":
            total_value = portfolio['cash']
            for t, data in portfolio['holdings'].items():
                current_price = get_realtime_price(t)
                total_value += data['shares'] * current_price
            target_value = total_value * (param / 100)
            current_value = 0
            if ticker in portfolio['holdings']:
                current_price = get_realtime_price(ticker)
                current_value = portfolio['holdings'][ticker]['shares'] * current_price
            difference = target_value - current_value
            if difference > 0:
                if difference <= portfolio['cash']:
                    execute_transaction(ticker, difference, transaction_type="BUY")
                    logging.info(f"Rule triggered: Rebalanced {ticker} by buying ${difference:.2f}")
                else:
                    execute_transaction(ticker, portfolio['cash'], transaction_type="BUY")
                    logging.info(f"Rule triggered: Rebalanced {ticker} by buying all available cash ${portfolio['cash']:.2f}")
            elif difference < 0:
                shares = portfolio['holdings'][ticker]['shares']
                current_price = get_realtime_price(ticker)
                shares_to_sell = abs(difference) / current_price
                if shares_to_sell <= shares:
                    execute_transaction(ticker, abs(difference), transaction_type="SELL", shares=shares_to_sell)
                    logging.info(f"Rule triggered: Rebalanced {ticker} by selling {shares_to_sell} shares")
    except Exception as e:
        logging.error(f"Error executing rule action: {str(e)}")

# ------------------ MAIN APP ------------------
def main():
    st.set_page_config(
        page_title="QuantPro Analytics",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if 'theme' not in st.session_state:
        st.session_state.theme = DARK_THEME
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = load_history()
    if 'selected_question' not in st.session_state:
        st.session_state.selected_question = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    if 'voice_enabled' not in st.session_state:
        st.session_state.voice_enabled = False
    if 'auto_speak' not in st.session_state:
        st.session_state.auto_speak = False

    with st.sidebar:
        st.header("Controls")
        if st.button("🌓 Toggle Theme"):
            st.session_state.theme = LIGHT_THEME if st.session_state.theme == DARK_THEME else DARK_THEME
        apply_theme(st.session_state.theme)
        
        st.markdown("#### Report Settings")
        report_enabled = st.checkbox("Enable Report Download", value=True)
        
        st.markdown("#### Font Size")
        font_size = st.slider("Adjust Chat Font Size", 12, 24, 16)
        
        st.markdown("#### Simulation Settings")
        sim_duration = st.slider("Simulation Duration (Days)", 10, 60, 30)
        
        st.divider()
        tickers = st.multiselect("Select Assets", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"],
                                 default=["AAPL", "MSFT"])
        start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
        end_date = st.date_input("End Date", pd.to_datetime(datetime.today().strftime('%Y-%m-%d')))
        n_states = st.slider("Market Regimes", 3, 7, 5)
        n_simulations = st.slider("Simulation Paths", 100, 5000, 1000)
        
        st.divider()
        st.markdown("#### Real-time Data")
        if tickers:
            for t in tickers:
                price = get_realtime_price(t)
                st.metric(f"{t} Price", f"${price:.2f}")
        
        if report_enabled:
            st.markdown("#### Export Report")
            if st.button("Download Report"):
                context_str = f"Analysis of {', '.join(tickers)} with {n_states} regimes"
                report = generate_report(st.session_state.chat_history, context_str, st.session_state.metrics)
                st.download_button(
                    label="📄 Download Report",
                    data=report,
                    file_name="quantpro_report.md",
                    mime="text/markdown"
                )

    with st.container():
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.image("https://via.placeholder.com/200x60.png?text=QuantPro", width=200)
        with col2:
            st.title("Financial Analytics Suite")
            st.markdown("*Market analysis powered by Markov Chain models*")
        with col3:
            if st.button("🔄 Refresh Data"):
                st.experimental_rerun()

    analysis_context = f"""
    **Portfolio Context**
    - Assets: {', '.join(tickers) if tickers else 'None selected'}
    - Analysis Window: {start_date} to {end_date}
    - Market Regimes: {n_states}
    - Simulation Duration: {sim_duration} days
    - Monte Carlo Paths: {n_simulations}
    - As of: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    """

    # Create main dashboard tabs including a new Portfolio Management tab.
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Market Overview", "📈 Simulation", "Portfolio Management", "💬 AI Analyst", "🔧 Miscellaneous"])

    with tab1:
        if tickers:
            try:
                with st.spinner("Loading market data..."):
                    raw_data = get_stock_data(tickers, start_date, end_date)
                    processed_data = assign_states(raw_data, n_states)
                    st.subheader("Historical Market Metrics")
                    cols = st.columns(len(tickers))
                    for idx, ticker in enumerate(tickers):
                        with cols[idx]:
                            latest = raw_data[ticker]['Close'].iloc[-1]
                            volatility = processed_data[ticker]['Daily_Return'].std()
                            st.markdown(f"""
                            <div class='metric-card' style='padding:1rem; margin-bottom:1rem; border-radius:8px;'>
                                <h3>{ticker}</h3>
                                <p>📈 Price: ${latest:.2f}</p>
                                <p>📉 Volatility: {volatility:.2f}%</p>
                                <p>📅 Regimes: {n_states}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    chart_data = pd.DataFrame()
                    for t in tickers:
                        norm_price = processed_data[t]['Close'] / processed_data[t]['Close'].iloc[0]
                        temp_df = pd.DataFrame({
                            'Date': processed_data[t].index,
                            'Normalized Price': norm_price,
                            'Ticker': t
                        })
                        chart_data = pd.concat([chart_data, temp_df])
                    chart = alt.Chart(chart_data).mark_line().encode(
                        x='Date:T',
                        y='Normalized Price:Q',
                        color='Ticker:N',
                        tooltip=['Date', 'Normalized Price', 'Ticker']
                    ).properties(
                        title="Normalized Price Trajectory Comparison"
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"Data loading failed: {str(e)}")
                logging.error(f"Data loading failed: {str(e)}")

    with tab2:
        if tickers:
            selected_ticker = st.selectbox("Select Asset for Simulation", tickers, key="sim_ticker")
            if st.button("Run Market Simulations", type="primary", key="run_sim"):
                try:
                    raw_data = get_stock_data(tickers, start_date, end_date)
                    processed_data = assign_states(raw_data, n_states)
                    transition_matrix, _ = get_transition_matrix(processed_data[selected_ticker]['State'].values, n_states)
                    simulation_results = []
                    for i in range(n_simulations):
                        sim_result = simulate_price_path(
                            processed_data[selected_ticker]['Close'].iloc[-1],
                            transition_matrix,
                            [processed_data[selected_ticker][processed_data[selected_ticker]['State'] == s]['Daily_Return'].values for s in range(n_states)],
                            sim_duration
                        )
                        if isinstance(sim_result, tuple):
                            sim_result = sim_result[0]
                        simulation_results.append(np.array(sim_result))
                    simulations = np.stack(simulation_results, axis=0)
                    simulation_data = pd.DataFrame({
                        'Trading Days': np.tile(np.arange(sim_duration + 1), n_simulations),
                        'Price Projection ($)': simulations.flatten(),
                        'Simulation': np.repeat(np.arange(n_simulations), sim_duration + 1)
                    })
                    median_path = np.percentile(simulations, 50, axis=0)
                    median_data = pd.DataFrame({
                        'Trading Days': np.arange(sim_duration + 1),
                        'Price Projection ($)': median_path
                    })
                    chart = alt.Chart(simulation_data).mark_line(opacity=0.03, color="#00ff88").encode(
                        x='Trading Days:Q',
                        y='Price Projection ($):Q',
                        detail='Simulation:N'
                    ).properties(
                        title=f"{selected_ticker} {sim_duration}-Day Simulation"
                    )
                    median_line = alt.Chart(median_data).mark_line(color="white", strokeWidth=2.5).encode(
                        x='Trading Days:Q',
                        y='Price Projection ($):Q'
                    )
                    final_chart = (chart + median_line).interactive()
                    st.altair_chart(final_chart, use_container_width=True)
                    metrics, summary_stats  = calculate_performance_metrics(
                        simulations=simulations,
                        df=processed_data[selected_ticker],
                        current_price=processed_data[selected_ticker]['Close'].iloc[-1],
                        forecast_days=sim_duration
                    )
                    st.session_state.metrics = metrics
                    if summary_stats:
                        summary_df = pd.DataFrame(summary_stats, columns=["Metric", "Value"])
                        st.markdown("### Simulation Performance Summary")
                        st.table(summary_df)
                    cols = st.columns(4)
                    risk_metrics = [
                        ("Value at Risk (95%)", metrics.get('Simulated VaR (95%)', 0)),
                        ("Expected Return", metrics.get('Simulated Mean Return', 0)),
                        ("Volatility", metrics.get('Simulated Volatility', 0)),
                        ("Best Case (5%)", metrics.get('Simulated Best Case (5%)', 0))
                    ]
                    for idx, (label, value) in enumerate(risk_metrics):
                        with cols[idx]:
                            st.metric(label, f"{value:.2f}%")
                except Exception as e:
                    st.error(f"Risk analysis failed: {str(e)}")
                    logging.error(f"Risk analysis failed: {str(e)}")

    with tab3:
        create_portfolio_tab()

    with tab4:
        st.header("AI Analyst")
        init_ai_analyst_tab()
        st.markdown('<div id="ai-analyst-tab" class="analyst-tab-container"></div>', unsafe_allow_html=True)
        cols = st.columns(3)
        SAMPLE_QUESTIONS = [
            "Analyze current volatility patterns",
            "Compare risk profiles of selected assets",
            "Predict next week's price movement for {ticker}",
            "Explain the current market regime",
            "What are the key risk factors?",
            "Suggest portfolio adjustments based on analysis"
        ]
        for i, q in enumerate(SAMPLE_QUESTIONS[:3]):
            with cols[i]:
                formatted_q = q.format(ticker=tickers[0]) if "{ticker}" in q and tickers else q
                st.markdown(f'<div class="sample-question">{formatted_q}</div>', unsafe_allow_html=True)
        voice_enabled, auto_speak = custom_voice_controls()
        st.markdown("### Conversation History")
        chat_style = f"""
        <div style='background-color: var(--surface);
                    border-left: 3px solid var(--primary);
                    padding: 1rem;
                    margin: 1rem 0;
                    border-radius: 8px;
                    color: var(--text);
                    line-height: 1.6;
                    font-size: {font_size}px;'>
            {{content}}
        </div>
        """
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(chat_style.format(content=msg["content"]), unsafe_allow_html=True)
                if voice_enabled and msg["role"] == "assistant":
                    speak_button(msg["content"])
        st.markdown("""
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(function() {
                document.querySelectorAll('[data-testid="stCheckbox"] *').forEach(el => {
                    el.style.setProperty('color', 'white', 'important');
                    el.style.setProperty('opacity', '1', 'important');
                    el.style.setProperty('visibility', 'visible', 'important');
                });
            }, 1000);
        });
        </script>
        """, unsafe_allow_html=True)

    with tab5:
        st.header("Settings & Tools")
        settings_col, tools_col = st.columns(2)
        with settings_col:
            st.subheader("Application Settings")
            st.markdown("#### Data Configuration")
            cache_option = st.radio("Data Caching", ["Enable (Faster)", "Disable (Real-time)"], index=0)
            precision = st.slider("Decimal Precision", 1, 4, 2)
            st.markdown("#### Notifications")
            enable_alerts = st.checkbox("Enable Price Alerts", value=False)
            if enable_alerts:
                alert_ticker = st.selectbox("Select Asset", tickers if tickers else ["AAPL"])
                alert_price = st.number_input("Alert Price ($)", value=get_realtime_price(alert_ticker) if tickers else 100.0)
                alert_condition = st.selectbox("Condition", ["Above", "Below"])
                st.button("Set Alert")
            st.markdown("#### Export Options")
            export_format = st.selectbox("Default Export Format", ["Markdown", "CSV", "JSON", "PDF"])
            include_charts = st.checkbox("Include Charts in Reports", value=True)
        with tools_col:
            st.subheader("Financial Tools")
            st.markdown("#### Portfolio Tracker")
            portfolio_file = st.file_uploader("Upload Portfolio (CSV)", type="csv")
            if portfolio_file:
                try:
                    portfolio_data = pd.read_csv(portfolio_file)
                    st.dataframe(portfolio_data)
                    if st.button("Analyze Portfolio"):
                        st.info("Portfolio analysis would run here")
                except Exception as e:
                    st.error(f"Error reading portfolio file: {e}")
            st.markdown("#### Financial Calculators")
            calc_type = st.selectbox("Calculator Type", [
                "Compound Interest", 
                "Stock Position Sizing", 
                "Risk/Reward Ratio", 
                "Kelly Criterion"
            ])
            if calc_type == "Compound Interest":
                principal = st.number_input("Initial Investment ($)", value=1000.0, step=100.0)
                rate = st.number_input("Annual Interest Rate (%)", value=7.0, step=0.1)
                years = st.number_input("Time Period (Years)", value=10, step=1)
                compound_freq = st.selectbox("Compound Frequency", ["Annually", "Semi-annually", "Quarterly", "Monthly", "Daily"])
                freq_map = {"Annually": 1, "Semi-annually": 2, "Quarterly": 4, "Monthly": 12, "Daily": 365}
                n = freq_map[compound_freq]
                if st.button("Calculate"):
                    result = principal * (1 + (rate / 100) / n) ** (n * years)
                    st.success(f"Future Value: ${result:.2f}")
            elif calc_type == "Stock Position Sizing":
                account_size = st.number_input("Account Size ($)", value=10000.0, step=1000.0)
                risk_percent = st.number_input("Risk Per Trade (%)", value=1.0, step=0.1, max_value=100.0)
                entry_price = st.number_input("Entry Price ($)", value=100.0, step=1.0)
                stop_price = st.number_input("Stop Loss Price ($)", value=95.0, step=1.0)
                if st.button("Calculate Position Size"):
                    risk_amount = account_size * (risk_percent / 100)
                    risk_per_share = abs(entry_price - stop_price)
                    shares = int(risk_amount / risk_per_share)
                    position_value = shares * entry_price
                    st.success(f"Position Size: {shares} shares (${position_value:.2f})")
        with st.expander("Documentation & Help"):
            st.markdown("""
            ### Quick Help Guide
            **Market Overview Tab**
            - Shows historical data and market metrics for selected assets
            - Displays normalized price comparison charts
            **Simulation Tab**
            - Run Monte Carlo simulations based on Markov Chain models
            - View risk metrics and potential price paths
            **AI Analyst Tab**
            - Ask questions about your assets and market conditions
            - Get AI-powered analysis and insights
            **Keyboard Shortcuts**
            - Ctrl+R: Refresh data
            - Ctrl+S: Run simulation
            - Ctrl+D: Download report
            For additional help, contact support@quantpro.example.com
            """)

    prompt = st.chat_input("Ask AI analyst...")
    if not prompt and st.session_state.selected_question:
        prompt = st.session_state.selected_question
        st.session_state.selected_question = None

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        try:
            with st.spinner("🧠 Analyzing with Gemini..."):
                response = stock_analysis_agent(prompt, analysis_context)
                with st.chat_message("assistant"):
                    st.markdown(response, unsafe_allow_html=True)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                save_history(st.session_state.chat_history)
                if st.session_state.voice_enabled:
                    if st.session_state.auto_speak:
                        clean_response = response.replace('`', '\\`').replace('"', '\\"').replace('\n', ' ')
                        speak_js = f"""
                        <script>
                            (function() {{
                                var utterance = new SpeechSynthesisUtterance(`{clean_response}`);
                                utterance.rate = 1.0;
                                window.speechSynthesis.cancel();
                                window.speechSynthesis.speak(utterance);
                            }})();
                        </script>
                        """
                        st.components.v1.html(speak_js, height=0)
                    else:
                        speak_button(response)
        except Exception as e:
            st.error(f"AI Analysis Failed: {str(e)}")
            st.session_state.chat_history.append({"role": "assistant", "content": f"Analysis Error: {str(e)}"})

if __name__ == "__main__":
    main()
