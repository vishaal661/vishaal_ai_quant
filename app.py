import streamlit as st
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib

# 1. PAGE CONFIG
st.set_page_config(page_title="AI Secure Quant v5.0", layout="wide")

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

ADMIN_HASH = hash_password("vishaal_admin")
USER_HASH = hash_password("user123")

# 2. LOGIN SYSTEM
st.sidebar.title("🔐 Secure Access")
user_role = st.sidebar.selectbox("Login As", ["User", "Admin"])
password = st.sidebar.text_input("Enter Password", type="password")

if not password:
    st.info("Please enter password to start.")
    st.stop()

if hash_password(password) not in [ADMIN_HASH, USER_HASH]:
    st.error("❌ Incorrect Password")
    st.stop()

# 3. AI PROCESSING FUNCTION
def process_stock(ticker, days):
    try:
        # Ticker check to avoid GOOGL empty data error
        raw_data = yf.download(ticker, period=f'{days}d', interval='1d', progress=False)
        
        if raw_data.empty or len(raw_data) < 35: # Minimum 35 days for stable MACD/MA
            return None
        
        df = raw_data.copy().astype(float)
        
        # --- MACD Calculation (Fixed) ---
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Features
        df['Day'] = np.arange(len(df))
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # RSI Logic
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df_clean = df.dropna().copy()
        if df_clean.empty:
            return None
            
        X = df_clean[['Day', 'MA50', 'RSI']]
        y = df_clean['Close']
        model = LinearRegression().fit(X, y)
        
        last_val = df_clean.iloc[-1]
        prediction = model.predict([[float(len(df)), float(last_val['MA50']), float(last_val['RSI'])]])
        
        # Dictionary structure to prevent KeyError
        return {
            'full_df': df, 
            'pred': float(prediction.flatten()[0]), 
            'curr': float(last_val['Close'])
        }
    except Exception:
        return None

# 4. MAIN APP DISPLAY
st.title("⚔️ AI Stock Battle")
ticker1 = st.sidebar.text_input("Stock 1", value="AAPL")
ticker2 = st.sidebar.text_input("Stock 2", value="TSLA")
days = st.sidebar.slider("Days", 100, 500, 250)

if st.sidebar.button("Run AI Analysis"):
    col_a, col_b = st.columns(2)
    
    for i, t in enumerate([ticker1, ticker2]):
        result = process_stock(t, days)
        current_col = col_a if i == 0 else col_b
        
        if result:
            # Result extraction
            stock_df = result['full_df']
            pred_val = result['pred']
            curr_val = result['curr']
            diff = ((pred_val - curr_val) / curr_val) * 100
            
            with current_col:
                st.header(f"📊 {t}")
                st.metric("Predicted Price", f"${pred_val:.2f}", f"{diff:.2f}%")
                
                # Main Price Chart
                fig = go.Figure(data=[go.Candlestick(x=stock_df.index, open=stock_df['Open'], high=stock_df['High'], low=stock_df['Low'], close=stock_df['Close'])])
                fig.update_layout(template="plotly_dark", height=350, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                # --- MACD VISUALIZATION (Fixed Safety Check) ---
                # This prevents the KeyError at Line 117
                if 'MACD' in stock_df.columns and 'Signal_Line' in stock_df.columns:
                    st.write("---")
                    st.subheader("🛠️ MACD Trend Analysis")
                    # Explicit column check
                    macd_data = stock_df[['MACD', 'Signal_Line']].dropna()
                    if not macd_data.empty:
                        st.line_chart(macd_data)
                        st.bar_chart(stock_df['MACD'] - stock_df['Signal_Line'])
                else:
                    st.warning("Insufficient data for MACD indicators.")
        else:
            current_col.error(f"⚠️ Analysis failed for {t}.")
