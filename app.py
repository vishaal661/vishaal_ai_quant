import streamlit as st
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib
import pandas as pd

# 1. PAGE CONFIG
st.set_page_config(page_title="AI Quant V8.0", layout="wide")

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

ADMIN_HASH = hash_password("vishaal_admin")
USER_HASH = hash_password("user123")

# 2. LOGIN
st.sidebar.title("🔐 Login")
pwd = st.sidebar.text_input("Password", type="password")

if not pwd or hash_password(pwd) not in [ADMIN_HASH, USER_HASH]:
    st.info("Please enter password to start.")
    st.stop()

# 3. DATA ENGINE (FIXED FOR MULTI-INDEX)
def get_data(ticker, days):
    try:
        # Download data
        df = yf.download(ticker, period=f'{days}d', interval='1d', progress=False)
        if df.empty: return None
        
        # --- CRITICAL FIX: Flattening Columns ---
        # Yfinance thara Multi-index-ah single layer-ah mathuroam
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Column names-ah clean pandrom
        df.columns = [str(c).strip() for c in df.columns]
        
        # Ensure we have a clean DataFrame with Date as Index
        data = pd.DataFrame(index=df.index)
        data['Close'] = df['Close']
        data['Open'] = df['Open']
        data['High'] = df['High']
        data['Low'] = df['Low']
        
        # --- TECHNICAL INDICATORS ---
        # Manual calculation to avoid KeyError
        ema12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = ema12 - ema26
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['Day_Idx'] = np.arange(len(data))
        
        return data.dropna()
    except Exception as e:
        return None

# 4. DASHBOARD UI
st.title("⚔️ AI Stock Battle - Final Fix")
t1 = st.sidebar.text_input("Stock 1", "AAPL")
t2 = st.sidebar.text_input("Stock 2", "TSLA")

if st.sidebar.button("Run Analysis"):
    cols = st.columns(2)
    for i, t in enumerate([t1, t2]):
        df = get_data(t, 300)
        with cols[i]:
            if df is not None:
                # AI Model
                X = df[['Day_Idx', 'MA50']].values
                y = df['Close'].values
                model = LinearRegression().fit(X, y)
                pred = model.predict([[len(df), df['MA50'].iloc[-1]]])[0]
                curr = df['Close'].iloc[-1]
                
                st.header(f"📊 {t}")
                st.metric("AI Prediction", f"${pred:.2f}", f"{((pred-curr)/curr)*100:.2f}%")

                # PLOTLY SUBPLOTS (Safe for indexing)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color='cyan')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color='yellow')), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name="Signal", line=dict(color='magenta')), row=2, col=1)
                
                fig.update_layout(height=500, template="plotly_dark", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Error loading {t}")
