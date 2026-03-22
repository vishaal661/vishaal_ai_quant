import streamlit as st
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib
import pandas as pd

# 1. PAGE CONFIG
st.set_page_config(page_title="AI Quant Pro V9.0", layout="wide")

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

ADMIN_HASH = hash_password("vishaal_admin")
USER_HASH = hash_password("user123")

# 2. LOGIN
st.sidebar.title("🔐 Secure Login")
pwd = st.sidebar.text_input("Password", type="password")

if not pwd or hash_password(pwd) not in [ADMIN_HASH, USER_HASH]:
    st.info("Unlock the Pro Dashboard with your password.")
    st.stop()

# 3. ADVANCED DATA ENGINE
def get_pro_data(ticker, days):
    try:
        df = yf.download(ticker, period=f'{days}d', interval='1d', progress=False)
        if df.empty: return None
        
        # Multi-index Fix
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        data = pd.DataFrame(index=df.index)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            data[col] = df[col]
            
        # Indicators
        ema12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = ema12 - ema26
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['Day_Idx'] = np.arange(len(data))
        
        return data.dropna()
    except Exception:
        return None

# 4. DASHBOARD
st.title("⚔️ AI Stock Battle - Pro Edition")
t1 = st.sidebar.text_input("Stock 1", "AAPL")
t2 = st.sidebar.text_input("Stock 2", "TSLA")

if st.sidebar.button("Execute Pro Analysis"):
    cols = st.columns(2)
    for i, t in enumerate([t1, t2]):
        df = get_pro_data(t, 300)
        with cols[i]:
            if df is not None:
                # AI Prediction
                X = df[['Day_Idx', 'MA50']].values
                y = df['Close'].values
                model = LinearRegression().fit(X, y)
                pred = model.predict([[len(df), df['MA50'].iloc[-1]]])[0]
                curr = df['Close'].iloc[-1]
                diff = ((pred - curr) / curr) * 100
                
                st.header(f"📉 {t}")
                st.metric("AI Target Price", f"${pred:.2f}", f"{diff:.2f}%")

                # --- BUY/SELL SIGNALS ---
                if diff > 2.0 and df['MACD'].iloc[-1] > df['Signal'].iloc[-1]:
                    st.success(f"🚀 SIGNAL: STRONG BUY ({t})")
                elif diff < -2.0:
                    st.error(f"⚠️ SIGNAL: SELL / BEARISH ({t})")
                else:
                    st.warning(f"⚖️ SIGNAL: NEUTRAL / HOLD")

                # --- PRO CANDLESTICK CHART ---
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.05, row_heights=[0.7, 0.3])

                # Candlesticks
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                             low=df['Low'], close=df['Close'], name="Market"), row=1, col=1)
                
                # MACD
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color='lime')), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name="Signal", line=dict(color='red')), row=2, col=1)

                fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Data missing for {t}")
