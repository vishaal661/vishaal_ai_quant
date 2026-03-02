import streamlit as st
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib

# 1. PAGE CONFIG & SECURITY SETUP
st.set_page_config(page_title="AI Secure Quant v5.0", layout="wide")

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# Passwords (Neenga mathikalaam)
ADMIN_HASH = hash_password("vishaal_admin")
USER_HASH = hash_password("user123")

# 2. LOGIN SYSTEM
st.sidebar.title("🔐 Secure Access")
user_role = st.sidebar.selectbox("Login As", ["User", "Admin"])
password = st.sidebar.text_input("Enter Password", type="password")

if not password:
    st.info("Please enter password in the sidebar to start.")
    st.stop()

if hash_password(password) not in [ADMIN_HASH, USER_HASH]:
    st.error("❌ Access Denied: Incorrect Password")
    st.stop()

# ---------------------------------------------------------
# IF LOGIN SUCCESSFUL, APP STARTS HERE
# ---------------------------------------------------------

st.sidebar.success(f"Logged in as {user_role}")

# --- GLOBAL DISCLAIMER ---
st.warning("⚠️ **DISCLAIMER**: This AI tool is for EDUCATIONAL PURPOSES only. Trading involves risk.")

# 3. AI PROCESSING FUNCTION
def process_stock(ticker, days):
    try:
        data = yf.download(ticker, period=f'{days}d')
        # MACD Calculation
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Displaying MACD Chart
st.subheader("MACD Indicator")
st.line_chart(data[['MACD', 'Signal_Line']])
        if data.empty: return None
        
        data = data.astype(float)
        data['Day'] = np.arange(len(data))
        data['MA50'] = data['Close'].rolling(window=50).mean()
        
        # RSI Logic
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        df_clean = data.dropna()
        X = df_clean[['Day', 'MA50', 'RSI']]
        y = df_clean['Close']
        model = LinearRegression().fit(X, y)
        
        last_val = df_clean.iloc[-1]
        prediction = model.predict([[float(len(data)), float(last_val['MA50']), float(last_val['RSI'])]])
        pred_val = float(prediction.flatten()[0])
        curr_val = float(last_val['Close'])
        acc = model.score(X, y)
        
        return data, df_clean, pred_val, curr_val, acc
    except:
        return None

# 4. ADMIN DASHBOARD LOGIC
if user_role == "Admin" and hash_password(password) == ADMIN_HASH:
    st.title("👨‍✈️ Admin Central Control")
    st.subheader("Global Market Overview (Admin Only)")
    
    # Example: Tracking multiple stocks at once
    track_list = ["AAPL", "TSLA", "SBIN.NS", "BTC-USD"]
    cols = st.columns(4)
    for i, t in enumerate(track_list):
        res = process_stock(t, 200)
        if res:
            cols[i].metric(t, f"${res[2]:.2f}")
    
# 5. USER / MAIN APP LOGIC
st.title("⚔️ AI Stock Battle - Secure Portal")
ticker1 = st.sidebar.text_input("Stock 1", value="GOOGL")
ticker2 = st.sidebar.text_input("Stock 2", value="TSLA")
days = st.sidebar.slider("Days of Data", 100, 500, 250)

if st.sidebar.button("Run AI Analysis"):
    col_a, col_b = st.columns(2)
    
    for i, t in enumerate([ticker1, ticker2]):
        result = process_stock(t, days)
        current_col = col_a if i == 0 else col_b
        
        if result:
            data, df_clean, pred_val, curr_val, acc = result
            diff = ((pred_val - curr_val) / curr_val) * 100
            
            with current_col:
                st.header(f"📊 {t}")
                st.metric("Predicted Price", f"${pred_val:.2f}", f"{diff:.2f}%")
                
                if diff > 2: st.success("🔥 STRONG BUY")
                elif diff > 0: st.warning("⚖️ HOLD")
                else: st.error("⚠️ SELL/AVOID")

                # Professional Plotly Chart
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_width=[0.3, 0.7])
                fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['MA50'], name='MA50', line=dict(color='yellow')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['RSI'], name='RSI', line=dict(color='magenta')), row=2, col=1)
                fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, showlegend=False)

                st.plotly_chart(fig, use_container_width=True)
