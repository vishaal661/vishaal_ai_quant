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

# Passwords
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
st.warning("⚠️ **DISCLAIMER**: This AI tool is for EDUCATIONAL PURPOSES only.")

# 3. AI PROCESSING FUNCTION
def process_stock(ticker, days):
    try:
        # Data Download
        data = yf.download(ticker, period=f'{days}d')
        if data.empty or len(data) < 30: 
            return None
        
        data = data.astype(float)
        
        # --- MACD Calculation (Fixed Logic) ---
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

        # Features for AI
        data['Day'] = np.arange(len(data))
        data['MA50'] = data['Close'].rolling(window=50).mean()
        
        # RSI Logic
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Clean data for AI training
        df_clean = data.dropna().copy()
        if df_clean.empty:
            return None
            
        X = df_clean[['Day', 'MA50', 'RSI']]
        y = df_clean['Close']
        model = LinearRegression().fit(X, y)
        
        # Predictions
        last_val = df_clean.iloc[-1]
        prediction = model.predict([[float(len(data)), float(last_val['MA50']), float(last_val['RSI'])]])
        pred_val = float(prediction.flatten()[0])
        curr_val = float(last_val['Close'])
        acc = model.score(X, y)
        
        # Return all calculated data
        return data, df_clean, pred_val, curr_val, acc
        
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        return None

# 4. ADMIN DASHBOARD
if user_role == "Admin" and hash_password(password) == ADMIN_HASH:
    st.title("👨‍✈️ Admin Central Control")
    track_list = ["AAPL", "TSLA", "BTC-USD", "RELIANCE.NS"]
    cols = st.columns(4)
    for i, t in enumerate(track_list):
        result = process_stock(t, 200)
        if result:
            d, dfc, pv, cv, ac = result
            cols[i].metric(t, f"${cv:.2f}", f"AI: ${pv:.2f}")

# 5. USER MAIN APP
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
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.1, row_width=[0.3, 0.7])
                fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['MA50'], name='MA50', line=dict(color='yellow')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['RSI'], name='RSI', line=dict(color='magenta')), row=2, col=1)
                fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                # --- MACD VISUALIZATION (Fixed for Line 152) ---
                if 'MACD' in data.columns and 'Signal_Line' in data.columns:
                    st.write("---")
                    st.subheader("🛠️ MACD Trend Analysis")
                    # Double brackets are essential for multi-column line charts
                    st.line_chart(data[['MACD', 'Signal_Line']])
                    st.bar_chart(data['MACD'] - data['Signal_Line'])
                else:
                    st.info("MACD" columns not found. Please check.")
        else:
            current_col.error(f"Data not found for {t}")

