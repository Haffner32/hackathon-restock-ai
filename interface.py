import streamlit as st
import pandas as pd
from prophet import Prophet
import ssl
import logging

# 1. Page Configuration & Styling
st.set_page_config(page_title="Team Synergy | Restock AI", layout="wide")

# Hide Prophet's background logs for a cleaner terminal
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)

st.title("📦 Predictive Restock Assistant")
st.markdown("---")

# 2. Sidebar Configuration
st.sidebar.header("Data Control Center")
url = st.sidebar.text_input(
    "Google Sheet CSV URL", 
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vQY-TxRVYD3vOWOO5yA_FJ9Mr5ccYuOxcnksoTBZTKwacKE7TMi1jGtXSGO7FgzoLni2zqrJxtkURLy/pub?output=csv"
)

# 3. Main Execution Logic
if st.sidebar.button("Run Intelligence Engine"):
    with st.spinner('Analyzing Seasonal Trends and Recent Anomalies...'):
        try:
            # Bypass SSL for Mac
            ssl._create_default_https_context = ssl._create_unverified_context
            
            # Load Data
            data = pd.read_csv(url)
            data['Date'] = pd.to_datetime(data['Date'])
            
            # --- DATA CLEANING & BURN CALCULATION ---
            # Yesterday - Today = Usage
            data['y'] = data['Current_Stock'].shift(1) - data['Current_Stock']
            
            # CRITICAL: If y < 0, it was a restock, not a sale. Set to 0 so AI doesn't break.
            data.loc[data['y'] < 0, 'y'] = 0 
            
            # Prepare for Prophet
            df_prophet = data.rename(columns={'Date': 'ds'}).dropna()

            # 4. Modeling
            # Model 365: Long-term patterns
            m365 = Prophet(yearly_seasonality=True, daily_seasonality=False)
            m365.fit(df_prophet)

            # Model 30: Short-term "panic" detection
            # We use a higher changepoint_prior_scale to make it more reactive to recent spikes
            m30 = Prophet(changepoint_prior_scale=0.5)
            m30.fit(df_prophet.tail(30))

            # 5. Predictions (Next 30 Days)
            future = m365.make_future_dataframe(periods=30)
            
            forecast_365 = m365.predict(future)['yhat'].tail(30).sum()
            forecast_30 = m30.predict(future)['yhat'].tail(30).sum()

            # Fallback Logic: If AI predicts 0 or negative, use historical daily average
            avg_daily_burn = df_prophet['y'].mean()
            fallback = avg_daily_burn * 30
            
            f365_final = max(forecast_365, fallback)
            f30_final = max(forecast_30, fallback)

            # 6. Safe-Maximum Decision
            final_need = max(f365_final, f30_final)
            current_stock = data['Current_Stock'].iloc[-1]
            order_qty = max(0, final_need - current_stock)
            
            # Determine which model won for the UI text
            winning_model = "Seasonal Baseline" if f365_final > f30_final else "Recent Anomaly"

            # --- 7. UI DISPLAY ---
            
            # Metric Row
            col1, col2, col3 = st.columns(3)
            
            # Calculate stock change for the "delta" arrow
            stock_change = data['Current_Stock'].iloc[-1] - data['Current_Stock'].iloc[-2]
            
            col1.metric("Current Stock", f"{current_stock:.0f} Units", delta=f"{stock_change:.0f} (vs Yesterday)")
            col2.metric("Seasonal Forecast", f"{f365_final:.0f} Units")
            col3.metric("Anomaly Forecast", f"{f30_final:.0f} Units")

            st.info(f"🛡️ **Safe-Maximum Logic Active:** Ordering based on the **{winning_model}** forecast to prevent stockouts.")

            # Big Result Box
            st.success(f"## Recommended Order: {order_qty:.0f} Units")
            
            # RPA Button Placeholder
            if st.button("🚀 Execute RPA Browser Bot"):
                st.balloons()
                st.write("Initializing Selenium WebDriver... Logging into supplier portal...")

        except Exception as e:
            st.error(f"Error processing data: {e}")
            st.warning("Check if your Google Sheet has 'Date' and 'Current_Stock' columns correctly named.")

else:
    st.write("Welcome back! Click the button in the sidebar to analyze your inventory.")