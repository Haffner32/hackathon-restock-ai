import streamlit as st  # This MUST be line 1
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
import ssl
import logging

# 1. Page Configuration
st.set_page_config(page_title="Team Synergy | Restock AI", layout="wide")

# Hide Prophet's background logs
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

# --- LOAD DATA FIRST TO POPULATE FILTER ---
@st.cache_data
def load_base_data(url):
    ssl._create_default_https_context = ssl._create_unverified_context
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

try:
    raw_data = load_base_data(url)

    # Stock ID Selector
    unique_stocks = sorted(raw_data['StockID'].unique())
    selected_stock = st.sidebar.selectbox("Select Stock ID to Analyze", unique_stocks)

    # Filter data for the specific Stock ID
    data = raw_data[raw_data['StockID'] == selected_stock].copy()
    item_name = data['Item Name'].iloc[0] if not data.empty else "Unknown"

    st.sidebar.write(f"**Item Name:** {item_name}")

except Exception as e:
    st.sidebar.error(f"Could not connect to Google Sheet.")
    st.stop()

# 3. Main Execution Logic
if st.sidebar.button("Run Intelligence Engine"):
    with st.spinner(f'Calculating Seasonal Trends for {item_name}...'):
        try:
            # --- DATA CLEANING & BURN CALCULATION ---
            data = data.sort_values('Date')
            data['y'] = data['Current_Stock'].shift(1) - data['Current_Stock']
            data.loc[data['y'] < 0, 'y'] = 0

            df_prophet = data[['Date', 'y']].rename(columns={'Date': 'ds'}).dropna()

            if len(df_prophet) < 2:
                st.warning("Not enough data to generate a forecast.")
            else:
                # 4. Modeling
                m365 = Prophet(yearly_seasonality=True, daily_seasonality=False)
                m365.fit(df_prophet)

                m30 = Prophet(changepoint_prior_scale=0.5)
                m30.fit(df_prophet.tail(30))

                # 5. Predictions
                future = m365.make_future_dataframe(periods=30)
                forecast_res_365 = m365.predict(future)

                f365_val = forecast_res_365['yhat'].tail(30).sum()
                f30_val = m30.predict(future)['yhat'].tail(30).sum()

                # Fallback Logic
                avg_daily_burn = df_prophet['y'].mean()
                fallback = avg_daily_burn * 30

                # Final Calculations
                f365_final = max(f365_val, fallback)
                f30_final = max(f30_val, fallback)
                final_need = max(f365_final, f30_final)

                current_stock = data['Current_Stock'].iloc[-1]
                order_qty = max(0, final_need - current_stock)

                # Logic explanation
                winning_model = "Seasonal Baseline" if f365_final > f30_final else "Recent Anomaly Detection"

                # --- 7. UI DISPLAY ---
                st.header(f"Restock Intelligence: {item_name} (ID: {selected_stock})")

                # Metrics Row
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                stock_change = data['Current_Stock'].iloc[-1] - data['Current_Stock'].iloc[-2] if len(data) > 1 else 0

                m_col1.metric("On Hand", f"{current_stock:.0f}", delta=f"{stock_change:.0f}")
                m_col2.metric("Seasonal Forecast (30d)", f"{f365_final:.0f}")
                m_col3.metric("Anomaly Forecast (30d)", f"{f30_final:.0f}")
                m_col4.metric("Avg Daily Burn", f"{avg_daily_burn:.1f}")

                # Result Box
                st.success(f"### 🛒 Recommended Order: {order_qty:.0f} Units")
                st.info(f"💡 **AI Logic:** The system chose the **{winning_model}** because it predicted a higher demand, ensuring you don't stock out.")

                # --- 8. VISUALIZATIONS ---
                st.markdown("---")
                tab1, tab2 = st.tabs(["📈 Inventory History", "🤖 Prophet AI Model"])

                with tab1:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("Historical Stock Levels")
                        st.line_chart(data.set_index('Date')['Current_Stock'])
                    with c2:
                        st.subheader("Daily Usage Pattern")
                        st.area_chart(data.set_index('Date')['y'])

                with tab2:
                    st.subheader("Interactive Seasonal Projection")
                    fig = plot_plotly(m365, forecast_res_365)
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction Error: {e}")

else:
    st.info(f"Dashboard Ready. Select a Stock ID and click 'Run Intelligence Engine'.")