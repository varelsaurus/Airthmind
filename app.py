import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import xgboost # Wajib import meski tidak dipanggil langsung

# --- 1. SETUP & CONFIG ---
st.set_page_config(
    page_title="AirthMind Dashboard",
    page_icon="🍃",
    layout="wide"
)

# --- 2. LOAD MODELS (CACHE) ---
@st.cache_resource
def load_airthmind_models():
    """Load semua model .pkl sekali saja agar cepat."""
    try:
        model_xgb = joblib.load('models/airthmind_model.pkl')
        scaler = joblib.load('models/airthmind_scaler.pkl')
        le_zone = joblib.load('models/airthmind_le_zone.pkl')
        model_prophet = joblib.load('models/airthmind_prophet_model.pkl')
        return model_xgb, scaler, le_zone, model_prophet
    except FileNotFoundError as e:
        st.error(f"❌ Error: File model tidak ditemukan! ({e})")
        st.stop()

# Load models
xgb_model, scaler, le_zone, prophet_model = load_airthmind_models()

# --- 3. UI: HEADER ---
st.title("🍃 AirthMind: Intelligent Indoor Air Monitor")
st.markdown("### AI-Powered Air Quality Prediction & Recommendation System")
st.caption("Predicting Health Zones (XGBoost) & Future Trends (Prophet)")

# --- 4. SIDEBAR: INPUT SENSOR ---
st.sidebar.header("📡 AeroSense Input Simulation")

# Input Parameter (8 Fitur Model)
# Urutan HARUS SAMA dengan saat training: 
# ['Temperature', 'Humidity', 'CO2', 'PM2.5', 'TVOC', 'Ventilation_Encoded', 'CO', 'PM10']

temp = st.sidebar.slider("Temperature (°C)", 15.0, 35.0, 24.5)
hum = st.sidebar.slider("Humidity (%)", 20, 90, 50)
co2 = st.sidebar.slider("CO₂ (ppm)", 400, 3000, 800)
pm25 = st.sidebar.slider("PM2.5 (µg/m³)", 0, 500, 25)
pm10 = st.sidebar.slider("PM10 (µg/m³)", 0, 500, 40) # Slider baru
tvoc = st.sidebar.slider("TVOC (ppb)", 0, 2000, 150)
co = st.sidebar.slider("CO (ppm)", 0, 50, 2)

st.sidebar.markdown("---")
vent_status = st.sidebar.radio("Ventilation Status", ["Closed", "Open"], index=0)
vent_encoded = 1 if vent_status == "Open" else 0

# --- 5. LOGIKA UTAMA ---
if st.button("🚀 Analyze Air Quality", type="primary"):
    
    # --- A. KLASIFIKASI ZONA (XGBoost) ---
    
    # 1. Susun array input (Urutan fitur SANGAT PENTING!)
    # Urutan: ['Temperature', 'Humidity', 'CO2', 'PM2_5', 'TVOC', 'Ventilation_Encoded', 'CO', 'PM10']
    # Catatan: Perhatikan urutan di notebook Anda. Di sini saya sesuaikan dengan yang umum kita pakai.
    # Jika di notebook urutannya beda, sesuaikan urutan variabel di dalam kurung siku di bawah ini.
    input_features = np.array([[temp, hum, co2, pm25, tvoc, vent_encoded, co, pm10]])
    
    # 2. Scaling (Wajib!)
    input_scaled = scaler.transform(input_features)
    
    # 3. Prediksi
    zone_idx = xgb_model.predict(input_scaled)[0]
    zone_name = le_zone.inverse_transform([zone_idx])[0]
    
    
    # --- B. PREDIKSI TREN PM2.5 (Prophet) ---
    
    # 1. Buat tanggal masa depan (6 jam ke depan)
    future = prophet_model.make_future_dataframe(periods=6, freq='H')
    
    # 2. Isi regressor (Ventilation) dengan status saat ini
    # (Asumsi ventilasi tidak berubah dalam 6 jam ke depan)
    future['Ventilation_Encoded'] = vent_encoded
    
    # 3. Forecast
    forecast = prophet_model.predict(future)
    forecast_6h = forecast.tail(6)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    
    # --- C. TAMPILAN DASHBOARD ---
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Current Status")
        
        # Logika Warna & Pesan
        if zone_name == "Healthy":
            st.success(f"# ✅ {zone_name}")
            st.markdown("**Air quality is good.** Enjoy your healthy space!")
        elif zone_name == "Moderate":
            st.warning(f"# ⚠️ {zone_name}")
            st.markdown("**Caution:** Air quality is degrading. Consider turning on air purifier.")
        else: # Critical
            st.error(f"# 🚨 {zone_name}")
            st.markdown("**WARNING!** Hazardous air detected. Open windows immediately!")
            
        # Detail Indikator Bahaya (Rule-based feedback)
        with st.expander("See Details"):
            if pm25 > 55: st.error(f"PM2.5 High: {pm25} µg/m³")
            if pm10 > 100: st.error(f"PM10 High: {pm10} µg/m³")
            if co2 > 1500: st.error(f"CO2 High: {co2} ppm")
            if co > 10: st.error(f"CO Hazardous: {co} ppm")
            if tvoc > 1000: st.error(f"Chemicals (TVOC) High: {tvoc} ppb")
            
    with col2:
        st.subheader("PM2.5 Forecast (Next 6 Hours)")
        
        # Chart
        st.line_chart(forecast_6h.set_index('ds')['yhat'])
        
        # Stats
        avg_pred = forecast_6h['yhat'].mean()
        st.metric("Avg Predicted PM2.5", f"{avg_pred:.1f} µg/m³", 
                  delta=f"{avg_pred - pm25:.1f} from now", 
                  delta_color="inverse")

# --- 6. FOOTER ---
st.markdown("---")
st.info("💡 **Tips:** PM2.5 is the main indicator for health forecast. CO2 indicates ventilation needs.")