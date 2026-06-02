import streamlit as st
import datetime
import pandas as pd
import numpy as np
import joblib
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

st.set_page_config(page_title="CA Wildfire Predictor", layout="wide")
st.title("🔥 Northern California Wildfire Probability Engine")

SCENARIOS = {
    "Winter Normal": {
        "avgtempF": 45.0, "totalSnow_cm": 4.5, "humid": 75.0, "wind": 7.0, 
        "precip": 0.2, "sunHour": 7.0, "ndvi": 0.42, "temp_humid_idx": 0.6
    },
    "Summer Normal": {
        "avgtempF": 74.0, "totalSnow_cm": 0.0, "humid": 22.0, "wind": 10.0, 
        "precip": 0.0, "sunHour": 13.0, "ndvi": 0.22, "temp_humid_idx": 4.5
    },
    "Summer Extreme Heat": {
        "avgtempF": 87.0, "totalSnow_cm": 0.0, "humid": 8.0, "wind": 18.0, 
        "precip": 0.0, "sunHour": 14.5, "ndvi": 0.12, "temp_humid_idx": 9.2
    },
    "Winter High Precipitation": {
        "avgtempF": 41.0, "totalSnow_cm": 15.2, "humid": 95.0, "wind": 14.0, 
        "precip": 3.8, "sunHour": 3.0, "ndvi": 0.48, "temp_humid_idx": 0.2
    }
}

@st.cache_resource
def load_ml_pipeline():
    """Loads the true calibrated_xgb model and county geography rules."""
    model = joblib.load('calibrated_wildfire_model.pkl')
    counties = pd.read_csv('county_metadata.csv')
    return model, counties

model, county_df = load_ml_pipeline()

@st.cache_data
def run_cached_inference(scenario_name):
    """Assembles features sequentially matching the exact training column alignment."""
    features = SCENARIOS[scenario_name]
    inference_data = []
    
    for _, row in county_df.iterrows():
        base = {"county": row['county'], "lat": row['lat'], "long": row['long']}
        base.update(features)
        inference_data.append(base)
        
    df = pd.DataFrame(inference_data)
    
    feature_cols = ['avgtempF', 'totalSnow_cm', 'humid', 'wind', 'precip', 'sunHour', 'lat', 'long', 'ndvi', 'temp_humid_idx']
    X_infer = df[feature_cols]
    
    probabilities = model.predict_proba(X_infer)[:, 1]
    df["fire_probability"] = probabilities
    return df

@st.cache_resource
def generate_raster_heatmap(scenario_name):
    """SOLUTION 3: Compiles a smooth continuous HeatMap plume using localized points."""
    df_results = run_cached_inference(scenario_name)
    
    # Initialize basic map centered over Northern CA
    m = folium.Map(location=[39.5, -121.5], zoom_start=6, tiles="cartodbpositron")
    
    df_results['lat'] = df_results['lat'].astype(float)
    df_results['long'] = df_results['long'].astype(float)
    df_results['fire_probability'] = df_results['fire_probability'].astype(float)
    
    # Parse data array matching Leaflet input shape: [lat, long, weight]
    heatmap_data = df_results[["lat", "long", "fire_probability"]].values.tolist()
    
    # Inject continuous raster overlay framework
    HeatMap(
        data=heatmap_data,
        radius=45,       # Smooth bleeding adjustments
        blur=25,
        min_opacity=0.15,
        max_zoom=10,
        gradient={0.1: "blue", 0.3: "green", 0.5: "orange", 0.75: "red"}
    ).add_to(m)
    
    return m

# ---------------------------------------------------------
# APP INTERFACE DISPLAY
# ---------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    today = datetime.date.today()
    max_date = today + datetime.timedelta(days=7)
    selected_date = st.date_input("Inference Target Date", today, min_value=today, max_value=max_date)
    
    st.markdown("---")
    st.subheader("📚 Demonstration Presets")
    preset = st.radio("Choose Scenario State:", list(SCENARIOS.keys()))

col_metrics, col_map = st.columns([1, 2])

with col_metrics:
    st.info(f"📋 **Active Features: {preset}**")
    st.write(pd.DataFrame([SCENARIOS[preset]]).T.rename(columns={0: "Value"}))
    
    st.markdown("---")
    st.success("⚡ **Inference Execution Log**")
    st.write("Model Framework: `CalibratedClassifierCV`")
    st.write("Base Estimator: `XGBClassifier`")
    st.write("Cache State: **Active (Instant Return)**")

with col_map:
    map_object = generate_raster_heatmap(preset)
    st_folium(map_object, width=800, height=520, key=f"heatmap_view_{preset}")
