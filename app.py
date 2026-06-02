import streamlit as st
import datetime
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk

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

# ---------------------------------------------------------
# PIPELINE INFERENCE EXECUTION LAYER
# ---------------------------------------------------------
@st.cache_resource
def load_ml_pipeline():
    """Loads the true calibrated_xgb model and county geography rules."""
    model = joblib.load('calibrated_wildfire_model.pkl')
    counties = pd.read_csv('county_metadata.csv')
    return model, counties

model, county_df = load_ml_pipeline()

@st.cache_data
def run_cached_inference(scenario_name):
    """Assembles features matching the exact training column alignment cleanly."""
    features = SCENARIOS[scenario_name]
    inference_data = []
    
    for _, row in county_df.iterrows():
        # Keep column naming completely uniform ('lat', 'long') throughout the pipeline
        base = {
            "county": row['county'], 
            "lat": float(row['lat']), 
            "long": float(row['long'])
        }
        base.update(features)
        inference_data.append(base)
        
    df = pd.DataFrame(inference_data)
    
    # EXACT COLUMN LIST AND ORDER MATCHING YOUR TRAINING LOGIC
    feature_cols = ['avgtempF', 'totalSnow_cm', 'humid', 'wind', 'precip', 'sunHour', 'lat', 'long', 'ndvi', 'temp_humid_idx']
    X_infer = df[feature_cols]
    
    # Run inference and append prediction directly to the structurally sound dataframe
    probabilities = model.predict_proba(X_infer)[:, 1]
    df["fire_probability"] = probabilities.astype(float)
    return df

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
    st.write("Map Layer: `Native Pydeck Heatmap`")

with col_map:
    # 1. Fetch our clean calculation matrix
    df_map = run_cached_inference(preset)
    
    # 2. Configure the native WebGL Heatmap layer pointing directly to our unified '[long, lat]' keys
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=df_map,
        get_position="[long, lat]",  # Pointing directly to our uniform dataframe keys
        get_weight="fire_probability",
        radius_pixels=90,      
        intensity=1.5,         
        threshold=0.01,
        opacity=0.85,
    )
    
    # 3. Establish camera perspective centered over Northern California
    view_state = pdk.ViewState(
        latitude=39.5,
        longitude=-121.5,
        zoom=5.8,
        pitch=0
    )
    
    # 4. Render native map component using a token-free open-source style
    st.pydeck_chart(pdk.Deck(
        layers=[heatmap_layer],
        initial_view_state=view_state,
        map_style="carto_light"
    ))