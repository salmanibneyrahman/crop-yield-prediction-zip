import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import zipfile
import os
import tempfile

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Bangladesh Crop & Yield Advisor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# LOAD MODELS & ENCODERS
# ============================================================================
@st.cache_resource
def load_models():
    """Load models directly from ZIP file without extraction"""
    try:
        if not os.path.exists('models.zip'):
            st.error("**models.zip** file not found!")
            st.info("Please place **models.zip** in the same folder as this app")
            st.stop()
        
        models = {}
        
        # Load each model directly from ZIP using ZipFile.open()
        with zipfile.ZipFile('models.zip', 'r') as zip_ref:
            # List all files in zip to debug
            file_list = zip_ref.namelist()
            st.info(f"Found files in models.zip: {len(file_list)}")
            
            # Load required models
            required_files = [
                'yield_model.pkl',
                'crop_model.pkl', 
                'label_encoder_crop.pkl',
                'label_encoder_season.pkl',
                'label_encoder_district.pkl'
            ]
            
            for filename in required_files:
                if filename in file_list:
                    with zip_ref.open(filename) as f:
                        # Load from BytesIO for joblib
                        import io
                        model_data = io.BytesIO(f.read())
                        if 'yield_model' in filename:
                            models['yield'] = joblib.load(model_data)
                        elif 'crop_model' in filename:
                            models['crop'] = joblib.load(model_data)
                        elif 'label_encoder_crop' in filename:
                            models['crop_encoder'] = joblib.load(model_data)
                        elif 'label_encoder_season' in filename:
                            models['season_encoder'] = joblib.load(model_data)
                        elif 'label_encoder_district' in filename:
                            models['district_encoder'] = joblib.load(model_data)
                else:
                    st.warning(f"Missing {filename} in models.zip")
            
            if len(models) == 5:
                st.success("All models loaded successfully!")
                return models
            else:
                st.error("Missing required model files in models.zip")
                st.info(f"Expected 5 files, loaded {len(models)}")
                return None
                
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("Check that models.zip contains the 5 required .pkl files")
        return None
models = load_models()
if models is None:
    st.stop()

# ============================================================================
# SAFE LOCATION + WEATHER HELPERS
# ============================================================================

@st.cache_data(ttl=600)
def get_user_location():
    """
    Try multiple IP geolocation APIs.
    NOTE: On Streamlit Cloud / GitHub hosting this usually returns
    the **server** location (not the end-user), so treat as approximate only.
    """
    apis = [
        # IPAPI (HTTPS, often works)
        {
            "name": "ipapi.co",
            "url": "https://ipapi.co/json",
            "parser": lambda d: {
                "city": d.get("city"),
                "country": d.get("country_name"),
                "lat": d.get("latitude"),
                "lon": d.get("longitude"),
            },
        },
        # IPINFO (HTTPS)
        {
            "name": "ipinfo.io",
            "url": "https://ipinfo.io/json",
            "parser": lambda d: {
                "city": d.get("city"),
                "country": d.get("country"),
                "lat": float(d.get("loc", "0,0").split(",")[0])
                if d.get("loc")
                else None,
                "lon": float(d.get("loc", "0,0").split(",")[1])
                if d.get("loc")
                else None,
            },
        },
    ]

    for api in apis:
        try:
            r = requests.get(api["url"], timeout=5)
            if r.status_code == 200:
                data = r.json()
                loc = api["parser"](data)
                if loc.get("lat") is not None and loc.get("lon") is not None:
                    return loc
        except Exception:
            continue

    return None


@st.cache_data(ttl=600)
def get_weather(lat: float, lon: float):
    """
    Fetch basic weather (temp & humidity) from Open-Meteo.
    """
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m"
            "&daily=temperature_2m_max,temperature_2m_min,"
            "relative_humidity_2m_max,relative_humidity_2m_min"
        )
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return None

        data = r.json()
        current = data.get("current", {})
        daily = data.get("daily", {})

        return {
            "avg_temp": current.get("temperature_2m"),
            "avg_humidity": current.get("relative_humidity_2m"),
            "min_temp": (daily.get("temperature_2m_min") or [None])[0],
            "max_temp": (daily.get("temperature_2m_max") or [None])[0],
            "min_humidity": (daily.get("relative_humidity_2m_min") or [None])[0],
            "max_humidity": (daily.get("relative_humidity_2m_max") or [None])[0],
        }
    except Exception:
        return None


# ============================================================================
# HEADER
# ============================================================================
st.title("🌾 Bangladesh Crop & Yield Intelligence System")

st.markdown(
    """
This app uses machine learning models trained on **Bangladesh SPAS data**  
to recommend suitable crops and estimate yield (tons/hectare).

- You can use **Auto mode** (approximate location from IP) or  
- **Manual mode** (you control all inputs – recommended for accuracy).
"""
)

st.markdown("---")

# ============================================================================
# INPUT MODE SELECTION
# ============================================================================
mode = st.radio(
    "How do you want to provide weather data?",
    ["Auto (approximate from IP)", "Manual"],
    horizontal=True,
)

st.markdown(
    "> ℹ️ On public hosting (Streamlit Cloud / GitHub), auto-detected location "
    "often reflects the **server**, not your exact device. Use **Manual** for accuracy."
)

st.markdown("---")

# ============================================================================
# COLLECT INPUTS
# ============================================================================
available_districts = list(models["district_encoder"].classes_)
available_seasons = list(models["season_encoder"].classes_)

col_left, col_right = st.columns(2)

# Shared inputs (district, season, area)
with col_left:
    st.subheader("Farm & Season")
    district = st.selectbox("District", options=available_districts)
    season = st.selectbox("Season", options=available_seasons)
    area_ha = st.number_input(
        "Cultivated Area (hectares)",
        min_value=0.1,
        max_value=1_000_000.0,
        value=10_000.0,
        step=100.0,
    )

with col_right:
    st.subheader("Weather Conditions (°C, %)")
    if mode == "Auto (approximate from IP)":
        loc = get_user_location()
        if loc is not None:
            st.info(
                f"Approximate server location: **{loc.get('city', 'Unknown')}, "
                f"{loc.get('country', 'Unknown')}**  \n"
                "Note: On cloud hosting this is usually **not your device location**."
            )
            weather = get_weather(loc["lat"], loc["lon"])
        else:
            st.warning("Could not detect location. Please switch to Manual mode.")
            weather = None

        if weather is not None:
            st.write(
                f"Detected Avg Temp: **{weather['avg_temp']:.1f}°C**, "
                f"Avg Humidity: **{weather['avg_humidity']:.0f}%**"
            )

            min_temp = st.number_input(
                "Min Temp (°C)",
                value=float(weather["min_temp"])
                if weather["min_temp"] is not None
                else 20.0,
            )
            avg_temp = st.number_input(
                "Avg Temp (°C)",
                value=float(weather["avg_temp"]) if weather["avg_temp"] is not None else 25.0,
            )
            max_temp = st.number_input(
                "Max Temp (°C)",
                value=float(weather["max_temp"])
                if weather["max_temp"] is not None
                else 32.0,
            )

            min_humidity = st.number_input(
                "Min Humidity (%)",
                min_value=0,
                max_value=100,
                value=int(weather["min_humidity"])
                if weather["min_humidity"] is not None
                else 40,
            )
            avg_humidity = st.number_input(
                "Avg Humidity (%)",
                min_value=0,
                max_value=100,
                value=int(weather["avg_humidity"])
                if weather["avg_humidity"] is not None
                else 70,
            )
            max_humidity = st.number_input(
                "Max Humidity (%)",
                min_value=0,
                max_value=100,
                value=int(weather["max_humidity"])
                if weather["max_humidity"] is not None
                else 95,
            )
        else:
            # Fallback manual controls
            st.warning("Falling back to manual weather input.")
            min_temp = st.number_input("Min Temp (°C)", value=20.0)
            avg_temp = st.number_input("Avg Temp (°C)", value=26.0)
            max_temp = st.number_input("Max Temp (°C)", value=32.0)

            min_humidity = st.number_input(
                "Min Humidity (%)", min_value=0, max_value=100, value=40
            )
            avg_humidity = st.number_input(
                "Avg Humidity (%)", min_value=0, max_value=100, value=70
            )
            max_humidity = st.number_input(
                "Max Humidity (%)", min_value=0, max_value=100, value=95
            )

    else:  # Manual mode
        min_temp = st.number_input("Min Temp (°C)", value=20.0)
        avg_temp = st.number_input("Avg Temp (°C)", value=26.0)
        max_temp = st.number_input("Max Temp (°C)", value=32.0)

        min_humidity = st.number_input(
            "Min Humidity (%)", min_value=0, max_value=100, value=40
        )
        avg_humidity = st.number_input(
            "Avg Humidity (%)", min_value=0, max_value=100, value=70
        )
        max_humidity = st.number_input(
            "Max Humidity (%)", min_value=0, max_value=100, value=95
        )

st.markdown("---")

# ============================================================================
# PREDICT BUTTON
# ============================================================================
if st.button("🔮 Predict Recommended Crop & Yield", use_container_width=True):
    try:
        # Encode season & district
        season_code = models["season_encoder"].transform([season])[0]
        district_code = models["district_encoder"].transform([district])[0]

        # Build feature arrays
        yield_features = np.array(
            [
                [
                    area_ha,
                    avg_temp,
                    avg_humidity,
                    max_temp,
                    min_temp,
                    max_humidity,
                    min_humidity,
                ]
            ]
        )

        crop_features = np.array(
            [
                [
                    avg_temp,
                    avg_humidity,
                    max_temp,
                    min_temp,
                    max_humidity,
                    min_humidity,
                    season_code,
                    district_code,
                ]
            ]
        )

        # Predictions
        yield_per_ha = float(models["yield"].predict(yield_features)[0])
        total_production = yield_per_ha * area_ha

        crop_idx = int(models["crop"].predict(crop_features)[0])
        crop_name = models["crop_encoder"].inverse_transform([crop_idx])[0]

        # Confidence (if supported)
        if hasattr(models["crop"], "predict_proba"):
            proba = models["crop"].predict_proba(crop_features)[0]
            conf = proba.max() * 100
            conf_str = f"{conf:.1f}%"
        else:
            conf_str = "N/A"

        # Display results
        st.subheader("Results")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Recommended Crop", crop_name.upper(), f"Confidence: {conf_str}")
        with c2:
            st.metric("Yield (tons / ha)", f"{yield_per_ha:.2f}")
        with c3:
            st.metric("Total Production (tons)", f"{total_production:,.0f}")

        st.markdown("### Input Summary")
        summary_df = pd.DataFrame(
            {
                "Parameter": [
                    "District",
                    "Season",
                    "Area (ha)",
                    "Min Temp (°C)",
                    "Avg Temp (°C)",
                    "Max Temp (°C)",
                    "Min Humidity (%)",
                    "Avg Humidity (%)",
                    "Max Humidity (%)",
                ],
                "Value": [
                    district,
                    season,
                    f"{area_ha:,.1f}",
                    f"{min_temp:.1f}",
                    f"{avg_temp:.1f}",
                    f"{max_temp:.1f}",
                    f"{min_humidity}",
                    f"{avg_humidity}",
                    f"{max_humidity}",
                ],
            }
        )
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

    except Exception as e:

        st.error(f"Error during prediction: {e}")

