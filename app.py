import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(page_title="Lagos Traffic Congestion Predictor", page_icon="🚦", layout="centered")

# ---------------------------------------------------------------------------
# LOAD MODEL (cached so it only loads once per session, not on every rerun)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("lagos_traffic_model.joblib")

model = load_model()

ROUTES = ['Agege Motor Road', 'Ajah-VI', 'Apapa-Oshodi Expressway',
          'Badagry Expressway', 'Ikeja-Airport Road', 'Ikorodu Road',
          'Lekki-Epe Expressway', 'Mile 2-Apapa', 'Ojota-Ketu',
          'Third Mainland Bridge']
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

st.title("🚦 Lagos Traffic Congestion Predictor")
st.caption("3MTT AI/ML Fellowship Project — Muideen Abogunrin")
st.write("Fill in the trip and road conditions below to predict the expected congestion level.")

# ---------------------------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        route = st.selectbox("Route", ROUTES)
        day_of_week = st.selectbox("Day of week", DAYS)
        hour = st.slider("Hour of day (0–23)", 0, 23, 8)
        month = st.selectbox("Month", list(range(1, 13)), index=datetime.now().month - 1)
        day_of_month = st.slider("Day of month", 1, 31, 15)
        vehicle_count_est = st.slider("Estimated vehicle count", 50, 1800, 650)
        lanes = st.selectbox("Number of lanes", [4, 6, 8, 10], index=1)
        length_km = st.slider("Route length (km)", 5.0, 30.0, 15.0)

    with col2:
        is_weekend = st.checkbox("Weekend")
        is_public_holiday = st.checkbox("Public holiday")
        is_school_day = st.checkbox("School day", value=True)
        has_toll = st.checkbox("Has toll gate")
        rain_intensity = st.select_slider("Rain intensity (0=none, 3=heavy)", options=[0, 1, 2, 3], value=0)
        visibility_km = st.slider("Visibility (km)", 0.5, 10.0, 8.0)
        has_accident = st.checkbox("Accident reported")
        has_roadwork = st.checkbox("Roadwork ongoing")
        has_police_checkpoint = st.checkbox("Police checkpoint present")
        has_event_nearby = st.checkbox("Event nearby (concert, match, etc.)")
        fuel_scarcity = st.checkbox("Fuel scarcity in effect")
        is_market_day_route = st.checkbox("Market day on this route")

    submitted = st.form_submit_button("Predict Congestion", use_container_width=True)

# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------
if submitted:
    record = {
        "route": route,
        "day_of_week": day_of_week,
        "hour": hour,
        "is_weekend": int(is_weekend),
        "is_public_holiday": int(is_public_holiday),
        "is_school_day": int(is_school_day),
        "lanes": lanes,
        "length_km": length_km,
        "has_toll": int(has_toll),
        "rain_intensity": rain_intensity,
        "visibility_km": visibility_km,
        "has_accident": int(has_accident),
        "has_roadwork": int(has_roadwork),
        "has_police_checkpoint": int(has_police_checkpoint),
        "has_event_nearby": int(has_event_nearby),
        "fuel_scarcity": int(fuel_scarcity),
        "is_market_day_route": int(is_market_day_route),
        "vehicle_count_est": vehicle_count_est,
        "month": month,
        "day_of_month": day_of_month,
    }

    row = pd.DataFrame([record])
    prediction = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    classes = model.named_steps["clf"].classes_

    colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    st.subheader(f"{colors.get(prediction, '')} Predicted congestion: **{prediction}**")

    proba_df = pd.DataFrame({"Level": classes, "Probability": proba}).sort_values("Probability", ascending=False)
    st.bar_chart(proba_df.set_index("Level"))

    with st.expander("See raw input sent to the model"):
        st.json(record)

st.divider()
st.caption("Model: tuned Random Forest, trained on 25,000 Lagos traffic records · macro-F1 ≈ 0.83")
