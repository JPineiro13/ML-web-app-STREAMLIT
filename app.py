import streamlit as st
import joblib
import numpy as np

# Cargar modelo y transformador
model = joblib.load('model.pkl')
imputer = joblib.load('imputer.pkl')

st.set_page_config(page_title="Predicción de Estaciones", layout="centered")

st.title("Predicción de Estación del Año")
st.markdown("Introduce los datos meteorológicos para predecir la estación:")

# Inputs del usuario
max_temp = st.number_input("Temperatura Máxima (°C)", value=25.0)
mean_temp = st.number_input("Temperatura Media (°C)", value=20.0)
min_temp = st.number_input("Temperatura Mínima (°C)", value=15.0)
max_humidity = st.number_input("Humedad Máxima (%)", value=80.0)
mean_humidity = st.number_input("Humedad Media (%)", value=60.0)
min_humidity = st.number_input("Humedad Mínima (%)", value=40.0)
max_wind = st.number_input("Viento Máximo (Km/h)", value=20.0)
mean_wind = st.number_input("Viento Medio (Km/h)", value=10.0)
precip = st.number_input("Precipitación (mm)", value=2.0)

# Botón de predicción
if st.button("Predecir estación"):
    input_data = np.array([[max_temp, mean_temp, min_temp, max_humidity,
                            mean_humidity, min_humidity, max_wind, mean_wind,
                            precip]])
    
    input_imputed = imputer.transform(input_data)
    prediction = model.predict(input_imputed)[0]

    st.success(f"La estación predicha es: **{prediction.upper()}**")
