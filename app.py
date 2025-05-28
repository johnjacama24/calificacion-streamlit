import streamlit as st
import joblib
import numpy as np

# Cargar modelo
modelo = joblib.load('modelo_calificacion.pkl')

st.title("Predicción de Calificación Final")

asistencia = st.slider("Porcentaje de Asistencia", 0, 100, 80)
nota_parcial = st.slider("Nota del Parcial", 0.0, 5.0, 3.5)

if st.button("Predecir Nota Final"):
    entrada = np.array([[asistencia, nota_parcial]])
    prediccion = modelo.predict(entrada)
    st.success(f"La calificación final estimada es: {prediccion[0]:.2f}")
