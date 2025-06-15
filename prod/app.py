import random
import streamlit as st

st.set_page_config(page_title="Mario Generator", layout="centered")

st.title("Generador de niveles de Mario!!")
st.write("Apreta el botón para generar un nuevo nivel de Mario.")

if st.button("Generar Nivel"):
    numero = random.randint(1, 100) 
    st.write("¡Nivel generado! "+ str(numero) + " puntos de dificultad.")

