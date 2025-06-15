import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np
from PIL import Image
# Importar funciones auxiliares
from utils import (
    cargar_sprites,Generator1, Generator2, Generator3,
    tensor_a_imagen, visualizar_segmento_texto, cargar_generadores, char2idx, idx2char, device)

# Configuración de la página
st.set_page_config(
    page_title="Mario Level Generator", 
    layout="wide",
    page_icon="🍄"
)



def generar_nivel(generador, nombre_modelo):
    """Genera un nivel usando el generador especificado."""
    with torch.no_grad():
        noise = torch.randn(1, 100, 1, 4).to(device)
        fake_level = generador(noise)
        return fake_level[0], nombre_modelo

# ========== INTERFAZ DE STREAMLIT ==========
def main():
    st.title("🍄 Generador de Niveles de Mario")
    st.markdown("---")
    
    # Información sobre los modelos
    st.sidebar.header("📊 Información de los Modelos")
    st.sidebar.markdown("""
    **🔹 GAN Densa**: Usa bloques densos para reutilización eficiente de características.
    
    **🔹 GAN Residual**: Emplea bloques residuales para entrenamiento profundo estable.
    
    **🔹 GAN Tradicional**: Utiliza convoluciones simples, más rápida de entrenar.
    """)
    
    # Cargar recursos
    sprites = cargar_sprites()
    G1, G2, G3 = cargar_generadores()
    
    st.markdown("### 🎮 Genera nuevos niveles de Mario con diferentes arquitecturas GAN")
    st.write("Presiona el botón para generar niveles usando los tres generadores entrenados.")
    
    # Botón principal
    if st.button("🚀 Generar Niveles", type="primary", use_container_width=True):
        with st.spinner("Generando niveles mágicos..."):
            # Generar niveles con los 3 modelos
            nivel1, nombre1 = generar_nivel(G1, "GAN Densa")
            nivel2, nombre2 = generar_nivel(G2, "GAN Residual") 
            nivel3, nombre3 = generar_nivel(G3, "GAN Tradicional")
            
            # Mostrar resultados en columnas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader(f"🏗️ {nombre1}")
                imagen1 = tensor_a_imagen(nivel1, idx2char, sprites)
                st.image(imagen1, use_container_width=True)
                
                # Mostrar versión texto
                with st.expander("Ver como texto"):
                    texto1 = visualizar_segmento_texto(nivel1, idx2char)
                    st.code(texto1, language=None)
            
            with col2:
                st.subheader(f"🔄 {nombre2}")
                imagen2 = tensor_a_imagen(nivel2, idx2char, sprites)
                st.image(imagen2, use_container_width=True)
                
                with st.expander("Ver como texto"):
                    texto2 = visualizar_segmento_texto(nivel2, idx2char)
                    st.code(texto2, language=None)
            
            with col3:
                st.subheader(f"⚡ {nombre3}")
                imagen3 = tensor_a_imagen(nivel3, idx2char, sprites)
                st.image(imagen3, use_container_width=True)
                
                with st.expander("Ver como texto"):
                    texto3 = visualizar_segmento_texto(nivel3, idx2char)
                    st.code(texto3, language=None)
        
        st.success("¡Niveles generados exitosamente! 🎉")
    
    

if __name__ == "__main__":
    main()