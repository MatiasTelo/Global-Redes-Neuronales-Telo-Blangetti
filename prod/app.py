import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np
from PIL import Image
import os

# Configuración de la página
st.set_page_config(
    page_title="Mario Level Generator", 
    layout="wide",
    page_icon="🍄"
)

# Definir vocabulario y constantes
CHARACTER_SET = ['X', 'S', '-', '?', 'Q', 'E', '<', '>', '[', ']', 'o', 'B', 'b']
char2idx = {char: idx for idx, char in enumerate(CHARACTER_SET)}
idx2char = {idx: char for char, idx in char2idx.items()}
VOCAB_SIZE = len(CHARACTER_SET)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== DEFINICIÓN DE MODELOS ==========

# Bloques base
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.growth_rate = growth_rate

        channels = in_channels
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, growth_rate, kernel_size=3, padding=1, bias=False)
            ))
            channels += growth_rate

        self.transition = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, in_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return self.transition(torch.cat(features, 1))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return x + self.block(x)

# ========== MODELO 1: GAN DENSA ==========
class DenseGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate=32, num_layers=4,
                 kernel_size=4, stride=2, padding=1):
        super(DenseGBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dense_block = DenseBlock(out_channels, growth_rate=growth_rate, num_layers=num_layers)

    def forward(self, x):
        x = self.upsample(x)
        x = self.relu(self.norm(x))
        x = self.dense_block(x)
        return x

class Generator1(nn.Module):
    def __init__(self, vocab_size, latent_dim=100, n_G=64):
        super(Generator1, self).__init__()
        self.net = nn.Sequential(
            DenseGBlock(in_channels=latent_dim, out_channels=n_G * 8,
                        kernel_size=(3, 5), stride=(1, 2), padding=(0, 1)),
            DenseGBlock(in_channels=n_G * 8, out_channels=n_G * 4),
            DenseGBlock(in_channels=n_G * 4, out_channels=n_G * 2),
            DenseBlock(in_channels=n_G * 2),
            DenseBlock(in_channels=n_G * 2),
            nn.ConvTranspose2d(n_G * 2, vocab_size,
                               kernel_size=(3, 1),
                               stride=(1, 1),
                               padding=(0, 2), bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# ========== MODELO 2: GAN RESIDUAL ==========
class ResidualGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(ResidualGBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.relu(self.norm(x))
        x = self.residual(x)
        return x

class Generator2(nn.Module):
    def __init__(self, vocab_size, latent_dim=100, n_G=64):
        super(Generator2, self).__init__()
        self.net = nn.Sequential(
            ResidualGBlock(in_channels=latent_dim, out_channels=n_G * 8,
                           kernel_size=(3, 5), stride=(1, 2), padding=(0, 1)),
            ResidualGBlock(in_channels=n_G * 8, out_channels=n_G * 4),
            ResidualGBlock(in_channels=n_G * 4, out_channels=n_G * 2),
            ResidualBlock(n_G * 2),
            ResidualBlock(n_G * 2),
            nn.ConvTranspose2d(n_G * 2, vocab_size,
                               kernel_size=(3, 1),
                               stride=(1, 1),
                               padding=(0, 2), bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# ========== MODELO 3: GAN TRADICIONAL ==========
class G_block(nn.Module):
    def __init__(self, out_channels, in_channels, kernel_size=4, strides=2, padding=1):
        super(G_block, self).__init__()
        self.conv2d_trans = nn.ConvTranspose2d(in_channels, out_channels,
                                               kernel_size, strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))

class Generator3(nn.Module):
    def __init__(self, vocab_size, latent_dim=100, n_G=64):
        super(Generator3, self).__init__()
        self.net = nn.Sequential(
            G_block(in_channels=latent_dim, out_channels=n_G * 8,
                    kernel_size=(3, 5), strides=(1, 2), padding=(0, 1)),
            G_block(in_channels=n_G * 8, out_channels=n_G * 4),
            G_block(in_channels=n_G * 4, out_channels=n_G * 2),
            nn.ConvTranspose2d(n_G * 2, vocab_size,
                               kernel_size=(3, 1),
                               stride=(1, 1),
                               padding=(0, 2), bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# ========== FUNCIONES DE UTILIDAD ==========
@st.cache_data
def cargar_sprites():
    """Carga los sprites de Mario desde la carpeta de sprites."""
    sprites = {}
    sprite_path = "../data/Sprites"
    
    # Mapeo de archivos a caracteres
    sprite_mapping = {
        "D.png": "<",
        "I.png": ">", 
        "P.png": "B",
        "A.png": "?",
        "X.png": "X",
        "S.png": "S",
        "-.png": "-",
        "Q.png": "Q",
        "E.png": "E",
        "[.png": "[",
        "].png": "]",
        "o.png": "o",
        "b.png": "b"
    }
    
    if os.path.exists(sprite_path):
        for filename in os.listdir(sprite_path):
            if filename.endswith(".png"):
                char = sprite_mapping.get(filename, filename.split('.')[0])
                try:
                    sprites[char] = Image.open(os.path.join(sprite_path, filename)).convert("RGBA")
                except Exception as e:
                    st.warning(f"No se pudo cargar el sprite {filename}: {e}")
    else:
        # Crear sprites simples con colores si no existen los archivos
        tile_size = (16, 16)
        colors = {
            'X': (139, 69, 19),    # Marrón para bloques
            'S': (0, 128, 0),      # Verde para suelo
            '-': (135, 206, 235),  # Azul cielo para aire
            '?': (255, 215, 0),    # Dorado para bloques de pregunta
            'Q': (255, 215, 0),    # Dorado
            'E': (255, 0, 0),      # Rojo para enemigos
            '<': (0, 255, 0),      # Verde para tubos
            '>': (0, 255, 0),      # Verde para tubos
            '[': (128, 128, 128),  # Gris para plataformas
            ']': (128, 128, 128),  # Gris para plataformas
            'o': (255, 255, 0),    # Amarillo para monedas
            'B': (139, 69, 19),    # Marrón para bloques
            'b': (160, 82, 45)     # Marrón claro para bloques rotos
        }
        
        for char, color in colors.items():
            img = Image.new('RGBA', tile_size, color + (255,))
            sprites[char] = img
    
    return sprites

def tensor_a_imagen(one_hot_tensor, idx2char, sprites, tile_size=(16, 16)):
    """Convierte un tensor one-hot en una imagen usando sprites."""
    idx_tensor = one_hot_tensor.argmax(dim=0).cpu().numpy()
    rows, cols = idx_tensor.shape
    tile_width, tile_height = tile_size
    
    output = Image.new('RGBA', (cols * tile_width, rows * tile_height), (135, 206, 235, 255))
    
    for y in range(rows):
        for x in range(cols):
            char = idx2char[idx_tensor[y][x]]
            sprite = sprites.get(char)
            if sprite:
                output.paste(sprite, (x * tile_width, y * tile_height), mask=sprite)
    
    return output

def visualizar_segmento_texto(one_hot_tensor, idx2char):
    """Convierte un tensor one-hot a texto."""
    idx_tensor = one_hot_tensor.argmax(dim=0).cpu().numpy()
    result = ""
    for row in idx_tensor:
        line = ''.join(idx2char[i] for i in row)
        result += line + "\n"
    return result

@st.cache_resource
def cargar_generadores():
    """Carga los tres generadores."""
    G1 = Generator1(vocab_size=VOCAB_SIZE)
    G2 = Generator2(vocab_size=VOCAB_SIZE)
    G3 = Generator3(vocab_size=VOCAB_SIZE)
    
    # Intentar cargar modelos entrenados si existen
    # models_path = "mod"
    
    try:
        if os.path.exists("generador1_entrenado.pth"):
            G1.load_state_dict(torch.load("generador1_entrenado.pth", map_location=device))
        if os.path.exists("generador2_entrenado.pth"):
            G2.load_state_dict(torch.load("generador2_entrenado.pth", map_location=device))
        if os.path.exists("generador3_entrenado.pth"):
            G3.load_state_dict(torch.load("generador3_entrenado.pth", map_location=device))
    except Exception as e:
        st.warning(f"No se pudieron cargar los modelos entrenados: {e}")
        st.info("Se usarán modelos con pesos aleatorios.")
    
    G1.to(device)
    G2.to(device) 
    G3.to(device)
    
    G1.eval()
    G2.eval()
    G3.eval()
    
    return G1, G2, G3

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