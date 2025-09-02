# Applicación TFG
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import zipfile, io, os, glob
import numpy as np
from pathlib import Path

# ------------------ CONFIG PÁGINA ------------------
st.set_page_config(
    page_title="Detección de Defectos en Paneles Fotovoltaicos - TFG",
    layout="wide",
)

# Rutas de logos
LOGO_UPV_PATH = os.path.join("logos", "C:\David\TFG\Fotos TFG\logoUPV.png")
LOGO_ETSII_PATH = os.path.join("logos", "C:\David\TFG\Fotos TFG\logoETSII.png")

# Ruta del modelo YOLO entrenado (best.pt)
APP_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = APP_DIR/"weights"/"best.pt"

# Ancho para mostrar imágenes
DISPLAY_WIDTH = 700

# ------------------ ENCABEZADO ------------------
col_title, col_logos = st.columns([6, 2])

with col_title:
    st.title("Detección de Defectos en Paneles Fotovoltaicos (YOLOv8)")
    st.caption("TFG David Andrei Simion · Universidad Politécnica de Valencia (UPV) · ETSII")

def show_logo(path, width=120):
    if os.path.exists(path):
        try:
            st.image(path, width=width)
        except Exception:
            pass

with col_logos:
    # Logos
    show_logo(LOGO_UPV_PATH, width=130)
    show_logo(LOGO_ETSII_PATH, width=130)

st.markdown("---")

# ------------------ CARGA DEL MODELO ------------------
@st.cache_resource
def load_model(weights_path: str):
    return YOLO(weights_path)

try:
    model = load_model(WEIGHTS_PATH)
except Exception as e:
    st.error(f"❌ Error al cargar el modelo YOLO. Verifica la ruta de pesos.\n\n{e}")
    st.stop()

# ------------------ DESCRIPCIÓN + SLIDER DE CONFIANZA ------------------
st.write(
    "Sube imágenes JPG/PNG (una o varias) o un .zip con varias imágenes. "
)

st.markdown("""
**Clases que detecta el modelo:**

- `crack`: grieta/fisura en el panel
- `dust`: suciedad/polvo sobre el panel
- `no_faulty`: panel sin defectos
""")

conf_threshold = st.slider(
    "Umbral de confianza",
    min_value=0.0, max_value=1.0, value=0.50, step=0.05,
    help="Confianza mínima para mostrar una detección."
)
st.caption("Sugerencia inicial de confianza: 0.25–0.35.")

# ------------------ ENTRADAS DE ARCHIVOS ------------------
uploaded_files = st.file_uploader(
    "Subir imágenes o archivo .zip",
    type=["jpg", "png", "jpeg", "zip"],
    accept_multiple_files=True
)


# ------------------ RECOLECCIÓN DE IMÁGENES ------------------
images_to_process = []
source_labels = []

# Cargar imágenes desde el uploader
if uploaded_files:
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        if filename.lower().endswith(".zip"):
            try:
                data = uploaded_file.read()
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for inner_name in zf.namelist():
                        if inner_name.lower().endswith((".png", ".jpg", ".jpeg")):
                            file_bytes = zf.read(inner_name)
                            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                            images_to_process.append(img)
                            source_labels.append(f"{filename} → {os.path.basename(inner_name)}")
            except zipfile.BadZipFile:
                st.error(f"❌ El archivo **{filename}** no es un ZIP válido.")
            except Exception as e:
                st.error(f"❌ No se pudo procesar **{filename}**: {e}")
        else:
            try:
                img = Image.open(uploaded_file).convert("RGB")
                images_to_process.append(img)
                source_labels.append(filename)
            except Exception as e:
                st.error(f"❌ No se pudo abrir **{filename}**: {e}")


# ------------------ INFERENCIA Y VISUALIZACIÓN ------------------
if images_to_process:
    st.subheader("Resultados de la detección")
    with st.spinner("Ejecutando inferencia…"):
        # YOLOv8 acepta listas de PIL Images; se aplica conf desde el slider
        results_list = model.predict(source=images_to_process, conf=conf_threshold)

    for idx, res in enumerate(results_list):
        # Imagen anotada (numpy). result.plot() dibuja cajas y etiquetas
        annotated = res.plot()

        # Invertir canales BGR->RGB:
         #if isinstance(annotated, np.ndarray) and annotated.ndim == 3 and annotated.shape[2] == 3:
             #annotated = annotated[:, :, ::-1]

        detections = len(res.boxes) if hasattr(res, "boxes") else 0
        caption = f"Imagen: {source_labels[idx]} · Detecciones: {detections}"

        # Mostrar todas las imágenes con el MISMO ancho (fijo en 700 px)
        st.image(annotated, caption=caption, width=DISPLAY_WIDTH)
else:
    st.info("📂 Carga imágenes (o un .zip / carpeta) para analizarlas aquí.")

# ------------------ FOOTER ------------------
st.markdown("---")

st.caption("© 2025 · TFG · UPV · ETSII — App de demostración con Streamlit y YOLOv8")






