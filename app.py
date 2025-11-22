import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import time
import os

# Vérifier si cv2 est disponible
try:
    import cv2
    CV2_VERSION = cv2.__version__
    st.success(f"✅ OpenCV {CV2_VERSION} installé avec succès !")
except ImportError as e:
    st.error("❌ OpenCV n'est pas installé correctement")
    
# ------------------------------
# Load YOLOv9 model
# ------------------------------
@st.cache_resource
def load_model():
    return YOLO("principal/best.pt")

model = load_model()

# ------------------------------------------------
# Page configuration
# ------------------------------------------------
st.set_page_config(
    page_title="YOLOv9 Waste Detection",
    layout="wide",
    page_icon="🗑️"
)

# CSS custom for clean UI
st.markdown("""
    <style>
    .uploadedFile { display: none }
    </style>
""", unsafe_allow_html=True)

st.title("🗑️ YOLOv9 – Détection d’objets dans images & vidéos")
st.write("Déployé sur Streamlit Cloud – Interface moderne & responsive")

# ------------------------------------------------
# Sidebar settings
# ------------------------------------------------
st.sidebar.header("⚙️ Paramètres du modèle")

conf_threshold = st.sidebar.slider(
    "Seuil de confiance", 0.1, 1.0, 0.5
)

img_size = st.sidebar.slider(
    "Taille d'inférence (imgsz)", 320, 1280, 640, step=64
)

max_frames = st.sidebar.slider(
    "Nombre de frames pour la détection vidéo",
    10, 200, 60
)

# Historique en session
if 'history' not in st.session_state:
    st.session_state.history = []

# ------------------------------------------------
# Upload Files
# ------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Importer une image ou une vidéo",
    type=["jpg", "jpeg", "png", "mp4", "avi"]
)

# ------------------------------------------------
# Function for image inference
# ------------------------------------------------
def process_image(image):
    results = model.predict(
        image,
        conf=conf_threshold,
        imgsz=img_size
    )[0]

    annotated = results.plot()
    return annotated, results

# ------------------------------------------------
# Function for video inference
# ------------------------------------------------
def process_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)

    frame_count = 0
    results_summary = {}

    stframe = st.empty()

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        res = model.predict(
            frame,
            conf=conf_threshold,
            imgsz=img_size
        )[0]

        # Count detections
        for box in res.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            results_summary[label] = results_summary.get(label, 0) + 1

        annotated = res.plot()

        stframe.image(annotated, channels="BGR")
        frame_count += 1

    cap.release()
    return results_summary

# ------------------------------------------------
# Processing section
# ------------------------------------------------
if uploaded_file:

    file_type = uploaded_file.type

    st.subheader("📎 Résultat")

    if "image" in file_type:
        # Process image
        img = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(img)

        annotated_img, results = process_image(img_array)

        st.image(annotated_img, caption="Détection YOLOv9", use_column_width=True)

        # Add to history
        st.session_state.history.append(uploaded_file.name)

        st.write("### Résumé des détections :")
        for box in results.boxes:
            label = model.names[int(box.cls[0])]
            score = float(box.conf[0])
            st.write(f"- `{label}` ({score:.2f})")

    elif "video" in file_type:

        st.info("Traitement vidéo en cours… ⏳")
        summary = process_video(uploaded_file)

        # Add to history
        st.session_state.history.append(uploaded_file.name)

        st.success("Vidéo traitée 🎉")
        st.write("### Résumé des objets détectés :")
        if summary:
            for label, count in summary.items():
                st.write(f"- `{label}` : {count}")
        else:
            st.write("Aucun objet détecté.")

# ------------------------------------------------
# History
# ------------------------------------------------
st.markdown("---")
st.subheader("📂 Historique des fichiers traités")

if len(st.session_state.history) == 0:
    st.write("Aucun fichier pour le moment.")
else:
    for item in st.session_state.history[-10:][::-1]:
        st.write(f"- {item}")
