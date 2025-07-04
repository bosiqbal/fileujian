import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile

# === Load model YOLOv8 (bisa ganti ke model custom) ===
model = YOLO('yolov8n.pt')  # Ganti ke 'best.pt' jika kamu punya model sendiri

# === UI Streamlit ===
st.set_page_config(page_title="Deteksi Objek YOLOv8", layout="centered")
st.title("?? Deteksi Objek Menggunakan YOLOv8")
st.write("Upload gambar dan lihat hasil deteksinya secara langsung di browser!")

# === Upload Gambar ===
uploaded_file = st.file_uploader("??? Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    img = Image.open(uploaded_file)
    st.image(img, caption="?? Gambar yang Diupload", use_container_width=True)

    # Simpan gambar sementara untuk diproses oleh YOLO
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        img.save(tmp_file.name)

        # Deteksi objek
        results = model(tmp_file.name)
        res_plotted = results[0].plot()  # Gambar hasil deteksi

        # Tampilkan hasil
        st.image(res_plotted, caption="? Hasil Deteksi Objek", use_container_width=True)
