import streamlit as st
import numpy as np
import pandas as pd # Butuh pandas buat grafik
import tensorflow as tf
import joblib
import random

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS
# ==========================================
st.set_page_config(
    page_title="AQI Forecaster",
    page_icon="☁️",
    layout="centered"
)

# CSS HACK: Memberikan background gradient halus & merapikan padding
st.markdown("""
    <style>
    /* Background Gradient Halus */
    .stApp {
        background: linear-gradient(to bottom right, #ffffff, #e6f7ff);
    }
    
    /* Mempercantik Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #007bff;
    }
    
    /* Kotak Identitas */
    .student-card {
        padding: 15px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 5px solid #4e54c8;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD RESOURCES
# ==========================================
@st.cache_resource
def load_resources():
    try:
        model = tf.keras.models.load_model('model_aqi_7days.h5', compile=False)
        scaler = joblib.load('scaler_aqi_7days.save')
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_resources()

# ==========================================
# 3. SIDEBAR (IDENTITAS & LOGO)
# ==========================================
with st.sidebar:
    st.title("🤖 AI Control Panel")
    
    # Identitas Mahasiswa (Penting buat Presentasi)
    st.markdown("""
    <div class="student-card">
        <b>Dibuat Oleh:</b><br>
        Nama Mahasiswa<br>
        <span style="color:gray; font-size:0.9em;">NIM: 123456789</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("**Tentang Model:**\n\nModel ini menggunakan arsitektur **LSTM (Long Short-Term Memory)** dengan *Window Size* 7 hari untuk memprediksi polusi udara.")
    
    # Tombol Reset
    if st.button("🔄 Reset Aplikasi"):
        st.session_state.clear()
        st.rerun()

# ==========================================
# 4. HEADER UTAMA
# ==========================================
st.title("☁️ Prediksi Kualitas Udara")
st.markdown("Sistem peramalan **Air Quality Index (AQI)** berbasis *Deep Learning*.")

# Cek Model
if model is None:
    st.error("⚠️ File Model tidak ditemukan! Upload model_aqi_7days.h5 ke GitHub/Folder Anda.")
    st.stop()

# ==========================================
# 5. AREA INPUT & GRAFIK VISUALISASI
# ==========================================
col_input, col_chart = st.columns([1, 1.5])

# Inisialisasi Session State
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

with col_input:
    st.subheader("1️⃣ Input Data")
    st.caption("Masukkan 7 data historis AQI:")
    
    # Tombol dadu kecil di atas text area
    if st.button("🎲 Isi Random (Demo)"):
        vals = [random.randint(50, 120) for _ in range(7)]
        # Buat pola naik turun biar grafik bagus
        vals = sorted(vals) if random.random() > 0.5 else vals
        st.session_state.input_text = ", ".join(map(str, vals))
        st.rerun()

    input_str = st.text_area(
        "Format: angka, angka, ...",
        value=st.session_state.input_text,
        height=150,
        placeholder="80, 85, 90, 88, 92, 95, 100"
    )

with col_chart:
    st.subheader("2️⃣ Tren Historis")
    # Menampilkan grafik langsung jika ada input
    if input_str:
        try:
            data_list = [float(x.strip()) for x in input_str.split(',')]
            if len(data_list) > 0:
                # Membuat DataFrame untuk grafik cantik
                chart_data = pd.DataFrame(data_list, columns=["Nilai AQI"])
                st.line_chart(chart_data, color="#4e54c8")
        except:
            st.warning("Menunggu input angka valid...")
    else:
        st.info("👈 Masukkan data untuk melihat grafik tren.")

# ==========================================
# 6. TOMBOL EKSEKUSI
# =================================
