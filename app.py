import streamlit as st
import numpy as np
import pandas as pd  # PENTING: Untuk membuat grafik
import tensorflow as tf
import joblib
import random

# ==========================================
# 1. KONFIGURASI HALAMAN & DESAIN
# ==========================================
st.set_page_config(
    page_title="AQI Forecaster Pro",
    page_icon="☁️",
    layout="wide"  # Menggunakan layout lebar agar grafik lega
)

# CSS Custom: Memberikan background gradient & kartu identitas
st.markdown("""
    <style>
    /* Background Gradient Halus (Biru Muda ke Putih) */
    .stApp {
        background: linear-gradient(to bottom, #f0f8ff, #ffffff);
    }
    
    /* Mempercantik Metrics (Angka Hasil) */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
    }
    
    /* Kartu Identitas di Sidebar */
    .student-card {
        padding: 15px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 5px solid #4e54c8;
        font-size: 14px;
    }
    
    /* Tombol Utama */
    .stButton>button {
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODEL & SCALER
# ==========================================
@st.cache_resource
def load_resources():
    try:
        # Ganti nama file sesuai file Anda
        model = tf.keras.models.load_model('model_aqi_7days.h5', compile=False)
        scaler = joblib.load('scaler_aqi_7days.save')
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_resources()

# ==========================================
# 3. SIDEBAR (IDENTITAS MAHASISWA)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2011/2011601.png", width=80)
    st.title("🎛️ Control Panel")
    
    # --- AREA IDENTITAS (Edit Bagian Ini) ---
    st.markdown("""
    <div class="student-card">
        <b>Dibuat Oleh:</b><br>
        👨‍🎓 Nama Mahasiswa<br>
        🆔 NIM: 12345678<br>
        🏫 Teknik Informatika
    </div>
    """, unsafe_allow_html=True)
    
    st.info("**Info Model:**\nLSTM (Long Short-Term Memory) dengan Window Size 7 Hari.")
    
    if st.button("🔄 Reset / Bersihkan"):
        st.session_state.clear()
        st.rerun()

# ==========================================
# 4. JUDUL UTAMA
# ==========================================
st.markdown("<h1 style='text-align: center; color: #4e54c8;'>☁️ Prediksi Kualitas Udara (AQI)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Dashboard Monitoring & Forecasting Berbasis Deep Learning</p>", unsafe_allow_html=True)
st.write("---")

# Cek apakah model berhasil di-load
if model is None:
    st.error("⚠️ **FILE HILANG:** Pastikan 'model_aqi_7days.h5' dan 'scaler_aqi_7days.save' ada di folder yang sama!")
    st.stop()

# ==========================================
# 5. AREA INPUT & VISUALISASI (LAYOUT 2 KOLOM)
# ==========================================
col_left, col_right = st.columns([1, 1.5]) # Kiri lebih kecil, Kanan (Grafik) lebih besar

# --- Session State untuk menyimpan input ---
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

with col_left:
    st.subheader("1️⃣ Input Data Historis")
    st.caption("Masukkan 7 data AQI terakhir:")
    
    # Tombol Dadu (Isi Otomatis)
    if st.button("🎲 Isi Random (Demo)"):
        # Membuat pola angka naik turun agar grafik terlihat bagus
        start = random.randint(50, 100)
        pola = [start]
        for _ in range(6):
            # Angka selanjutnya naik/turun sedikit dari angka sebelumnya
            next_val = pola[-1] + random.randint(-15, 20)
            if next_val < 10: next_val = 20
            pola.append(next_val)
            
        st.session_state.input_text = ", ".join(map(str, pola))
        st.rerun() # Refresh agar grafik langsung muncul

    # Text Area Input
    input_str = st.text_area(
        "Format: angka, angka, ...",
        value=st.session_state.input_text,
        height=150,
        placeholder="Contoh: 80, 85, 90, 88, 92, 95, 100"
    )

with col_right:
    st.subheader("2️⃣ Grafik Tren Data")
    
    # Logika Menampilkan Grafik Secara Realtime
    if input_str:
        try:
            # Mengubah string input menjadi list angka
            data_list = [float(x.strip()) for x in input_str.split(',')]
            
            if len(data_list) > 0:
                # Membuat DataFrame agar bisa dibaca st.line_chart
                chart_data = pd.DataFrame({
                    'Hari ke-': range(1, len(data_list)+1),
                    'Nilai AQI': data_list
                }).set_index('Hari ke-')
                
                # Menampilkan Grafik Garis
                st.line_chart(chart_data, color="#4e54c8")
            else:
                st.warning("Data kosong.")
        except:
            st.warning("Menunggu input angka yang valid...")
    else:
        # Tampilan kosong (Placeholder) jika belum ada input
        st.info("👈 Masukkan data atau klik 'Isi Random' untuk melihat grafik tren udara.")

# ==========================================
# 6. TOMBOL & PROSES PREDIKSI
# ==========================================
st.write("")
analyze_btn = st.button("🚀 ANALISIS & PREDIKSI MASA DEPAN", type="primary", use_container_width=True)

if analyze_btn:
    if not input_str:
        st.toast("⚠️ Data input masih kosong!", icon="❌")
    else:
        try:
            # 1. Parsing Data
            data_list = [float(x.strip()) for x in input_str.split(',')]
            
            # 2. Validasi Jumlah (Harus 7)
            if len(data_list) != 7:
                st.error(f"❌ Data harus berjumlah 7 hari! (Anda memasukkan {len(data_list)} data)")
            else:
                with st.spinner('Sedang memproses algoritma LSTM...'):
                    # 3. Preprocessing (Reshape & Scale)
                    features = np.array(data_list).reshape(-1, 1)
                    scaled_features = scaler.transform(features)
                    final_input = np.reshape(scaled_features, (1, 7, 1))
                    
                    # 4. Prediksi
                    prediction = model.predict(final_input)
                    result = scaler.inverse_transform(prediction)
                    final_aqi = float(result[0][0])
                    
                    # Hitung selisih dengan hari terakhir
                    last_aqi = data_list[-1]
                    delta = final_aqi - last_aqi

                # 5. MENAMPILKAN HASIL (METRICS)
                st.success("✅ Prediksi Selesai!")
                
                with st.container():
                    # Membagi hasil menjadi 3 kolom
                    m1, m2, m3 = st.columns(3)
                    
                    with m1:
                        st.metric("AQI Hari Ini (Terakhir)", f"{last_aqi:.0f}")
                        
                    with m2:
                        # Menampilkan Prediksi dengan panah Delta (Hijau/Merah otomatis)
                        st.metric("Prediksi Esok", f"{final_aqi:.2f}", delta=f"{delta:.2f} poin")
                        
                    with m3:
                        # Logika Status Warna-Warni
                        if final_aqi <= 50:
                            st.success("**Status: BAIK (Good)** 🍃")
                        elif final_aqi <= 100:
                            st.warning("**Status: SEDANG (Moderate)** 🙂")
                        elif final_aqi <= 150:
                            st.warning("**Status: TIDAK SEHAT** 😷")
                        else:
                            st.error("**Status: BERBAHAYA!** ☠️")

                # 6. FITUR TAMBAHAN: DETAIL TEKNIS (Expandable)
                # Dosen suka ini karena menunjukkan "Isi Jeroan" aplikasi
                with st.expander("🔍 Lihat Detail Teknis (JSON Output)"):
                    st.write("Data ini diproses langsung dari output model:")
                    st.json({
                        "input_shape": "(1, 7, 1)",
                        "raw_prediction_scaled": float(prediction[0][0]),
                        "final_aqi": final_aqi,
                        "data_historis": data_list
                    })

        except ValueError:
            st.error("❌ Format Error: Pastikan input hanya berupa angka dan koma.")
