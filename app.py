import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import random

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="AQI Prediksi",
    page_icon="☁️",
    layout="centered" # Kita pakai layout tengah biar fokus
)

# CSS Custom: Background & Desain Kartu
st.markdown("""
    <style>
    /* Background Gradient Halus */
    .stApp {
        background: linear-gradient(to bottom, #f8f9fa, #e9ecef);
    }
    
    /* Mempercantik Angka Hasil (Metrics) */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: bold;
        color: #007bff;
    }
    
    /* Kartu Identitas Sidebar */
    .student-card {
        padding: 15px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-left: 5px solid #007bff;
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
        model = tf.keras.models.load_model('model_aqi_7days.h5', compile=False)
        scaler = joblib.load('scaler_aqi_7days.save')
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_resources()

# ==========================================
# 3. SIDEBAR (IDENTITAS)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910202.png", width=80)
    st.title("Panel Kontrol")
    
    # --- AREA IDENTITAS ---
    st.markdown("""
    <div class="student-card">
        <b>Dibuat Oleh:</b><br>
        👨‍🎓 Maulida Nabila<br>
        🆔 NIM: 220401061<br>
        🏫 Teknik Informatika
    </div>
    """, unsafe_allow_html=True)
    
    st.info("Aplikasi ini menggunakan model **LSTM** untuk memprediksi Indeks Kualitas Udara (AQI) berdasarkan tren 7 hari terakhir.")
    
    if st.button("🔄 Reset Form"):
        st.session_state.clear()
        st.rerun()

# ==========================================
# 4. AREA UTAMA
# ==========================================
st.title("☁️ Prediksi Kualitas Udara")
st.write("Masukkan data pemantauan udara 7 hari terakhir untuk mengetahui prediksi besok.")

if model is None:
    st.error("⚠️ File Model tidak ditemukan! Pastikan file .h5 dan .save sudah diupload.")
    st.stop()

# --- INPUT SECTION ---
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

# Tombol isi otomatis (Biar gampang demo)
col_label, col_rand = st.columns([3, 1])
with col_label:
    st.subheader("📋 Input Data Historis")
with col_rand:
    st.write("") # Spacer
    if st.button("🎲 Isi Demo"):
        # Generate angka acak
        vals = [random.randint(60, 110) for _ in range(7)]
        st.session_state.input_text = ", ".join(map(str, vals))
        st.rerun()

input_str = st.text_area(
    "Masukkan 7 nilai AQI dipisahkan koma:",
    value=st.session_state.input_text,
    height=100,
    placeholder="Contoh: 80, 85, 90, 88, 92, 95, 100"
)

# Tombol Prediksi
analyze_btn = st.button("🚀 PREDIKSI SEKARANG", type="primary", use_container_width=True)

# ==========================================
# 5. LOGIKA PREDIKSI & HASIL
# ==========================================
if analyze_btn:
    if not input_str:
        st.warning("⚠️ Mohon isi data terlebih dahulu.")
    else:
        try:
            # Parsing Data
            data_list = [float(x.strip()) for x in input_str.split(',')]
            
            # Validasi Jumlah Data
            if len(data_list) != 7:
                st.error(f"❌ Data harus berjumlah 7 hari! (Anda memasukkan {len(data_list)} data)")
            else:
                with st.spinner('Sedang menghitung...'):
                    # Preprocessing
                    features = np.array(data_list).reshape(-1, 1)
                    scaled_features = scaler.transform(features)
                    final_input = np.reshape(scaled_features, (1, 7, 1))
                    
                    # Prediksi
                    prediction = model.predict(final_input)
                    result = scaler.inverse_transform(prediction)
                    final_aqi = float(result[0][0])
                    
                    # Hitung Delta (Selisih)
                    last_aqi = data_list[-1]
                    delta = final_aqi - last_aqi

                # TAMPILKAN HASIL
                st.write("---")
                st.success("✅ Analisis Selesai!")
                
                # Menggunakan Container biar rapi
                with st.container():
                    c1, c2, c3 = st.columns(3)
                    
                    with c1:
                        st.metric("Hari Ini", f"{last_aqi:.0f}")
                        
                    with c2:
                        # Delta color otomatis hijau jika turun, merah jika naik (inverse)
                        # Tapi default streamlit: Hijau = Naik. Jadi kita biarkan default saja untuk menunjukkan kenaikan angka.
                        st.metric("Prediksi Besok", f"{final_aqi:.2f}", delta=f"{delta:.2f}")
                        
                    with c3:
                        # Logic Status & Warna
                        if final_aqi <= 50:
                            st.success("**Status: BAIK** 🍃")
                        elif final_aqi <= 100:
                            st.warning("**Status: SEDANG** 🙂")
                        elif final_aqi <= 150:
                            st.warning("**Status: TIDAK SEHAT** 😷")
                        else:
                            st.error("**Status: BERBAHAYA** ☠️")


        except ValueError:
            st.error("❌ Format Salah: Pastikan input hanya berupa angka dan koma.")


