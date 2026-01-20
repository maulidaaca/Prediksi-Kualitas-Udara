import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import os
import random

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Prediksi AQI (LSTM)",
    page_icon="🌤️",
    layout="centered"
)

# CSS Custom untuk mempercantik tampilan (Opsional)
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .student-info {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        text-align: center;
        border: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODEL & SCALER (CACHE)
# ==========================================
# @st.cache_resource agar model tidak di-load ulang setiap kali klik tombol
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
# 3. SIDEBAR (IDENTITAS MAHASISWA)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.title("About Project")
    st.info("""
    **Metode:** LSTM (Deep Learning)
    **Window Size:** 7 Hari
    **Framework:** TensorFlow & Streamlit
    """)
    
    st.divider()
    st.write("👨‍🎓 **Identitas Mahasiswa**")
    st.success("""
    **Nama:** Maulida Nabila Murtasya
    **NIM:** 220401061
    """)

# ==========================================
# 4. TAMPILAN UTAMA
# ==========================================
st.title("🌤️ Prediksi Kualitas Udara (AQI)")
st.write("Sistem peramalan kualitas udara berbasis Time Series menggunakan algoritma Long Short-Term Memory (LSTM).")

# Cek Model
if model is None:
    st.error("❌ Gagal memuat Model atau Scaler! Pastikan file .h5 dan .save ada di folder yang sama.")
    st.stop()

# --- INPUT SECTION ---
st.write("---")
st.subheader("📋 Input Data Historis")
st.caption("Masukkan 7 nilai AQI dari 7 hari terakhir secara berurutan.")

# Inisialisasi Session State untuk Auto Fill
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

col1, col2 = st.columns([3, 1])

with col2:
    st.write("") # Spacer
    st.write("") 
    if st.button("🎲 Isi Otomatis"):
        # Generate 7 angka acak
        dummy = []
        val = random.randint(60, 100)
        for _ in range(7):
            val += random.randint(-15, 20)
            if val < 10: val = 20
            dummy.append(str(val))
        st.session_state.input_text = ", ".join(dummy)
        st.rerun() # Refresh halaman

with col1:
    input_str = st.text_area(
        "Data AQI (Pisahkan dengan koma)", 
        value=st.session_state.input_text,
        placeholder="Contoh: 85, 90, 95, 110, 105, 100, 98",
        height=100
    )

# --- PREDICT BUTTON ---
if st.button("🚀 PREDIKSI BESOK", type="primary"):
    if not input_str:
        st.warning("⚠️ Harap masukkan data terlebih dahulu!")
    else:
        try:
            # 1. Parsing Data
            data_list = [float(x.strip()) for x in input_str.split(',')]
            
            # 2. Validasi Jumlah Data
            if len(data_list) != 7:
                st.error(f"❌ Data harus berjumlah 7! (Anda memasukkan {len(data_list)} data)")
            else:
                with st.spinner('Sedang menganalisis pola udara...'):
                    # 3. Preprocessing
                    features = np.array(data_list).reshape(-1, 1)
                    scaled_features = scaler.transform(features)
                    final_input = np.reshape(scaled_features, (1, 7, 1))
                    
                    # 4. Prediksi
                    prediction = model.predict(final_input)
                    result = scaler.inverse_transform(prediction)
                    final_aqi = float(result[0][0])
                
                # 5. Tampilkan Hasil
                st.success("✅ Prediksi Selesai!")
                
                # Layout Hasil
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    st.metric(label="Prediksi AQI Besok", value=f"{final_aqi:.2f}")
                
                with res_col2:
                    # Logika Status
                    if final_aqi <= 50:
                        st.balloons()
                        st.info(f"**Status: BAIK (Good)** 🍃\n\nUdara segar, aman untuk aktivitas luar ruangan.")
                    elif final_aqi <= 100:
                        st.warning(f"**Status: SEDANG (Moderate)** 🙂\n\nUdara cukup aman, namun sensitif bagi sebagian orang.")
                    elif final_aqi <= 150:
                        st.warning(f"**Status: TIDAK SEHAT (Unhealthy)** 😷\n\nKurangi aktivitas berat di luar ruangan.")
                    else:
                        st.error(f"**Status: BERBAHAYA (Hazardous)** ☠️\n\nHindari keluar rumah atau gunakan masker medis!")

        except ValueError:
            st.error("❌ Input error! Pastikan hanya memasukkan angka dan koma.")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

# Footer
st.write("---")
st.caption("© 2026 Deep Learning - Universitas Muhammadiyah Riau")