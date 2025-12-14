import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 1. KONFIGURASI HALAMAN
st.set_page_config(page_title="ECG Heartbeat Classifier", page_icon="🫀")

# 2. FUNGSI LOAD ASSETS (Model & Scaler)
@st.cache_resource
def load_assets():
    # Memuat model CNN (.h5) dan Scaler (.pkl)
    model = load_model('model_cnn_final.h5')
    scaler = joblib.load('scaler_final.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    assets_loaded = True
except Exception as e:
    st.error(f"Gagal memuat model atau scaler: {e}")
    assets_loaded = False

# 3. MAPPING KELAS (Berdasarkan Paper PhysioNet & DAMI)
class_mapping = {
    0: "Normal",
    1: "R-on-T Premature Ventricular Contraction (SVEB)",
    2: "Premature Ventricular Contraction (VEB)",
    3: "Supraventricular Premature Beat (Fusion Beat)"
}
# Catatan: Kelas 5 (Unclassified) diabaikan sesuai keputusan pengembangan

# 4. ANTARMUKA PENGGUNA (UI)
st.title("🫀 ECG Heartbeat Classification")
st.markdown("""
Aplikasi ini mendeteksi anomali detak jantung menggunakan model **Convolutional Neural Network (CNN)**. 
Data berasal dari rekaman *chf07* pada *BIDMC Congestive Heart Failure Database*.
""")

st.sidebar.header("Opsi Input")
input_mode = st.sidebar.radio("Pilih Cara Input:", ["Unggah CSV", "Input Manual (Contoh)"])

if assets_loaded:
    input_data = None

    if input_mode == "Unggah CSV":
        uploaded_file = st.file_uploader("Unggah file CSV (140 kolom sinyal)", type="csv")
        if uploaded_file:
            input_data = pd.read_csv(uploaded_file)
    else:
        st.info("Fitur input manual dapat ditambahkan dengan data dummy dari X_test.")

    if input_data is not None:
        st.subheader("📊 Visualisasi Sinyal ECG")
        
        # Ambil baris pertama sebagai contoh jika banyak data
        sample_signal = input_data.iloc[0].values
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sample_signal, color='#1f77b4', linewidth=2)
        ax.set_title("Bentuk Gelombang Detak Jantung")
        ax.set_xlabel("Titik Waktu (Timesteps)")
        ax.set_ylabel("Amplitudo")
        st.pyplot(fig)

        # 5. PREPROCESSING & PREDIKSI
        if st.button("Klasifikasikan Detak Jantung"):
            # Scaling menggunakan scaler asli rekaman chf07
            scaled_data = scaler.transform(input_data)
            
            # Reshape untuk input CNN (Samples, 140, 1)
            reshaped_data = scaled_data.reshape(scaled_data.shape[0], 140, 1)
            
            # Prediksi
            predictions = model.predict(reshaped_data)
            pred_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions) * 100

            # Tampilkan Hasil
            st.divider()
            hasil = class_mapping.get(pred_class, "Kelas Tidak Dikenal")
            st.subheader(f"Hasil Prediksi: **{hasil}**")
            st.write(f"Tingkat Kepercayaan Model: **{confidence:.2f}%**")
            
            if pred_class == 0:
                st.success("Kondisi Jantung Terdeteksi Normal.")
            else:
                st.warning("Peringatan: Terdeteksi Anomali pada Sinyal Jantung.")

st.sidebar.markdown("---")
st.sidebar.write("Sumber Data: [PhysioNet](https://physionet.org/)")