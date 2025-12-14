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
    # Pastikan nama file sesuai dengan yang diunggah ke GitHub
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
        st.info("Silakan unggah file CSV hasil pembersihan data untuk memulai.")

    if input_data is not None:
        # --- PERBAIKAN DI SINI ---
        # 1. Hanya ambil kolom numerik (membuang kolom teks seperti 'class_name')
        numeric_data = input_data.select_dtypes(include=[np.number])
        
        # 2. Buang kolom 'target' jika ada, agar hanya tersisa 140 fitur sinyal
        if 'target' in numeric_data.columns:
            signal_features = numeric_data.drop(columns=['target'])
        else:
            signal_features = numeric_data.iloc[:, :140]

        # Pastikan kita hanya mengambil 140 kolom pertama (sesuai input model)
        signal_features = signal_features.iloc[:, :140]
        # -------------------------

        st.subheader("📊 Visualisasi Sinyal ECG")
        
        # Ambil baris pertama dari data yang sudah bersih dari teks
        sample_signal = signal_features.iloc[0].values
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sample_signal, color='#1f77b4', linewidth=2)
        ax.set_title("Bentuk Gelombang Detak Jantung (Sampel Pertama)")
        ax.set_xlabel("Titik Waktu (Timesteps)")
        ax.set_ylabel("Amplitudo")
        st.pyplot(fig)

        # 5. PREPROCESSING & PREDIKSI
        if st.button("Klasifikasikan Detak Jantung"):
            try:
                # Scaling menggunakan fitur yang sudah bersih
                scaled_data = scaler.transform(signal_features)
                
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
            
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")

st.sidebar.markdown("---")
st.sidebar.write("Sumber Data: [PhysioNet](https://physionet.org/)")
