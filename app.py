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
        # 1. Identifikasi Fitur Sinyal (Hanya angka!)
        numeric_data = input_data.select_dtypes(include=[np.number])
        
        # Buat kolom target jika ada untuk keperluan filter
        target_col = 'target' if 'target' in input_data.columns else None
        
        if target_col:
            signal_features = numeric_data.drop(columns=[target_col])
        else:
            signal_features = numeric_data.iloc[:, :140]
        
        signal_features = signal_features.iloc[:, :140]

        # --- FITUR DINAMIS BARU ---
        st.sidebar.divider()
        st.sidebar.subheader("🔍 Navigasi Data")
        
        # Slider untuk memilih nomor baris (Indeks)
        max_index = len(signal_features) - 1
        selected_index = st.sidebar.slider("Pilih Nomor Sampel:", 0, max_index, 0)
        
        # Tombol untuk memilih acak
        if st.sidebar.button("🎲 Pilih Sampel Acak"):
            selected_index = np.random.randint(0, max_index)
            # Karena Streamlit merefresh page, kita gunakan session_state jika ingin index menetap
            st.session_state.selected_index = selected_index
        
        # Gunakan index dari slider atau tombol acak
        idx = selected_index
        # --------------------------

        st.subheader(f"📊 Visualisasi Sinyal ECG (Sampel Ke-{idx})")
        
        # Ambil baris berdasarkan pilihan user (Dynamic Indexing)
        sample_signal = signal_features.iloc[idx].values
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sample_signal, color='#1f77b4', linewidth=2)
        
        # Tambahkan informasi label asli jika tersedia di CSV
        if 'class_name' in input_data.columns:
            actual_label = input_data.iloc[idx]['class_name']
            ax.set_title(f"Bentuk Gelombang Detak Jantung | Label Asli: {actual_label}")
        else:
            ax.set_title(f"Bentuk Gelombang Detak Jantung (Indeks: {idx})")
            
        ax.set_xlabel("Titik Waktu (Timesteps)")
        ax.set_ylabel("Amplitudo")
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)

        # 5. PREPROCESSING & PREDIKSI
        if st.button(f"Klasifikasikan Sampel Ke-{idx}"):
            try:
                # Ambil hanya satu baris yang dipilih untuk diprediksi
                single_sample = signal_features.iloc[[idx]] 
                scaled_data = scaler.transform(single_sample)
                reshaped_data = scaled_data.reshape(1, 140, 1)
                
                # Prediksi
                predictions = model.predict(reshaped_data)
                pred_class = np.argmax(predictions, axis=1)[0]
                confidence = np.max(predictions) * 100

                # Tampilkan Hasil
                st.divider()
                hasil = class_mapping.get(pred_class, "Kelas Tidak Dikenal")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Hasil Prediksi", hasil)
                with col2:
                    st.metric("Kepercayaan Model", f"{confidence:.2f}%")
                
                if pred_class == 0:
                    st.success("Analisis: Kondisi Jantung Terdeteksi Normal.")
                else:
                    st.warning("Peringatan: Terdeteksi Pola Anomali.")
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")

st.sidebar.markdown("---")
st.sidebar.write("Sumber Data: [PhysioNet](https://physionet.org/)")

