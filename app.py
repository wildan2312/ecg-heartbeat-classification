import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 1. KONFIGURASI HALAMAN
st.set_page_config(page_title="ECG Heartbeat Classifier", page_icon="🫀")

# 2. FUNGSI LOAD ASSETS
@st.cache_resource
def load_assets():
    model = load_model('model_cnn_final.h5')
    scaler = joblib.load('scaler_final.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    assets_loaded = True
except Exception as e:
    st.error(f"Gagal memuat model atau scaler: {e}")
    assets_loaded = False

# 3. MAPPING KELAS
class_mapping = {
    0: "Normal",
    1: "R-on-T Premature Ventricular Contraction (SVEB)",
    2: "Premature Ventricular Contraction (VEB)",
    3: "Supraventricular Premature Beat (Fusion Beat)"
}

# 4. ANTARMUKA PENGGUNA (UI)
st.title("🫀 ECG Heartbeat Classification")
st.markdown("Aplikasi ini mendeteksi anomali detak jantung menggunakan model CNN.")

st.sidebar.header("Opsi Input")
input_mode = st.sidebar.radio("Pilih Cara Input:", ["Unggah CSV", "Input Manual"])

if assets_loaded:
    input_data = None

    if input_mode == "Unggah CSV":
        uploaded_file = st.file_uploader("Unggah file CSV (140 kolom sinyal)", type="csv")
        if uploaded_file:
            input_data = pd.read_csv(uploaded_file)
    else:
        st.info("Silakan unggah file CSV untuk memulai.")

    if input_data is not None:
        # --- PRE-PROCESSING KOLOM ---
        numeric_data = input_data.select_dtypes(include=[np.number])
        if 'target' in numeric_data.columns:
            signal_features = numeric_data.drop(columns=['target'])
        else:
            signal_features = numeric_data.iloc[:, :140]
        
        signal_features = signal_features.iloc[:, :140]

        # --- FITUR INPUT NOMOR SAMPEL (DINAMIS) ---
        st.sidebar.divider()
        st.sidebar.subheader("🔍 Navigasi Sampel")
        
        max_idx = len(signal_features) - 1
        
        # Menggunakan number_input agar bisa diketik langsung
        selected_index = st.sidebar.number_input(
            f"Masukkan Nomor Sampel (0 - {max_idx}):", 
            min_value=0, 
            max_value=max_idx, 
            value=0,
            step=1
        )
        
        # Tombol acak tetap ada jika ingin eksplorasi
        if st.sidebar.button("🎲 Pilih Acak"):
            selected_index = np.random.randint(0, max_idx)
            st.rerun() # Refresh halaman dengan index baru

        idx = selected_index
        # ------------------------------------------

        st.subheader(f"📊 Visualisasi Sinyal ECG (Sampel Ke-{idx})")
        
        sample_signal = signal_features.iloc[idx].values
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sample_signal, color='#1f77b4', linewidth=2)
        
        if 'class_name' in input_data.columns:
            actual_label = input_data.iloc[idx]['class_name']
            ax.set_title(f"Sinyal Pasien | Label Asli: {actual_label}")
        else:
            ax.set_title(f"Sinyal Pasien (Indeks: {idx})")
            
        ax.set_xlabel("Titik Waktu")
        ax.set_ylabel("Amplitudo")
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)

        # 5. PREDIKSI
        if st.button(f"Klasifikasikan Sampel Ke-{idx}"):
            try:
                single_sample = signal_features.iloc[[idx]] 
                scaled_data = scaler.transform(single_sample)
                reshaped_data = scaled_data.reshape(1, 140, 1)
                
                predictions = model.predict(reshaped_data)
                pred_class = np.argmax(predictions, axis=1)[0]
                confidence = np.max(predictions) * 100

                st.divider()
                hasil = class_mapping.get(pred_class, "Kelas Tidak Dikenal")
                
                c1, c2 = st.columns(2)
                c1.metric("Hasil Prediksi", hasil)
                c2.metric("Kepercayaan", f"{confidence:.2f}%")
                
                if pred_class == 0:
                    st.success("Kondisi Jantung Terdeteksi Normal.")
                else:
                    st.warning("Peringatan: Terdeteksi Pola Anomali.")
            except Exception as e:
                st.error(f"Error: {e}")

st.sidebar.markdown("---")
st.sidebar.write("Sumber Data: [PhysioNet](https://physionet.org/)")
