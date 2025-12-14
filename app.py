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
st.markdown("""
Aplikasi ini mendeteksi anomali detak jantung menggunakan model **CNN**. 
Data bersumber dari rekaman pasien *Congestive Heart Failure* di PhysioNet.
""")

if assets_loaded:
    # Langsung ke menu Unggah CSV
    uploaded_file = st.file_uploader("Unggah file CSV (140 kolom fitur)", type="csv")

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        
        # --- PRE-PROCESSING KOLOM (HANYA ANGKA) ---
        numeric_data = input_data.select_dtypes(include=[np.number])
        if 'target' in numeric_data.columns:
            signal_features = numeric_data.drop(columns=['target'])
        else:
            signal_features = numeric_data.iloc[:, :140]
        
        signal_features = signal_features.iloc[:, :140]

        # --- NAVIGASI SAMPEL (INPUT NOMOR) ---
        st.sidebar.header("🔍 Navigasi")
        max_idx = len(signal_features) - 1
        
        idx = st.sidebar.number_input(
            f"Masukkan Nomor Sampel (0 - {max_idx}):", 
            min_value=0, max_value=max_idx, value=0, step=1
        )
        
        if st.sidebar.button("🎲 Pilih Acak"):
            st.session_state.random_idx = np.random.randint(0, max_idx)
            st.rerun()

        # 5. VISUALISASI
        st.subheader(f"📊 Visualisasi Sinyal ECG (Indeks: {idx})")
        sample_signal = signal_features.iloc[idx].values
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sample_signal, color='#1f77b4', linewidth=2)
        
        if 'class_name' in input_data.columns:
            actual = input_data.iloc[idx]['class_name']
            ax.set_title(f"Sinyal Aktual | Label Asli: {actual}")
        
        ax.set_xlabel("Waktu (Timesteps)")
        ax.set_ylabel("Amplitudo")
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)

        # 6. PREDIKSI
        if st.button(f"Klasifikasikan Sampel Ke-{idx}"):
            try:
                single_sample = signal_features.iloc[[idx]] 
                scaled = scaler.transform(single_sample)
                reshaped = scaled.reshape(1, 140, 1)
                
                predictions = model.predict(reshaped)
                pred_class = np.argmax(predictions, axis=1)[0]
                confidence = np.max(predictions) * 100

                st.divider()
                hasil = class_mapping.get(pred_class, "Kelas Tidak Dikenal")
                
                c1, c2 = st.columns(2)
                c1.metric("Hasil Prediksi", hasil)
                c2.metric("Tingkat Kepercayaan", f"{confidence:.2f}%")
                
                if pred_class == 0:
                    st.success("Analisis: Kondisi Jantung Normal.")
                else:
                    st.warning("Peringatan: Terdeteksi Anomali Jantung.")
            except Exception as e:
                st.error(f"Error Prediksi: {e}")
    else:
        st.info("💡 Silakan unggah file CSV untuk memulai analisis.")

st.sidebar.markdown("---")
st.sidebar.write("Sumber: [PhysioNet](https://physionet.org/)")
