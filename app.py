import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. Konfigurasi Halaman Streamlit ---
st.set_page_config(page_title="Prediksi Daerah Rawan/Aman", layout="centered")
st.title("Sistem Prediksi Daerah Rawan/Aman Berdasarkan Kabupaten/Kota")
st.write("Aplikasi ini memprediksi status 'Rawan' atau 'Aman' suatu daerah berdasarkan nama Kabupaten/Kota yang dipilih dari data yang ada.")
st.markdown("---")

# --- 2. Load Data, Model, dan Scaler ---

# PENTING: Pastikan 'datasetzzz.csv' berada di direktori yang sama!
try:
    # Memuat Dataset Asli sebagai Lookup Table
    # Ganti 'datasetzzz.csv' jika nama file Anda berbeda
    df_lookup = pd.read_csv('datasetzzz.csv', sep=';') 
    
    # Ambil daftar unik Kabupaten/Kota untuk dropdown
    list_kabupaten = df_lookup['Kabupaten/Kota '].unique().tolist()
    
    st.sidebar.success("Dataset Referensi Berhasil Dimuat!")
    
except FileNotFoundError:
    st.error("ERROR: File 'datasetzzz.csv' tidak ditemukan.")
    st.warning("Pastikan file dataset Anda berada di direktori yang sama dengan 'app.py'.")
    st.stop() # Hentikan eksekusi jika data tidak ada
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat dataset: {e}")
    st.stop()


model = None
scaler = None

try:
    # Memuat Model KNN (model.sav) dan Scaler (scaler.sav)
    with open('model.sav', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.sav', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    st.sidebar.success("Model dan Scaler berhasil dimuat!")
except:
    st.error("ERROR: File 'model.sav' atau 'scaler.sav' tidak ditemukan/rusak.")
    st.warning("Pastikan kedua file telah dibuat dengan benar di notebook UAS.ipynb.")


# --- 3. Input Pengguna Berdasarkan Kabupaten/Kota ---
st.header("Pilih Lokasi")

if model is not None and scaler is not None:
    
    # Input utama: Dropdown menu untuk memilih Kabupaten/Kota
    selected_kabupaten = st.selectbox(
        "Pilih Kabupaten/Kota yang Akan Diprediksi:",
        options=['--- Pilih Daerah ---'] + list_kabupaten
    )
    
    # Tombol Prediksi
    if st.button("Lakukan Prediksi"):
        
        if selected_kabupaten == '--- Pilih Daerah ---':
             st.warning("Mohon pilih Kabupaten/Kota terlebih dahulu.")
             st.stop()
        
        # --- 4. Lookup Data & Preprocessing ---
        
        # 1. Ambil data fitur numerik dari baris yang dipilih
        data_row = df_lookup[df_lookup['Kabupaten/Kota'] == selected_kabupaten]
        
        # PENTING: GANTI ['Jumlah'] dengan daftar kolom fitur numerik yang Anda gunakan 
        # untuk melatih model KNN (misalnya: ['Jumlah', 'Kepadatan', 'FiturX'])
        feature_columns = ['Jumlah'] 
        
        if not all(col in data_row.columns for col in feature_columns):
            st.error(f"Kolom fitur {feature_columns} tidak ditemukan di dataset lookup.")
            st.warning("Pastikan Anda mengganti `feature_columns` di kode dengan nama fitur yang benar.")
            st.stop()

        # Ambil nilai numerik dan ubah ke format yang diterima model (2D array)
        input_features = data_row[feature_columns].values
        
        # 2. Scaling data input
        scaled_input = scaler.transform(input_features)
        
        # 3. Prediksi menggunakan model KNN
        prediction = model.predict(scaled_input)
        
        # --- 5. Tampilkan Hasil ---
        st.markdown("---")
        st.subheader(f"Hasil Klasifikasi untuk **{selected_kabupaten}**")
        
        hasil_prediksi = prediction[0]

        if hasil_prediksi == "Daerah Rawan":
            st.error(f"Prediksi: **{hasil_prediksi}** ⚠️")
            st.write("Daerah ini diklasifikasikan sebagai **Daerah Rawan**.")
        elif hasil_prediksi == "Daerah Aman":
            st.success(f"Prediksi: **{hasil_prediksi}** ✅")
            st.write("Daerah ini diklasifikasikan sebagai **Daerah Aman**.")
        else:
            st.info(f"Hasil prediksi: **{hasil_prediksi}**")
