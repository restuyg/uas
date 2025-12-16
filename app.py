import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==============================
# Konfigurasi Halaman
# ==============================
st.set_page_config(
    page_title="Prediksi Daerah Rawan / Aman",
    layout="centered"
)

st.title("Prediksi Daerah Rawan / Aman")
st.write("Pilih **Kabupaten/Kota**, sistem akan menampilkan hasil prediksi.")

# ==============================
# Load Model & Scaler
# ==============================
model = pickle.load(open("model.sav", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))

# ==============================
# Load Dataset
# ==============================
df = pd.read_csv("datasetzzz.csv", sep=";")

# Ganti '-' menjadi 0
df = df.replace("-", 0)

# Ubah semua kolom numerik ke integer
for col in df.columns:
    if col != "Kabupaten/Kota":
        df[col] = df[col].astype(int)

# ==============================
# Input User
# ==============================
kabupaten = st.selectbox(
    "Pilih Kabupaten / Kota",
    sorted(df["Kabupaten/Kota "].unique())
)

# ==============================
# Prediksi
# ==============================
if st.button("Prediksi"):
    # Ambil data kabupaten terpilih
    data_kab = df[df["Kabupaten/Kota"] == kabupaten]

    # Ambil fitur (kecuali nama kabupaten)
    X = data_kab.drop(columns=["Kabupaten/Kota"])

    # Scaling
    X_scaled = scaler.transform(X)

    # Prediksi
    hasil = model.predict(X_scaled)[0]

    # Output
    st.subheader("Hasil Prediksi")
    if hasil == 1:
        st.success("✅ Daerah **AMAN**")
    else:
        st.error("⚠️ Daerah **RAWAN**")
