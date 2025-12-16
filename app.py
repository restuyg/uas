import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ==============================
# Konfigurasi Halaman
# ==============================
st.set_page_config(
    page_title="Prediksi Daerah Rawan/Aman",
    layout="centered"
)

st.title("Prediksi Daerah Rawan / Aman")
st.write("Pilih **Kabupaten/Kota**, lalu sistem akan menampilkan hasil prediksi.")

# ==============================
# Load Model & Scaler
# ==============================
model = pickle.load(open("model.sav", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))

# ==============================
# Load Data Kabupaten
# ==============================
df = pd.read_csv("datasetzzz.csv")

# ==============================
# Input User
# ==============================
kabupaten = st.selectbox(
    "Pilih Kabupaten / Kota",
    sorted(df["kabupaten"].unique())
)

# ==============================
# Prediksi
# ==============================
if st.button("Prediksi"):
    # Ambil nilai numerik kabupaten
    nilai = df[df["kabupaten"] == kabupaten]["nilai"].values[0]

    # Ubah ke array 2D
    X = np.array([[nilai]])

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
