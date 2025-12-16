import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Prediksi Daerah Rawan / Aman",
    layout="centered"
)

st.title("Prediksi Daerah Rawan / Aman")
st.write("Pilih Kabupaten/Kota untuk melihat hasil prediksi.")

# Load model & scaler
model = pickle.load(open("model.sav", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))

# Load dataset
df = pd.read_csv("datasetzzz.csv", sep=";")

# Bersihkan data
df = df.replace("-", np.nan)
for col in df.columns:
    if col != "Kabupaten/Kota ":
        df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.fillna(0)

# Input
kabupaten = st.selectbox(
    "Pilih Kabupaten / Kota",
    sorted(df["Kabupaten/Kota "].unique())
)

# Prediksi
if st.button("Prediksi"):
    X = df[df["Kabupaten/Kota "] == kabupaten].drop(columns=["Kabupaten/Kota "])
    X_scaled = scaler.transform(X)
    hasil = model.predict(X_scaled)[0]

    st.subheader("Hasil Prediksi")
    if hasil == 1:
        st.success("✅ Daerah **AMAN**")
    else:
        st.error("⚠️ Daerah **RAWAN**")
