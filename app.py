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
st.write("Pilih Kabupaten/Kota untuk melihat hasil prediksi dan indeks kriminal.")

# ==============================
# Load Model & Scaler
# ==============================
model = pickle.load(open("model.sav", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))

# ==============================
# Load Dataset
# ==============================
df = pd.read_csv("datasetzzz.csv", sep=";")

# Bersihkan data
df = df.replace("-", np.nan)
for col in df.columns:
    if col != "Kabupaten/Kota ":
        df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.fillna(0)

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
    # Ambil data kabupaten
    data_kab = df[df["Kabupaten/Kota "] == kabupaten]

    # Ambil fitur numerik
    X = data_kab.drop(columns=["Kabupaten/Kota "])

    # ==============================
    # Hitung Indeks Kriminal
    # ==============================
    indeks_kriminal = X.mean(axis=1).values[0]

    # Scaling & Prediksi
    X_scaled = scaler.transform(X)
    hasil = model.predict(X_scaled)[0]

    # ==============================
    # Output Teks
    # ==============================
    st.subheader("Hasil Prediksi")

    st.metric(
        label="Indeks Kriminal",
        value=f"{indeks_kriminal:.2f}"
    )

    if hasil == 1:
        st.success("✅ Daerah **AMAN**")
        st.caption(
            "Indeks kriminal relatif rendah sehingga daerah diklasifikasikan sebagai aman."
        )
    else:
        st.error("⚠️ Daerah **RAWAN**")
        st.caption(
            "Indeks kriminal relatif tinggi sehingga daerah diklasifikasikan sebagai rawan."
        )

    # ==============================
    # Grafik Indikator Kejahatan
    # ==============================
    st.subheader("Grafik Indikator Kriminalitas")

    grafik_df = X.T
    grafik_df.columns = ["Jumlah Kasus"]

    st.bar_chart(grafik_df)
