import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set konfigurasi halaman
st.set_page_config(page_title="Prediksi Kelayakan Air Minum", layout="centered")

# Judul aplikasi
st.title("Prediksi Kelayakan Air Minum")

# Input untuk setiap fitur
st.write("Masukkan nilai untuk masing-masing fitur di bawah ini, lalu klik tombol prediksi untuk mengetahui apakah air layak diminum.")
ph = st.number_input("pH (0-14)", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
Hardness = st.number_input("Hardness (mg/L)", min_value=0.0, value=200.0, step=0.1)
Solids = st.number_input("Solids (ppm)", min_value=0.0, value=10000.0, step=0.1)
Chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, value=4.0, step=0.1)
Sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=250.0, step=0.1)
Conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, value=500.0, step=0.1)
Organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, value=10.0, step=0.1)
Trihalomethanes = st.number_input("Trihalomethanes (ppb)", min_value=0.0, value=80.0, step=0.1)
Turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=5.0, step=0.1)

# Membaca data dari file
file_path = 'water_potability.csv'  # Ganti dengan path dataset Anda
try:
    data = pd.read_csv(file_path)
    X = data[["ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]]
    y = data["Potability"]

    # Membagi data menjadi pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalisasi data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Melatih model
    best_model = RandomForestClassifier(random_state=42)
    best_model.fit(X_train_scaled, y_train)

    # Evaluasi model
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    # Tombol untuk prediksi
    if st.button("Prediksi Kelayakan Air"):
        # Membentuk array input pengguna
        user_input = np.array([[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])
        user_input_scaled = scaler.transform(user_input)

        # Prediksi menggunakan model
        prediction = best_model.predict(user_input_scaled)

        # Menampilkan hasil prediksi
        if prediction[0] == 1:
            st.success("Air ini layak untuk diminum.")
        else:
            st.error("Air ini tidak layak untuk diminum.")

    # Menampilkan akurasi model saat ini
    st.write("### Akurasi Model Random Forest Saat Ini:")
    st.write(f"{accuracy * 100:.2f}%")

except FileNotFoundError:
    st.error(f"Dataset tidak ditemukan di lokasi: {file_path}. Pastikan file tersedia.")
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
