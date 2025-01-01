import streamlit as st
import numpy as np
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

# Simulasi scaler dan model (untuk contoh, sesuaikan dengan model yang telah dilatih)
scaler = MinMaxScaler()
best_model = RandomForestClassifier(random_state=42)

# Data dummy untuk melatih model (gunakan dataset asli pada implementasi)
dummy_data = np.random.rand(100, 9)
dummy_labels = np.random.randint(0, 2, 100)
scaler.fit(dummy_data)
best_model.fit(scaler.transform(dummy_data), dummy_labels)

# Menghitung akurasi model dengan data dummy
accuracy = accuracy_score(dummy_labels, best_model.predict(scaler.transform(dummy_data)))

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
