# 1. Instalasi Streamlit
# Buka terminal/command prompt dan jalankan:
# pip install streamlit

# 2. Buat file baru bernama 'Water Quality Analysis.py' dan masukkan kode berikut:

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Set page config
st.set_page_config(page_title="Water Quality", layout="wide")

# Title
st.title("Klasifikasi Data Kualitas Air dan Kelayakan Minum")

# Step 1: Pengumpulan Data
st.header("1. Pengumpulan Data")
file_path = 'water_potability.csv'
water_data = pd.read_csv(file_path)
st.dataframe(water_data.head())

# Step 2: Menelaah Data
st.header("2. Menelaah Data")
st.subheader("Informasi Dataset")
buffer = st.expander("Klik untuk melihat informasi dataset")
with buffer:
    st.write("Jumlah Baris:", water_data.shape[0])
    st.write("Tipe Data Tiap Kolom:")
    st.write(water_data.dtypes)
    st.write("\nNilai Unik Tiap Kolom:")
    st.write(water_data.nunique())

# Step 3: Validasi dan Visualisasi Data
st.header("3. Validasi dan Visualisasi Data")
st.subheader("Missing Values")
st.write(water_data.isnull().sum())

# Imputasi missing values
water_data.fillna(water_data.mean(), inplace=True)

# Penanganan outlier
for column in water_data.columns[:-1]:
    Q1 = water_data[column].quantile(0.25)
    Q3 = water_data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    water_data = water_data[(water_data[column] >= lower_bound) & (water_data[column] <= upper_bound)]

# Visualisasi distribusi
st.subheader("Distribusi Data")
col1, col2 = st.columns(2)  # Membuat 2 kolom untuk menampilkan grafik

with col1:
    fig_dist = plt.figure(figsize=(10, 6))
    sns.countplot(x='Potability', data=water_data, palette='viridis')
    # Visualisasi sebelum resampling
    plt.title('Distribusi Data Sebelum Resampling')
    st.pyplot(fig_dist)

with col2:
    # Menentukan Objek Data
    features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 
                'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    target = 'Potability'
    X = water_data[features]
    y = water_data[target]

    # Resampling
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    # Visualisasi setelah resampling
    fig_resample = plt.figure(figsize=(10, 6))
    sns.countplot(x=y_resampled, palette='viridis')
    plt.title('Distribusi Data Setelah Resampling')
    st.pyplot(fig_resample)

# Step 5: Membersihkan Data
st.header("5. Membersihkan Data")
# Korelasi Heatmap
st.subheader("Korelasi Heatmap")
fig_corr = plt.figure(figsize=(12, 8))
# fig_corr = plt.figure(figsize=(8, 6))
sns.heatmap(water_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
st.pyplot(fig_corr)

# Distribusi Histogram Plot dengan Bar dan Line
st.subheader("Distribusi Histogram")
cols = st.columns(3)  # Membuat 3 kolom untuk menampilkan histogram
for i, feature in enumerate(features):
    with cols[i % 3]:  # Menggunakan modulus untuk mengatur kolom
        fig_hist = plt.figure(figsize=(10, 6))
        sns.histplot(data=water_data[feature], stat='density', color='skyblue', bins=15, edgecolor='black')
        sns.kdeplot(data=water_data[feature], color='blue', linewidth=2)
        plt.title(f'Distribusi Histogram Atribut: {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frekuensi')
        st.pyplot(fig_hist)

# Step 6: Konstruksi Data (Penskalaan Data)
st.header("6-8. Modeling dan Evaluasi")

# Preprocessing
scaler = MinMaxScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

# Step 7: Pemodelan
# Split data (dengan data sudah dinormalisasi sebelumnya)
X_train, X_test, y_train, y_test = train_test_split(X_resampled_scaled, y_resampled, 
                                                    test_size=0.2, random_state=42)
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_resampled, 
                                                    y_resampled, test_size=0.2, random_state=42)

# Model definisi
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
}

# Step 8: Evaluasi Model
# Train dan evaluasi model
results_normalized = {}
results_raw = {}
conf_matrices = {}

for name, model in models.items():
    # Normalized data
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results_normalized[name] = accuracy_score(y_test, y_pred)
    conf_matrices[name] = confusion_matrix(y_test, y_pred)
    
    # Raw data
    model.fit(X_train_raw, y_train_raw)
    y_pred_raw = model.predict(X_test_raw)
    results_raw[name] = accuracy_score(y_test_raw, y_pred_raw)

# Tampilkan hasil
st.subheader("Hasil Akurasi")
col1, col2 = st.columns(2)

with col1:
    st.write("Akurasi Sebelum Normalisasi:")
    for model, acc in results_raw.items():
        st.write(f"{model}: {acc*100:.2f}%")

with col2:
    st.write("Akurasi Setelah Normalisasi:")
    for model, acc in results_normalized.items():
        st.write(f"{model}: {acc*100:.2f}%")

# Confusion Matrix
st.subheader("Confusion Matrix")
num_models = len(conf_matrices)
cols = st.columns(min(num_models, 2))  # Maksimal 2 kolom

for i, (model, matrix) in enumerate(conf_matrices.items()):
    with cols[i % 2]:  # Menggunakan modulus untuk mengatur kolom
        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)  # Menambahkan gap
        fig_conf = plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Potable', 'Potable'],
                    yticklabels=['Not Potable', 'Potable'])
        plt.title(f'Confusion Matrix for {model}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig_conf)

# 3. Cara menjalankan aplikasi:
# Buka terminal/command prompt
# Navigasi ke direktori tempat file app.py berada
# Jalankan perintah: streamlit run app.py

#add prediction :

# Menambahkan Input untuk Prediksi Potability
st.header("Prediksi Kelayakan Air Minum")
st.write("Masukkan nilai untuk masing-masing fitur di bawah ini, lalu klik tombol prediksi untuk mengetahui apakah air layak diminum.")

# Input untuk setiap fitur
ph = st.number_input("pH (0-14)", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
Hardness = st.number_input("Hardness (mg/L)", min_value=0.0, value=200.0, step=0.1)
Solids = st.number_input("Solids (ppm)", min_value=0.0, value=10000.0, step=0.1)
Chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, value=4.0, step=0.1)
Sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=250.0, step=0.1)
Conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, value=500.0, step=0.1)
Organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, value=10.0, step=0.1)
Trihalomethanes = st.number_input("Trihalomethanes (ppb)", min_value=0.0, value=80.0, step=0.1)
Turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=5.0, step=0.1)

# Tombol untuk prediksi
if st.button("Prediksi Kelayakan Air"):
    # Membentuk array input pengguna
    user_input = np.array([[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])

    # Melakukan penskalaan data
    user_input_scaled = scaler.transform(user_input)

    # Menggunakan model terbaik (misalnya Random Forest) untuk prediksi
    best_model = models['Random Forest']
    prediction = best_model.predict(user_input_scaled)

    # Menampilkan hasil prediksi
    if prediction[0] == 1:
        st.success("Air ini layak untuk diminum.")
    else:
        st.error("Air ini tidak layak untuk diminum.")


# Kesimpulan
st.header("Kesimpulan dari Hasil Analisis")
st.write("Berdasarkan hasil analisis perbandingan algoritma yang digunakan, berikut adalah kesimpulan:")

# Tingkat akurasi setiap algoritma
st.subheader("Tingkat Akurasi Setiap Algoritma")
st.write("Akurasi Sebelum Normalisasi:")
for model, acc in results_raw.items():
    st.write(f"{model}: {acc*100:.2f}%")

st.write("Akurasi Setelah Normalisasi:")
for model, acc in results_normalized.items():
    st.write(f"{model}: {acc*100:.2f}%")

# Keunggulan dan keterbatasan
st.subheader("Keunggulan dan Keterbatasan")
st.write("""
- **Random Forest**: 
  - Keunggulan: Tahan terhadap overfitting, dapat menangani data yang hilang.
  - Keterbatasan: Memerlukan lebih banyak waktu untuk pelatihan dibandingkan algoritma lain.
  
- **Decision Tree**: 
  - Keunggulan: Mudah dipahami dan diinterpretasikan.
  - Keterbatasan: Rentan terhadap overfitting jika tidak dipangkas dengan baik.
  
- **XGBoost**: 
  - Keunggulan: Sangat cepat dan efisien, sering memberikan hasil yang sangat baik.
  - Keterbatasan: Memerlukan tuning parameter yang lebih banyak.
  
- **CatBoost**: 
  - Keunggulan: Baik untuk data kategorikal, tidak memerlukan preprocessing yang banyak.
  - Keterbatasan: Memerlukan lebih banyak sumber daya komputasi.
""")

# Rekomendasi algoritma yang paling efektif
st.subheader("Rekomendasi Algoritma yang Paling Efektif")
st.write("""
Berdasarkan analisis di atas, **Random Forest** direkomendasikan sebagai algoritma yang paling efektif untuk kasus ini. 
Alasan mengapa Random Forest optimal adalah:
- Tingkat akurasi yang tinggi setelah normalisasi.
- Tahan terhadap overfitting dan dapat menangani data yang hilang dengan baik.
- Kemampuan untuk memberikan hasil yang stabil dan akurat pada berbagai jenis data.
""")