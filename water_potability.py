# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from imblearn.over_sampling import RandomOverSampler

# # Set page config with custom theme
# st.set_page_config(
#     page_title="Water Potability Prediction",
#     page_icon="💧",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
# <style>
#     .main { padding: 2rem; }
#     .stTitle {
#         color: #2c3e50;
#         font-size: 3rem !important;
#         padding-bottom: 1rem;
#         border-bottom: 2px solid #3498db;
#         margin-bottom: 2rem;
#     }
#     .stHeader { color: #2980b9; padding-top: 1rem; }
#     .prediction-box {
#         padding: 1.5rem;
#         border-radius: 10px;
#         margin: 1rem 0;
#     }
#     .info-box {
#         background-color: #f8f9fa;
#         padding: 1.5rem;
#         border-radius: 10px;
#         border-left: 5px solid #3498db;
#         margin: 1rem 0;
#     }
#     .parameter-info {
#         background-color: #f8f9fa;
#         padding: 1.5rem;
#         border-radius: 10px;
#         margin-top: 1rem;
#     }
#     .metric-card {
#         background-color: #ffffff;
#         padding: 1rem;
#         border-radius: 10px;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#         margin: 0.5rem 0;
#     }
#     table {
#         width: 100%;
#         border-collapse: collapse;
#     }
#     th, td {
#         padding: 8px;
#         text-align: left;
#         border-bottom: 1px solid #ddd;
#     }
#     th {
#         background-color: #f8f9fa;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Title with icon
# st.markdown("# 💧 Prediksi Potabilitas Air menggunakan Random Forest")
# st.markdown("##### *Analisis kualitas air untuk menentukan kelayakan konsumsi*")

# # Load and prepare data
# @st.cache_data
# def load_data():
#     df = pd.read_csv('water_potability.csv')
#     df.fillna(df.mean(), inplace=True)
    
#     # Handle outliers
#     for column in df.columns[:-1]:
#         Q1 = df[column].quantile(0.25)
#         Q3 = df[column].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
#         df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
#     return df

# # Load the data
# with st.spinner('Memuat dan mempersiapkan data...'):
#     water_data = load_data()

# # Prepare features and target
# features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
#             'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
# X = water_data[features]
# y = water_data['Potability']

# # Resample data
# ros = RandomOverSampler(random_state=42)
# X_resampled, y_resampled = ros.fit_resample(X, y)

# # Scale the features
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X_resampled)

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

# # Train the model
# rf_model = RandomForestClassifier(random_state=42)
# rf_model.fit(X_train, y_train)

# # Calculate accuracy
# y_pred = rf_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)

# # Display model accuracy in a card layout
# st.markdown("### 📊 Performa Model")
# col1, col2, col3 = st.columns([1,2,1])
# with col1:
#     st.markdown(f"""
#     <div class="metric-card">
#         <h4 style="color: #2980b9;">Akurasi Model</h4>
#         <h2 style="color: #27ae60;">{accuracy:.2%}</h2>
#     </div>
#     """, unsafe_allow_html=True)
# with col2:
#     st.markdown("""
#     <div class="info-box">
#         <h4>Informasi Model 🔍</h4>
#         <p>Model Random Forest ini dilatih menggunakan:</p>
#         <ul>
#             <li>80% data training dan 20% data testing</li>
#             <li>Teknik oversampling untuk balance data</li>
#             <li>Normalisasi fitur menggunakan MinMaxScaler</li>
#         </ul>
#     </div>
#     """, unsafe_allow_html=True)

# # Create input form
# st.markdown("### 📝 Input Parameter Kualitas Air")
# st.markdown("Masukkan nilai parameter kualitas air yang akan dianalisis:")

# # Create three columns for input fields with better styling
# col1, col2, col3 = st.columns(3)

# with col1:
#     st.markdown('<div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px;">', unsafe_allow_html=True)
#     ph = st.number_input('pH 🧪', min_value=0.0, max_value=14.0, value=7.0)
#     hardness = st.number_input('Hardness 💎', min_value=0.0, value=200.0)
#     solids = st.number_input('Solids (TDS) 🌫️', min_value=0.0, value=20000.0)
#     st.markdown('</div>', unsafe_allow_html=True)

# with col2:
#     st.markdown('<div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px;">', unsafe_allow_html=True)
#     chloramines = st.number_input('Chloramines 🧬', min_value=0.0, value=4.0)
#     sulfate = st.number_input('Sulfate ⚗️', min_value=0.0, value=250.0)
#     conductivity = st.number_input('Conductivity ⚡', min_value=0.0, value=400.0)
#     st.markdown('</div>', unsafe_allow_html=True)

# with col3:
#     st.markdown('<div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px;">', unsafe_allow_html=True)
#     organic_carbon = st.number_input('Organic Carbon 🍃', min_value=0.0, value=10.0)
#     trihalomethanes = st.number_input('Trihalomethanes 🔬', min_value=0.0, value=50.0)
#     turbidity = st.number_input('Turbidity 💨', min_value=0.0, value=5.0)
#     st.markdown('</div>', unsafe_allow_html=True)

# # Create a prediction button with better styling
# if st.button('🔮 Prediksi Potabilitas Air', help='Klik untuk memprediksi kualitas air'):
#     with st.spinner('Menganalisis kualitas air...'):
#         # Prepare input data
#         input_data = np.array([[ph, hardness, solids, chloramines, sulfate, 
#                                conductivity, organic_carbon, trihalomethanes, turbidity]])
        
#         # Scale input data
#         input_scaled = scaler.transform(input_data)
        
#         # Make prediction
#         prediction = rf_model.predict(input_scaled)
#         probability = rf_model.predict_proba(input_scaled)
        
#         # Display results
#         st.markdown("### 🎯 Hasil Prediksi")
        
#         # Create columns for results with better styling
#         col1, col2 = st.columns(2)
        
#         with col1:
#             if prediction[0] == 1:
#                 st.markdown("""
#                 <div style="background-color: #d4edda; color: #155724; padding: 1rem; border-radius: 10px; text-align: center;">
#                     <h3>✅ Air LAYAK untuk diminum</h3>
#                 </div>
#                 """, unsafe_allow_html=True)
#             else:
#                 st.markdown("""
#                 <div style="background-color: #f8d7da; color: #721c24; padding: 1rem; border-radius: 10px; text-align: center;">
#                     <h3>❌ Air TIDAK LAYAK untuk diminum</h3>
#                 </div>
#                 """, unsafe_allow_html=True)
        
#         with col2:
#             st.markdown(f"""
#             <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px;">
#                 <h4>Probabilitas Prediksi:</h4>
#                 <ul>
#                     <li>Tidak Layak Minum: {probability[0][0]:.2%}</li>
#                     <li>Layak Minum: {probability[0][1]:.2%}</li>
#                 </ul>
#             </div>
#             """, unsafe_allow_html=True)
        
#         # Display feature importance with better styling
#         st.markdown("### 📈 Pentingnya Parameter")
#         feature_importance = pd.DataFrame({
#             'Parameter': features,
#             'Importance': rf_model.feature_importances_
#         }).sort_values('Importance', ascending=False)
        
#         st.bar_chart(feature_importance.set_index('Parameter'))

# # Add information about parameters with better styling
# st.markdown("### ℹ️ Informasi Parameter")
# param_info = """
# <div class="parameter-info">
#     <table>
#         <tr>
#             <th>Parameter</th>
#             <th>Deskripsi</th>
#             <th>Satuan</th>
#         </tr>
#         <tr>
#             <td>pH 🧪</td>
#             <td>Tingkat keasaman air</td>
#             <td>0-14</td>
#         </tr>
#         <tr>
#             <td>Hardness 💎</td>
#             <td>Tingkat kesadahan air</td>
#             <td>mg/L</td>
#         </tr>
#         <tr>
#             <td>Solids 🌫️</td>
#             <td>Total padatan terlarut dalam air</td>
#             <td>mg/L</td>
#         </tr>
#         <tr>
#             <td>Chloramines 🧬</td>
#             <td>Kadar kloramin dalam air</td>
#             <td>mg/L</td>
#         </tr>
#         <tr>
#             <td>Sulfate ⚗️</td>
#             <td>Kadar sulfat dalam air</td>
#             <td>mg/L</td>
#         </tr>
#         <tr>
#             <td>Conductivity ⚡</td>
#             <td>Konduktivitas air</td>
#             <td>μS/cm</td>
#         </tr>
#         <tr>
#             <td>Organic Carbon 🍃</td>
#             <td>Kadar karbon organik dalam air</td>
#             <td>mg/L</td>
#         </tr>
#         <tr>
#             <td>Trihalomethanes 🔬</td>
#             <td>Kadar trihalometan dalam air</td>
#             <td>μg/L</td>
#         </tr>
#         <tr>
#             <td>Turbidity 💨</td>
#             <td>Tingkat kekeruhan air</td>
#             <td>NTU</td>
#         </tr>
#     </table>
# </div>
# """
# st.markdown(param_info, unsafe_allow_html=True)

# # Footer
# st.markdown("""
# <div style="text-align: center; margin-top: 2rem; padding: 1rem; background-color: #f8f9fa; border-radius: 10px;">
#     <p>💧 Aplikasi Prediksi Potabilitas Air | Dibuat dengan Streamlit</p>
# </div>
# """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler

# Set page config
st.set_page_config(page_title="Water Potability Prediction", layout="wide")

# Title
st.title("\U0001F4A7 Prediksi Potabilitas Air dengan Random Forest")
st.write("Aplikasi ini memprediksi apakah air layak minum berdasarkan parameter kualitas air.")

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('water_potability.csv')
    df.fillna(df.mean(), inplace=True)
    
    # Handle outliers
    for column in df.columns[:-1]:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return df

# Load the data
water_data = load_data()

# Prepare features and target
features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
X = water_data[features]
y = water_data['Potability']

# Resample data
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

# Train the model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Calculate accuracy
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display model accuracy
st.sidebar.header("Performa Model")
st.sidebar.metric(label="Akurasi Model", value=f"{accuracy:.2%}")
st.sidebar.info(
    "Model Random Forest ini dilatih dengan teknik oversampling untuk menangani ketidakseimbangan kelas."
)

# Create input form
st.header("Input Parameter Kualitas Air")

# Create three columns for input fields
col1, col2, col3 = st.columns(3)

with col1:
    ph = st.number_input('pH', min_value=0.0, max_value=14.0, value=7.0)
    hardness = st.number_input('Hardness', min_value=0.0, value=200.0)
    solids = st.number_input('Solids (Total Dissolved Solids)', min_value=0.0, value=20000.0)

with col2:
    chloramines = st.number_input('Chloramines', min_value=0.0, value=4.0)
    sulfate = st.number_input('Sulfate', min_value=0.0, value=250.0)
    conductivity = st.number_input('Conductivity', min_value=0.0, value=400.0)

with col3:
    organic_carbon = st.number_input('Organic Carbon', min_value=0.0, value=10.0)
    trihalomethanes = st.number_input('Trihalomethanes', min_value=0.0, value=50.0)
    turbidity = st.number_input('Turbidity', min_value=0.0, value=5.0)

# Create a prediction button
if st.button('Prediksi Potabilitas Air'):
    # Prepare input data
    input_data = np.array([[ph, hardness, solids, chloramines, sulfate, 
                           conductivity, organic_carbon, trihalomethanes, turbidity]])
    
    # Scale input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = rf_model.predict(input_scaled)
    probability = rf_model.predict_proba(input_scaled)
    
    # Display results
    st.subheader("Hasil Prediksi")
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction[0] == 1:
            st.success("\U0001F4A7 Air LAYAK untuk diminum")
        else:
            st.error("\U0001F6AB Air TIDAK LAYAK untuk diminum")
    
    with col2:
        st.write("Probabilitas Prediksi:")
        st.write(f"Tidak Layak Minum: {probability[0][0]:.2%}")
        st.write(f"Layak Minum: {probability[0][1]:.2%}")
    
    # Display feature importance
    st.subheader("Pentingnya Parameter")
    feature_importance = pd.DataFrame({
        'Parameter': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    st.bar_chart(feature_importance.set_index('Parameter'))

# Add information about parameters
st.sidebar.header("Informasi Parameter")
st.sidebar.markdown("""
- **pH**: Tingkat keasaman air (0-14)
- **Hardness**: Tingkat kesadahan air (mg/L)
- **Solids**: Total padatan terlarut dalam air (mg/L)
- **Chloramines**: Kadar kloramin dalam air (mg/L)
- **Sulfate**: Kadar sulfat dalam air (mg/L)
- **Conductivity**: Konduktivitas air (\u03bcS/cm)
- **Organic Carbon**: Kadar karbon organik dalam air (mg/L)
- **Trihalomethanes**: Kadar trihalometan dalam air (\u03bcg/L)
- **Turbidity**: Tingkat kekeruhan air (NTU)
""")
