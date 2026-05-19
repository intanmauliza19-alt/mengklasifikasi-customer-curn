import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set judul aplikasi
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📊 Customer Churn Prediction App")
st.markdown("Aplikasi ini memprediksi apakah pelanggan akan berhenti berlangganan (Churn) atau tidak.")

# 1. Fungsi untuk memuat model dan komponen
@st.cache_resource
def load_components():
    with open('best_model_churn.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, scaler, encoders

try:
    model, scaler, encoders = load_components()
except FileNotFoundError:
    st.error("File model/scaler/encoder tidak ditemukan. Pastikan sudah menjalankan training sebelumnya.")
    st.stop()

# 2. Membuat Form Input di Sidebar
st.sidebar.header("Input Data Pelanggan")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ("Female", "Male"))
    subscription = st.sidebar.selectbox("Subscription Type", ("Basic", "Standard", "Premium"))
    contract = st.sidebar.selectbox("Contract Length", ("Annual", "Monthly", "Quarterly"))
    tenure = st.sidebar.slider("Tenure (Bulan)", 1, 60, 12)
    usage = st.sidebar.slider("Usage Frequency", 1, 30, 15)
    support = st.sidebar.slider("Support Calls", 0, 10, 3)
    delay = st.sidebar.slider("Payment Delay", 0, 30, 5)

    data = {
        'Gender': gender,
        'Tenure': tenure,
        'Usage Frequency': usage,
        'Support Calls': support,
        'Payment Delay': delay,
        'Subscription Type': subscription,
        'Contract Length': contract
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 3. Tampilkan Ringkasan Input
st.subheader("Data Input Pelanggan")
st.write(input_df)

# 4. Preprocessing
processed_df = input_df.copy()

# Mapping manual (sebagai backup jika encoder mismatch)
manual_map = {
    'Gender': {'Female': 0, 'Male': 1},
    'Subscription Type': {'Basic': 0, 'Premium': 1, 'Standard': 2},
    'Contract Length': {'Annual': 0, 'Monthly': 1, 'Quarterly': 2}
}

for col in ['Gender', 'Subscription Type', 'Contract Length']:
    try:
        processed_df[col] = encoders[col].transform(processed_df[col])
    except:
        processed_df[col] = processed_df[col].map(manual_map[col])

# Scaling
X_scaled = scaler.transform(processed_df)

# 5. Prediksi
if st.button("Lakukan Prediksi"):
    prediction = model.predict(X_scaled)
    prediction_proba = model.predict_proba(X_scaled)

    st.subheader("Hasil Prediksi")
    if prediction[0] == 1:
        st.error("Status: CHURN (Kemungkinan Berhenti)")
    else:
        st.success("Status: RETAIN (Kemungkinan Bertahan)")

    st.write(f"**Probabilitas Churn:** {prediction_proba[0][1]:.2%}")
    
    # Visualisasi probabilitas
    st.progress(prediction_proba[0][1])
