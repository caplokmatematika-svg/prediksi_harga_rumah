import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load resources
@st.cache_resource
def load_resources():
    model = pickle.load(open('gb_model.pkl', 'rb'))
    scaler_fitur = pickle.load(open('scaler_fitur.pkl', 'rb'))
    target_scaler = pickle.load(open('target_scaler.pkl', 'rb'))
    encoders = pickle.load(open('label_encoder.pkl', 'rb'))
    rentang = pickle.load(open('rentang_fitur.pkl', 'rb'))
    return model, scaler_fitur, target_scaler, encoders, rentang

model, scaler_fitur, target_scaler, encoders, rentang = load_resources()

st.title("Aplikasi Prediksi Harga Rumah")
st.write("Masukkan detail properti untuk mendapatkan estimasi harga.")

# Input Numerik menggunakan Slider
st.header("Fitur Numerik")
bedrooms = st.slider("Jumlah Kamar Tidur", 
                     min_value=float(rentang['bedrooms']['min']), 
                     max_value=float(rentang['bedrooms']['max']), 
                     value=float(rentang['bedrooms']['min']))

bathrooms = st.slider("Jumlah Kamar Mandi", 
                      min_value=float(rentang['bathrooms']['min']), 
                      max_value=float(rentang['bathrooms']['max']), 
                      value=float(rentang['bathrooms']['min']))

sqft_living = st.slider("Luas Bangunan (sqft living)", 
                        min_value=int(rentang['sqft_living']['min']), 
                        max_value=int(rentang['sqft_living']['max']), 
                        value=int(rentang['sqft_living']['min']))

floors = st.slider("Jumlah Lantai", 
                   min_value=float(rentang['floors']['min']), 
                   max_value=float(rentang['floors']['max']), 
                   value=float(rentang['floors']['min']))

sqft_above = st.slider("Luas Lantai Atas (sqft above)", 
                       min_value=int(rentang['sqft_above']['min']), 
                       max_value=int(rentang['sqft_above']['max']), 
                       value=int(rentang['sqft_above']['min']))

# Input Kategorikal menggunakan Combo Box (Selectbox)
st.header("Lokasi Properti")
city_list = encoders['city'].classes_
city = st.selectbox("Pilih Kota", city_list)

zip_list = encoders['statezip'].classes_
statezip = st.selectbox("Pilih State Zip", zip_list)

if st.button("Prediksi Harga"):
    # Transformasi input kategorikal
    enc_city = encoders['city'].transform([city])[0]
    enc_zip = encoders['statezip'].transform([statezip])[0]

    # Membuat DataFrame input
    X_input = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'sqft_living': [sqft_living],
        'floors': [floors],
        'sqft_above': [sqft_above],
        'city': [enc_city],
        'statezip': [enc_zip]
    })

    # Scaling fitur
    X_scaled = scaler_fitur.transform(X_input)

    # Prediksi
    pred_skala = model.predict(X_scaled)

    # Inverse transform untuk harga asli
    harga_asli = target_scaler.inverse_transform(pred_skala.reshape(-1, 1))

    st.success(f"Estimasi Harga Rumah adalah: **${harga_asli[0][0]:,.2f}**")
