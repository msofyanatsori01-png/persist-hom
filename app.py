import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ===== PASSWORD =====
password = st.text_input("Masukkan password:", type="password")

if password != "opsi2027":
    st.warning("Password salah!")
    st.stop()

# ===== WEB APP =====
st.title("PERSIST-HOM")
st.subheader("Klasifikasi Kanker Berbasis Topologi")

model = pickle.load(open("model.pkl", "rb"))

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data berhasil diupload!")
    
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])
    
    gen_cols = [c for c in data.columns if c.startswith("gene_")]
    
    if len(gen_cols) == 0:
        st.error("Tidak ada kolom gene_")
    else:
        st.write(f"Jumlah gen: {len(gen_cols)}")
        
        if st.button("Analisis"):
            X = data[gen_cols].values
            
            if X.shape[1] != 20531:
                if X.shape[1] > 20531:
                    X = X[:, :20531]
                else:
                    pad = np.zeros((X.shape[0], 20531 - X.shape[1]))
                    X = np.hstack([X, pad])
            
            prediction = model.predict(X)
            probability = model.predict_proba(X)
            
            st.success(f"Subtipe: {prediction[0]}")
            st.write(f"Confidence: {probability[0].max():.2%}")
            
            st.write("### Probabilitas per Kelas")
            for i, cls in enumerate(model.classes_):
                st.write(f"{cls}: {probability[0][i]:.2%}")
