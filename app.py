import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

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
        st.dataframe(data[gen_cols].head())
        
        if st.button("Analisis"):
            X = data[gen_cols].values
            
            if X.shape[1] != 20531:
                if X.shape[1] > 20531:
                    X = X[:, :20531]
                else:
                    pad = np.zeros((X.shape[0], 20531 - X.shape[1]))
                    X = np.hstack([X, pad])
            
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            
            st.success(f"Jumlah sampel: {len(predictions)}")
            
            hasil = pd.DataFrame({
                "Sampel": range(1, len(predictions)+1),
                "Prediksi": predictions,
                "Confidence": probabilities.max(axis=1)
            })
            st.dataframe(hasil)
            
            st.write("### Probabilitas Sampel Pertama")
            fig, ax = plt.subplots()
            ax.bar(model.classes_, probabilities[0])
            ax.set_ylabel("Probabilitas")
            st.pyplot(fig)
            
            st.write("### Interpretasi")
            pred = predictions[0]
            if pred == "BRCA":
                st.info("BRCA: Kanker payudara. Terkait mutasi BRCA1/BRCA2.")
            elif pred == "KIRC":
                st.info("KIRC: Kanker ginjal. Terkait mutasi VHL.")
            elif pred == "LUAD":
                st.info("LUAD: Kanker paru. Terkait mutasi EGFR/KRAS.")
            elif pred == "PRAD":
                st.info("PRAD: Kanker prostat. Terkait mutasi AR.")
            elif pred == "COAD":
                st.info("COAD: Kanker usus. Terkait mutasi APC/KRAS.")
            
            csv = hasil.to_csv(index=False)
            st.download_button(
                label="Download Hasil",
                data=csv,
                file_name="hasil_prediksi.csv",
                mime="text/csv"
            )
