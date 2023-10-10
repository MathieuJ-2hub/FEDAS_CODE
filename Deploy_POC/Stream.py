import streamlit as st
import pandas as pd
from Load_and_give_results import load_and_preprocess_data,make_prediction

# UI
st.title("Application Streamlit pour la prédiction des FEDAS CODE")

# Bouton pour choisir le fichier
Inputfile= st.text_input('InputFile')
OutputFile= st.text_input('OutputFile')

if OutputFile and Inputfile is not None:
    if st.button("Run Program"):
        with st.spinner('Veuillez patienter... La fonction est en cours d\'exécution...'):
            results=make_prediction(load_and_preprocess_data(Inputfile), OutputFile)
            print(results)



