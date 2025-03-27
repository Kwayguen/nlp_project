import streamlit as st
from ocr_processor import OCRProcessor

st.title("Analyseur de documents OCR")
st.write("Uploader une image ou un PDF pour extraire le texte.")

# Autoriser uniquement les formats supportés
uploaded_file = st.file_uploader("Choisissez un fichier", type=["jpg", "jpeg", "png", "bmp", "tiff", "pdf"])

if st.button("Analyser"):
    if uploaded_file is not None:
        # Récupérer l'extension du fichier
        file_extension = uploaded_file.name.split('.')[-1]
        ocr = OCRProcessor()
        try:
            # Extraction du texte
            extracted_text = ocr.extract_text(uploaded_file, file_extension)
            st.text_area("Texte extrait", extracted_text, height=300)
        except Exception as e:
            st.error(f"Erreur lors de l'extraction du texte : {e}")
    else:
        st.warning("Veuillez uploader un fichier.")
