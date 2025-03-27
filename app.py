import streamlit as st
from ocr_processor import OCRProcessor
from langage_analyser import LanguageAnalyzer

st.title("Analyseur de documents OCR")
st.write("Uploader une image ou un PDF pour extraire le texte.")

# Autoriser uniquement les formats supportés
uploaded_file = st.file_uploader("Choisissez un fichier", type=["jpg", "jpeg", "png", "bmp", "tiff", "pdf"])

# Utilisation de session_state pour conserver le texte extrait entre les interactions
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

# Disposer les boutons "Analyser" et "Détecter Langue" côte à côte
col1, col2 = st.columns(2)

with col1:
    if st.button("Analyser"):
        if uploaded_file is not None:
            # Récupérer l'extension du fichier
            file_extension = uploaded_file.name.split('.')[-1]
            ocr = OCRProcessor()
            try:
                # Extraction du texte
                extracted_text = ocr.extract_text(uploaded_file, file_extension)
                st.session_state.extracted_text = extracted_text
                st.text_area("Texte extrait", extracted_text, height=300)
            except Exception as e:
                st.error(f"Erreur lors de l'extraction du texte : {e}")
        else:
            st.warning("Veuillez uploader un fichier.")

with col2:
    if st.button("Détecter Langue"):
        if st.session_state.extracted_text:
            analyzer = LanguageAnalyzer()
            language = analyzer.detect_language(st.session_state.extracted_text)
            st.write("La langue détectée est :", language)
        else:
            st.warning("Aucun texte à analyser. Veuillez d'abord extraire le texte.")
