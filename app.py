import streamlit as st
from ocr_processor import OCRProcessor
from langage_analyser import LanguageAnalyzer

st.title("Analyseur de documents OCR")

# Barre latérale pour la navigation entre les pages
page = st.sidebar.radio("Navigation", ["OCR", "Détecter Langue"])

# Utilisation de session_state pour conserver le texte extrait entre les interactions
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

# Page OCR : Extraction du texte depuis une image ou un PDF
if page == "OCR":
    st.write("Uploader une image ou un PDF pour extraire le texte.")
    uploaded_file = st.file_uploader("Choisissez un fichier", type=["jpg", "jpeg", "png", "bmp", "tiff", "pdf"])
    if st.button("Analyser"):
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1]
            ocr = OCRProcessor()
            try:
                extracted_text = ocr.extract_text(uploaded_file, file_extension)
                st.session_state.extracted_text = extracted_text
                st.text_area("Texte extrait", extracted_text, height=300)
            except Exception as e:
                st.error(f"Erreur lors de l'extraction du texte : {e}")
        else:
            st.warning("Veuillez uploader un fichier.")

# Page Détecter Langue : Analyse du texte extrait pour détecter la langue utilisée
elif page == "Détecter Langue":
    st.write("Détecter la langue du texte extrait.")
    if st.session_state.extracted_text:
        st.text_area("Texte extrait", st.session_state.extracted_text, height=300)
        if st.button("Détecter Langue"):
            analyzer = LanguageAnalyzer()
            language = analyzer.detect_language(st.session_state.extracted_text)
            st.write("La langue détectée est :", language)
    else:
        st.warning("Aucun texte disponible. Veuillez d'abord extraire le texte sur la page OCR.")


    