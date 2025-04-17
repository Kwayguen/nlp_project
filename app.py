import streamlit as st
from ocr_processor import OCRProcessor
from langage_analyser import LanguageAnalyzer
from document_type_detector import DocumentTypeDetector
from document_summarizer import DocumentSummarizer


st.title("Analyseur de documents OCR")

# Barre latérale pour la navigation entre les pages
page = st.sidebar.radio("Navigation", ["OCR", "Détecter Langue + traduction", "Détecter le type de document", "Résumer le document"])

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
elif page == "Détecter Langue + traduction":
    st.write("Détecter la langue du texte extrait et le traduire dans la langue opposée.")
    if st.session_state.extracted_text:
        st.text_area("Texte extrait", st.session_state.extracted_text, height=300)
        if st.button("Détecter Langue et Traduire"):
            analyzer = LanguageAnalyzer()
            language = analyzer.detect_language(st.session_state.extracted_text)
            st.write("La langue détectée est :", language)
            translation = analyzer.translate_text(st.session_state.extracted_text)
            st.text_area("Texte traduit", translation, height=300)
    else:
        st.warning("Aucun texte disponible. Veuillez d'abord extraire le texte sur la page OCR.")

elif page == "Détecter le type de document":
    st.write("Détecter le type de document à partir du texte extrait.")
    if st.session_state.extracted_text:
        st.text_area("Texte extrait", st.session_state.extracted_text, height=300)
        if st.button("Détecter Type de Document"):
            if "doc_detector" not in st.session_state:
                st.session_state.doc_detector = DocumentTypeDetector()
            scores = st.session_state.doc_detector(st.session_state.extracted_text)

            st.subheader("Scores (ordre décroissant)")
            for label, sc in scores:
                st.write(f"- **{label}** : {sc:.2%}")
    else:
        st.warning("Aucun texte disponible. Veuillez d'abord extraire le texte sur la page OCR.")



elif page == "Résumer le document":
    st.write("Résumer le document à partir du texte extrait.")
    if st.session_state.extracted_text:
        st.text_area("Texte extrait", st.session_state.extracted_text, height=300)
        if st.button("Résumer le Document"):
            from document_summarizer import DocumentSummarizer  # Assurez-vous que ce fichier est accessible
            summarizer = DocumentSummarizer()
            summary = summarizer.summarize_text(st.session_state.extracted_text, max_length=150)
            st.subheader("Résumé du document")
            st.text_area("Résumé du document", summary, height=150)
    else:
        st.warning("Aucun texte disponible. Veuillez d'abord extraire le texte sur la page OCR.")

