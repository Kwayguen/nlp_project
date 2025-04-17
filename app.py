import os
import io

import streamlit as st
from ocr_processor import OCRProcessor
from langage_analyser import LanguageAnalyzer
from document_type_detector import DocumentTypeDetector
from document_summarizer import DocumentSummarizer

st.title("Analyseur de documents OCR")

# Barre latérale pour la navigation entre les pages
page = st.sidebar.radio(
    "Navigation",
    ["OCR", "Détecter Langue + traduction", "Détecter le type de document", "Résumer le document"]
)

# Initialisation du texte extrait en session
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

# ─────────────── Page OCR ─────────────────────
if page == "OCR":
    st.write("Choisissez la source du document à analyser :")
    source = st.radio("Source", ["Téléverser un fichier", "Documents locaux"])

    file_obj = None
    file_ext = None

    if source == "Téléverser un fichier":
        uploaded_file = st.file_uploader(
            "Choisissez un fichier",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "pdf"]
        )
        if uploaded_file is not None:
            file_obj = uploaded_file
            file_ext = uploaded_file.name.rsplit(".", 1)[-1]

    else:  # Documents locaux
        docs_dir = "Documents"
        if os.path.isdir(docs_dir):
            # on liste les fichiers valides
            files = [
                f for f in os.listdir(docs_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".pdf"))
            ]
            if files:
                selected = st.selectbox("Fichiers disponibles", files)
                if selected:
                    path = os.path.join(docs_dir, selected)
                    with open(path, "rb") as f:
                        data = f.read()
                    file_obj = io.BytesIO(data)
                    file_ext = selected.rsplit(".", 1)[-1]
            else:
                st.warning("Le dossier `Documents/` est vide.")
        else:
            st.error("Le dossier `Documents/` n'existe pas à la racine du projet.")

    if st.button("Analyser"):
        if file_obj is not None:
            ocr = OCRProcessor()
            try:
                extracted = ocr.extract_text(file_obj, file_ext)
                st.session_state.extracted_text = extracted
                st.text_area("Texte extrait", extracted, height=300)
            except Exception as e:
                st.error(f"Erreur lors de l'extraction du texte : {e}")
        else:
            st.warning("Veuillez d'abord sélectionner ou téléverser un document.")

# ─── Page Détecter Langue + traduction ───
elif page == "Détecter Langue + traduction":
    st.write("Détecter la langue du texte extrait et le traduire.")
    if st.session_state.extracted_text:
        st.text_area("Texte extrait", st.session_state.extracted_text, height=300)
        if st.button("Détecter et Traduire"):
            analyzer = LanguageAnalyzer()
            lang = analyzer.detect_language(st.session_state.extracted_text)
            st.write("Langue détectée :", lang)
            st.text_area("Texte traduit", analyzer.translate_text(st.session_state.extracted_text), height=300)
    else:
        st.warning("Aucun texte disponible. Passez d'abord par OCR.")

# ─── Page Détecter le type de document ───
elif page == "Détecter le type de document":
    st.write("Détecter le type de document à partir du texte extrait.")
    if st.session_state.extracted_text:
        st.text_area("Texte extrait", st.session_state.extracted_text, height=300)
        if st.button("Détecter Type"):
            if "doc_detector" not in st.session_state:
                st.session_state.doc_detector = DocumentTypeDetector()
            scores = st.session_state.doc_detector(st.session_state.extracted_text)
            st.subheader("Scores (ordre décroissant)")
            for label, score in scores:
                st.write(f"- **{label}** : {score:.2%}")
    else:
        st.warning("Aucun texte disponible. Passez d'abord par OCR.")

# ─── Page Résumer le document ───
elif page == "Résumer le document":
    st.write("Résumer le document à partir du texte extrait.")
    if st.session_state.extracted_text:
        st.text_area("Texte extrait", st.session_state.extracted_text, height=300)
        if st.button("Résumer"):
            summarizer = DocumentSummarizer()
            summary = summarizer.summarize_text(st.session_state.extracted_text, max_length=150)
            st.subheader("Résumé")
            st.text_area("Résumé du document", summary, height=150)
    else:
        st.warning("Aucun texte disponible. Passez d'abord par OCR.")
