# Analyseur de documents OCR

Cette application Streamlit vous permet de traiter des documents (images) pour en extraire du texte via OCR, détecter la langue, traduire, déterminer le type de document et générer un résumé.

# 📁 Structure du projet

nlp_project/

├── Documents/                # dossier de vos documents locaux

├── ocr_processor.py          # extraction OCR via Tesseract

├── langage_analyser.py       # détection de langue et traduction automatique

├── document_type_detector.py # classification du type de document (zero-shot)

├── document_summarizer.py    # génération de résumé

├── app.py                    # interface Streamlit

├── requirements.txt          # dépendances Python

└── README.md                 # ce fichier

⚙️ Installation et configuration
Cloner le dépôt

git clone https://github.com/<votre‑user>/<nom‑du‑repo>.git

cd nlp_project

Créer et activer un environnement virtuel Python (recommandé)

# Windows PowerShell
python -m venv venv

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser  # si nécessaire

.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv

source venv/bin/activate

# Installer les dépendances : 
pip install --upgrade pip

pip install -r requirements.txt

# Installer Tesseract OCR ou Easyocr
Téléchargez l’installeur Windows (bundle UB Mannheim) ici :
https://github.com/UB-Mannheim/tesseract/wiki
Installez et ajoutez le chemin vers tesseract.exe dans la variable d’environnement PATH.
Vérifiez l’installation :
tesseract --version

Notre choix c'est porté sur Easyocr, plus simple pour la mise en ligne sur Streamlit Community Cloud. Nous avons tout de même garder le code pour l'utilisation de Tesseract 

Seulement ce choix fait que les performances de la détection du type de document avec l'ORC Easyocr sont moins efficaces que Tesseract mais gardent tout de même de bons résultats.

# 🚀 Lancement de l’application
Dans le dossier racine :
streamlit run app.py

Ouvrez ensuite dans votre navigateur :
http://localhost:8501

# 🖼️ Page « OCR »
Source du document :
Téléverser un fichier (JPG, JPEG, PNG, BMP, TIFF, PDF)
Documents locaux : choisissez un fichier dans Documents/ à la racine.
Extraction : utilise Tesseract pour récupérer le texte brut.

# 🌐 Page « Détecter Langue + traduction »
Détecte la langue (français/anglais) du texte extrait.
Traduit automatiquement vers l’autre langue via MarianMT.

# 📄 Page « Détecter le type de document »
Classifie le texte en 8 types : identité, passeport, facture, relevé bancaire, contrat, lettre, document juridique, sujet d’évaluation.
Utilise un modèle zero-shot (DeBERTa + XNLI) et des règles heuristiques.

# 📝 Page « Résumer le document »
Génère un résumé court (max 150 tokens) du texte extrait.
Basé sur un modèle de résumé (ex. BART ou T5).

# 🛠️ Déploiement en ligne
Streamlit Community Cloud

Url de l'application en ligne : https://nlpproject-dyowzremci2ede3v3ditky.streamlit.app/

Formats supportés : JPG, JPEG, PNG, BMP, TIFF, PDF, TIF.
Auteur : Roumagne Hugo & Maulavé Julien – Avril 2025

