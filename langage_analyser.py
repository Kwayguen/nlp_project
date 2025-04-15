from langdetect import detect, DetectorFactory, LangDetectException
from transformers import pipeline

class LanguageAnalyzer:
    """
    Classe qui analyse un texte, retourne le code de langue détecté
    et traduit le texte dans la langue opposée (anglais <-> français)
    en utilisant un modèle Hugging Face.
    
    Nécessite :
      - langdetect (pip install langdetect)
      - transformers et sentencepiece (pip install transformers sentencepiece)
    """
    def __init__(self):
        # Pour obtenir des résultats reproductibles
        DetectorFactory.seed = 0
        # Initialisation des pipelines de traduction
        self.trans_en_fr = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
        self.trans_fr_en = pipeline("translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en")

    def detect_language(self, text):
        """
        Détecte la langue du texte fourni.
        
        :param text: chaîne de caractères à analyser
        :return: code de langue (ex: 'fr' pour français, 'en' pour anglais)
        """
        try:
            language = detect(text)
            return language
        except LangDetectException as e:
            return f"Erreur lors de la détection de la langue : {e}"

    def translate_text(self, text):
        """
        Traduit le texte dans la langue opposée :
        - Si le texte est en anglais, il sera traduit en français.
        - Si le texte est en français, il sera traduit en anglais.
        
        La traduction est réalisée via le modèle Hugging Face approprié.
        
        :param text: texte à traduire 
        :return: texte traduit ou message d'erreur si la langue n'est pas supportée.
        """
        lang = self.detect_language(text)
        if lang == "en":
            try:
                result = self.trans_en_fr(text)
                return result[0]["translation_text"]
            except Exception as e:
                return f"Erreur lors de la traduction : {e}"
        elif lang == "fr":
            try:
                result = self.trans_fr_en(text)
                return result[0]["translation_text"]
            except Exception as e:
                return f"Erreur lors de la traduction : {e}"
        else:
            return "Langue non supportée pour la traduction (seulement 'en' et 'fr')."
