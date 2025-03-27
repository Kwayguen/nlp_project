from langdetect import detect, DetectorFactory, LangDetectException

class LanguageAnalyzer:
    """
    Classe qui analyse un texte et retourne le code de langue détecté.
    Nécessite le package langdetect (pip install langdetect).
    """
    def __init__(self):
        # Pour obtenir des résultats reproductibles
        DetectorFactory.seed = 0

    def detect_language(self, text):
        """
        Détecte la langue du texte fourni.
        
        :param text: chaîne de caractères à analyser
        :return: code de langue (ex: 'fr' pour français, 'en' pour anglais), ou un message d'erreur si la détection échoue
        """
        try:
            language = detect(text)
            return language
        except LangDetectException as e:
            return f"Erreur lors de la détection de la langue : {e}"
