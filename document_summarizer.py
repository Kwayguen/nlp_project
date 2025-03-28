from transformers import pipeline
import logging

class DocumentSummarizer:
    """
    Classe pour résumer un texte à l'aide d'un modèle de Hugging Face.
    Par défaut, le modèle utilisé est "facebook/bart-large-cnn".
    """

    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: int = -1):
        """
        Initialise le pipeline de résumé.
        
        Args:
            model_name (str): Nom du modèle Hugging Face à utiliser.
            device (int): -1 pour utiliser le CPU, sinon le numéro du GPU.
        """
        try:
            self.summarizer = pipeline("summarization", model=model_name, device=device)
        except Exception as e:
            logging.error(f"Erreur lors du chargement du modèle '{model_name}': {e}")
            raise e

    def summarize_text(self, text: str, max_length: int = 130, min_length: int = 30, do_sample: bool = False) -> str:
        """
        Génère un résumé du texte fourni.
        
        Args:
            text (str): Le texte à résumer.
            max_length (int): Longueur maximale du résumé généré.
            min_length (int): Longueur minimale du résumé généré.
            do_sample (bool): Si True, active l'échantillonnage (utile pour des variations).
        
        Returns:
            str: Le résumé généré.
        """
        try:
            # Appel au pipeline de résumé
            summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
            return summary[0]['summary_text']
        except Exception as e:
            logging.error(f"Erreur lors de la génération du résumé: {e}")
            return f"Erreur lors de la génération du résumé: {e}"
