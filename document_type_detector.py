from transformers import pipeline

class DocumentTypeDetector:
    """
    Classe pour détecter le type de document à partir du texte extrait
    en utilisant un modèle Hugging Face de classification zéro-shot.
    
    Nécessite :
      - transformers (pip install transformers)
    """
    def __init__(self):
        # Initialisation du pipeline de classification zéro-shot avec le modèle "facebook/bart-large-mnli"
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        # Définition des catégories candidates pour le type de document.
        self.candidate_labels = [
            "document d'identité",
            "passeport",
            "facture",
            "relevé bancaire",
            "contrat",
            "lettre",
            "document administratif",
            "document juridique"
        ]

    def detect_document_type(self, text):
        """
        Détecte le type de document à partir du texte extrait.
        
        :param text: texte extrait du document
        :return: tuple (type de document détecté, score de confiance)
        """
        if not text.strip():
            return "Texte vide, impossible de déterminer le type de document.", None
        
        result = self.classifier(text, candidate_labels=self.candidate_labels)
        best_label = result["labels"][0]
        best_score = result["scores"][0]
        return best_label, best_score
