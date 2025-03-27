# resume.py

from transformers import pipeline

class Doctype:
    """
    Cette classe permet de recevoir un texte et de déterminer le type de document
    grâce au modèle document_type_identification hébergé sur Hugging Face.
    """

    def __init__(self, model_name: str = "sguarnaccio/document_type_identification"):
        """
        Initialise le pipeline de classification de documents.
        :param model_name: Nom du modèle Hugging Face à utiliser.
        """
        self.model_name = model_name
        self.classifier = pipeline("text-classification", model=self.model_name)

    def guess_document_type(self, text: str) -> str:
        """
        Utilise le pipeline de classification pour prédire le type de document.
        :param text: Le texte sur lequel on veut faire la prédiction.
        :return: Le label (type de document) prédit par le modèle.
        """
        if not text.strip():
            return "Texte vide, impossible de déterminer le type de document."

        # Le pipeline retourne en général un tableau de prédictions, ex : [{'label': '...', 'score': 0.98}]
        result = self.classifier(text)
        if not result:
            return "Impossible de déterminer le type de document."

        # On récupère le premier label (le mieux classé) du pipeline
        doc_type = result[0]["label"]
        return doc_type
