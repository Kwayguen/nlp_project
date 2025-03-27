from transformers import pipeline

def get_translator(model_name: str):
    """
    Crée et renvoie un pipeline de traduction basé sur le modèle Hugging Face spécifié.
    
    :param model_name: Nom du modèle sur Hugging Face (ex: "Helsinki-NLP/opus-mt-fr-en")
    :return: Un objet pipeline de traduction.
    """
    translator = pipeline("translation", model=model_name)
    return translator