from functools import lru_cache
from transformers import pipeline, AutoTokenizer
import torch
import re
from langdetect import detect as lang_detect, DetectorFactory, LangDetectException

# Configuration du détecteur de langue
DetectorFactory.seed = 0

# Limites pour la traduction automatique (MarianMT)
_MAX_MODEL_LEN = 512       # longueur max du modèle
_CHUNK_MARGIN   = 400      # marge (~80 % de la capacité)

@lru_cache
def _translator(model_name: str, device: int):
    """Retourne un pipeline de traduction pré‑chargé."""
    return pipeline("translation", model=model_name, device=device)

class LanguageAnalyzer:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        # pipelines de traduction
        self.en_fr = _translator("Helsinki-NLP/opus-mt-en-fr", self.device)
        self.fr_en = _translator("Helsinki-NLP/opus-mt-fr-en", self.device)
        # tokenizers correspondants
        self.en_tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        self.fr_tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

    def detect_language(self, text: str):
        """
        Détecte la langue d'un texte (retourne 'en' ou 'fr').
        En cas d'erreur de détection, renvoie un message d'erreur.
        """
        try:
            return lang_detect(text)
        except LangDetectException as e:
            return f"Erreur : {e}"

    def translate_text(self, text: str):
        """
        Traduit un texte de l'anglais vers le français ou vice versa.
        Ne prend en charge que 'en' et 'fr'.
        """
        lang = self.detect_language(text)
        if lang not in ("en", "fr"):
            return "Langue non prise en charge."

        # Choix du modèle et du tokenizer
        if lang == "en":
            model, tokenizer = self.en_fr, self.en_tok
        else:
            model, tokenizer = self.fr_en, self.fr_tok

        # Découpage intelligent pour ne pas dépasser la longueur max
        chunks = self._smart_split(text, tokenizer)
        translations = []
        for chunk in chunks:
            res = model(chunk, max_length=_MAX_MODEL_LEN, truncation=True)[0]["translation_text"]
            translations.append(res.strip())

        return "\n\n".join(translations)

    def _smart_split(self, text: str, tokenizer):
        """
        Scinde le texte en segments d'environ _CHUNK_MARGIN tokens
        sans couper au milieu des phrases.
        """
        parts = []
        buffer = ""
        for sent in re.split(r'(?<=[.!?])\s+', text):
            candidate = (buffer + " " + sent).strip()
            if len(tokenizer(candidate)["input_ids"]) < _CHUNK_MARGIN:
                buffer = candidate
            else:
                if buffer:
                    parts.append(buffer)
                buffer = sent
        if buffer:
            parts.append(buffer)
        return parts
