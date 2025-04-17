# document_summarizer.py  –  résumé FR fiable avec fallback si pas de Java
from __future__ import annotations
from transformers import pipeline
import unicodedata
import re
import logging

# Modèle de résumé français
_SUM_MODEL = "plguillou/t5-base-fr-sum-cnndm"

# Essaye d'activer LanguageTool (nécessite Java); sinon on désactive la correction
try:
    import language_tool_python
    _TOOL = language_tool_python.LanguageTool("fr")
    _GRAMMAR_ENABLED = True
    logging.info("LanguageTool activé pour correction grammaticale.")
except (ModuleNotFoundError, OSError) as e:
    _TOOL = None
    _GRAMMAR_ENABLED = False
    logging.warning(
        "LanguageTool désactivé (pas de Java ou installation manquante). "
        "Le résumé sera retourné sans correction grammaticale."
    )

def _clean(txt: str) -> str:
    """Nettoyage minimal : unicode, ponctuation non désirée, espaces."""
    txt = unicodedata.normalize("NFKC", txt)
    txt = re.sub(r"[^\w\s\.,;:!\?()€%/\-]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()

class DocumentSummarizer:
    """Génère un résumé en français, avec correction grammaticale si possible."""

    def __init__(
        self,
        model_name: str | None = None,
        device: int = -1
    ):
        model_name = model_name or _SUM_MODEL
        try:
            self.pipe = pipeline(
                task="summarization",
                model=model_name,
                tokenizer=model_name,
                device=device
            )
        except Exception as e:
            logging.error(f"Impossible de charger le modèle {model_name} : {e}")
            raise

    def summarize_text(
        self,
        text: str,
        max_length: int = 120,
        min_length: int = 20,
        do_sample: bool = False
    ) -> str:
        if not text or not text.strip():
            return ""

        try:
            # 1) nettoyage du texte d'entrée
            txt = _clean(text)

            # 2) préfixe requis par T5
            if not txt.lower().startswith("summarize:"):
                txt = "summarize: " + txt

            # 3) génération du résumé
            raw = self.pipe(
                txt,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                clean_up_tokenization_spaces=True,
            )[0]["summary_text"].strip()

            # 4) correction grammaticale si disponible
            if _GRAMMAR_ENABLED and _TOOL is not None:
                try:
                    return _TOOL.correct(raw).strip()
                except Exception as e:
                    logging.warning(f"Échec correction grammaticale : {e}")
                    return raw

            # sinon on renvoie juste le résumé brut
            return raw

        except Exception as e:
            logging.error(f"Erreur lors du résumé : {e}")
            return f"⚠️ Erreur lors du résumé : {e}"
