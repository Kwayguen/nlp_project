# document_summarizer.py  –  résumé FR fiable
from __future__ import annotations
from transformers import pipeline
import language_tool_python, unicodedata, re, logging

_SUM_MODEL = "plguillou/t5-base-fr-sum-cnndm"   # modèle FR dispo sur HF
_TOOL      = language_tool_python.LanguageTool("fr")  # correcteur grammaire

def _clean(txt: str) -> str:
    """Nettoyage minimal : accents, ponctuation, espaces."""
    txt = unicodedata.normalize("NFKC", txt)
    txt = re.sub(r"[^\w\s.,;:!?()€%/-]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()

class DocumentSummarizer:
    """Génère un résumé *en français* tout en corrigeant la grammaire."""

    def __init__(self, model_name: str | None = None, device: int = -1):
        model_name = model_name or _SUM_MODEL
        try:
            self.pipe = pipeline(
                task="summarization",
                model=model_name,
                tokenizer=model_name,
                device=device
            )
        except Exception as e:
            logging.error(f"Impossible de charger le modèle {model_name}: {e}")
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
            # 1) pré‑nettoyage
            txt = _clean(text)

            # 2) préfixe requis par T5
            if not txt.lower().startswith("summarize:"):
                txt = "summarize: " + txt

            # 3) résumé
            raw = self.pipe(
                txt,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                clean_up_tokenization_spaces=True,
            )[0]["summary_text"]

            # 4) relecture grammaticale
            return _TOOL.correct(raw).strip()

        except Exception as e:
            logging.error(f"Erreur de résumé: {e}")
            return f"⚠️ Erreur lors du résumé : {e}"
