# document_type_detector.py – v4.1 (examen prioritaire)

from transformers import pipeline, AutoTokenizer
import torch, functools, re, collections

_MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
_HYPOTPL  = "Ce document est un(e) {}."

@functools.lru_cache
def _classifier(dev:int):
    return pipeline("zero-shot-classification",
                    model=_MODEL_ID,
                    device=dev,
                    framework="pt",
                    hypothesis_template=_HYPOTPL,
                    truncation=True)

@functools.lru_cache
def _tok():
    return AutoTokenizer.from_pretrained(_MODEL_ID)

# ────────────────────────────────────────────────────────────────────
class DocumentTypeDetector:
    LABELS = [
        "document d'identité", "passeport", "facture", "relevé bancaire",
        "contrat", "lettre", "document juridique", "sujet d'évaluation"
    ]

    ALIASES = {
        # … identiques à v4.0, sauf sujet d’évaluation enrichi …
        "document d'identité": [
            "document d'identité", "pièce d'identité", "carte d'identité",
            "identity document", "id card", "national id"
        ],
        "passeport": ["passeport", "passport", "passeport biométrique",
                      "passeport électronique", "travel passport"],
        "facture": ["facture", "factures", "invoice", "bill",
                    "utility bill", "statement of charges"],
        "relevé bancaire": ["relevé bancaire", "relevé de compte", "bank statement",
                            "account statement", "statement bancaire", "iban",
                            "rib", "bic", "solde", "débit", "crédit"],
        "contrat": ["contrat", "contract", "avenant", "cdi", "cdd",
                    "type de contrat", "date de conclusion", "date de fin",
                    "durée hebdomadaire", "rémunération", "smic", "smc",
                    "salaire brut", "signature de l'employeur"],
        "lettre": ["lettre", "courrier", "correspondance", "letter",
                   "mail letter", "personal letter",
                   "madame", "monsieur", "objet", "cordialement",
                   "à l'attention"],
        "document juridique": ["document juridique", "document légal",
                               "legal document", "legal paperwork",
                               "legal agreement", "legal form",
                               "autorisation", "droit à l'image", "tribunal",
                               "article", "loi", "règlement", "responsabilité"],
        "sujet d'évaluation": ["sujet d'évaluation", "exam subject", "test paper",
                               "exam question", "fiche d'évaluation",
                               "épreuve", "examen", "note", "notes",
                               "session", "barème", "jury", "oral", "dnb",
                               "classe", "points"],
    }
    CANDIDATES = sorted({w for v in ALIASES.values() for w in v})

    # hyper‑params
    WINDOW, OVERLAP = 400, 100
    KW = 0.25
    BOOST_INV       = 1.3
    BOOST_EXAM      = 2.0   # ↑
    BOOST_BANK      = 1.8
    BOOST_CONTRAT   = 1.7
    BOOST_LETTER    = 2.0
    BOOST_JUR       = 1.8

    PEN_EXAM_FEW    = 0.5   # exam mais 0–1 marqueur
    PEN_LET_NO_MARK = 0.5
    PEN_LET_BANK    = 0.5
    PEN_LET_CONTR   = 0.4
    PEN_CONTR_JUR   = 0.5

    # regex
    _rx_art = re.compile(r"[|_—]+")
    _rx_spc = re.compile(r"\s+")
    _rx_w   = re.compile(r"\b\w+\b", re.I)

    _rx_let  = re.compile(r"\b(madame|monsieur|objet|cordialement|à l'attention)\b", re.I)
    _rx_bank = re.compile(r"\b(iban|rib|bic|solde|débit|crédit|relevé de compte)\b", re.I)
    _rx_ctr  = re.compile(r"\b(contrat|avenant|cdi|cdd|smic|smc|salaire|date de conclusion|type de contrat)\b", re.I)
    _rx_jur  = re.compile(r"\b(autorisation|tribunal|juridiction|article|loi|règlement|responsabilité)\b", re.I)
    _rx_exam = re.compile(r"\b(examen|épreuve|fiche d'évaluation|bar[eè]me|note[s]?|session|jury|oral|dnb)\b", re.I)

    # ----------------------------------------------------------------
    def __init__(self, device=None):
        device = 0 if (device is None and torch.cuda.is_available()) else -1
        self.clf = _classifier(device)
        self.tok = _tok()

    def _clean(self, t:str) -> str:
        return self._rx_spc.sub(" ", self._rx_art.sub(" ", t)).replace("’", "'").lower().strip()

    # heuristique
    def _adj(self, lbl, chunk, s):
        if lbl in self.ALIASES["facture"]:
            return s * self.BOOST_INV

        if lbl in self.ALIASES["relevé bancaire"]:
            return s * self.BOOST_BANK if len(self._rx_bank.findall(chunk)) >= 2 else s

        if lbl in self.ALIASES["contrat"]:
            nb_c, nb_j = len(self._rx_ctr.findall(chunk)), len(self._rx_jur.findall(chunk))
            if nb_j >= 3 and nb_c <= 1:
                return s * self.PEN_CONTR_JUR
            if nb_c >= 2:
                return s * self.BOOST_CONTRAT
            return s

        if lbl in self.ALIASES["document juridique"]:
            return s * self.BOOST_JUR if len(self._rx_jur.findall(chunk)) >= 2 else s

        if lbl in self.ALIASES["sujet d'évaluation"]:
            nb_e = len(self._rx_exam.findall(chunk))
            return s * self.BOOST_EXAM if nb_e >= 2 else s * self.PEN_EXAM_FEW

        if lbl in self.ALIASES["lettre"]:
            nb_l = len(self._rx_let.findall(chunk))
            if nb_l == 0:
                return s * self.PEN_LET_NO_MARK
            if len(self._rx_bank.findall(chunk)) >= 2:
                return s * self.PEN_LET_BANK
            if len(self._rx_ctr.findall(chunk)) >= 2:
                return s * self.PEN_LET_CONTR
            if nb_l >= 2:
                return s * self.BOOST_LETTER
            return s
        return s

    # API
    def detect_document_type(self, text:str):
        if not text.strip():
            return []

        txt = self._clean(text)
        ids = self.tok(txt, add_special_tokens=False)["input_ids"]
        raw = collections.defaultdict(float)

        for i in range(0, len(ids), self.WINDOW - self.OVERLAP):
            chunk = self.tok.decode(ids[i:i+self.WINDOW], skip_special_tokens=True)
            words = set(self._rx_w.findall(chunk))
            kw = {a: len(words & {a}) * self.KW for a in self.CANDIDATES}

            out = self.clf(chunk, self.CANDIDATES, multi_label=False)
            for lbl, sc in zip(out["labels"], out["scores"]):
                raw[lbl] += self._adj(lbl, chunk, sc + kw.get(lbl, 0))

        scores = [(c, sum(raw[a] for a in alias)) for c, alias in self.ALIASES.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        tot = sum(s for _, s in scores) or 1
        return [(l, s / tot) for l, s in scores]

    __call__ = detect_document_type
