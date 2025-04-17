# ocr_processor.py – extraction de texte avec EasyOCR

import io
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes
import easyocr
import torch

class OCRProcessor:
    """
    Classe pour extraire le texte d'un fichier image ou PDF en utilisant EasyOCR.
    Nécessite : easyocr, pillow, pdf2image (et poppler pour PDF).
    """

    def __init__(self, langs: list[str] | None = None):
        """
        langs : liste de codes de langue pour EasyOCR (ex. ['fr','en']).
        """
        langs = langs or ["fr", "en"]
        # GPU si disponible, sinon CPU
        use_gpu = torch.cuda.is_available()
        self.reader = easyocr.Reader(langs, gpu=use_gpu)

    def extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extrait le texte d'une image PIL.Image via EasyOCR.
        """
        img = image.convert("RGB")
        arr = np.array(img)
        # detail=0 => on ne récupère que les chaînes
        lines = self.reader.readtext(arr, detail=0)
        return "\n".join(lines)

    def extract_text_from_pdf(self, file_obj: io.BytesIO) -> str:
        """
        Extrait le texte d'un PDF :
        - Lit les bytes
        - Convertit en images (une page = une image)
        - OCR page par page
        """
        file_obj.seek(0)
        pdf_bytes = file_obj.read()
        pages = convert_from_bytes(pdf_bytes)
        all_text = []
        for page in pages:
            all_text.append(self.extract_text_from_image(page))
        return "\n\n".join(all_text)

    def extract_text(self, file_obj: io.BytesIO, file_extension: str) -> str:
        """
        Routeur selon l'extension.
        file_extension : 'pdf', 'jpg', 'png', etc.
        """
        ext = file_extension.lower()
        if ext in ["jpg", "jpeg", "png", "bmp", "tiff", "tif"]:
            image = Image.open(file_obj)
            return self.extract_text_from_image(image)
        elif ext == "pdf":
            return self.extract_text_from_pdf(file_obj)
        else:
            raise ValueError(f"Format non supporté : {file_extension}")
