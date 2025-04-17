import pytesseract
from PIL import Image
import tempfile
import os
from pdf2image import convert_from_path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class OCRProcessor:
    """
    Classe pour extraire le texte d'un fichier image ou PDF en utilisant OCR.
    Nécessite : pytesseract, pillow, pdf2image et poppler (pour pdf2image).
    """
    def __init__(self):
        pass

    def extract_text_from_image(self, image):
        """Extrait le texte d'une image PIL."""
        return pytesseract.image_to_string(image)

    def extract_text_from_pdf(self, file):
        """
        Extrait le texte d'un fichier PDF.
        Le fichier est sauvegardé temporairement, converti en images puis traité.
        """
        # Sauvegarde du PDF dans un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        # Conversion du PDF en images (une image par page)
        pages = convert_from_path(tmp_path)
        text = ""
        for page in pages:
            text += self.extract_text_from_image(page) + "\n"

        # Suppression du fichier temporaire
        os.remove(tmp_path)
        return text

    def extract_text(self, file, file_extension):
        """
        Extrait le texte du fichier en fonction de son extension.
        file : objet file-like (par exemple un BytesIO)
        file_extension : extension du fichier (ex: "pdf", "jpg", "png", etc.)
        """
        file.seek(0)  # Réinitialiser le curseur
        if file_extension.lower() in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
            image = Image.open(file)
            return self.extract_text_from_image(image)
        elif file_extension.lower() == 'pdf':
            return self.extract_text_from_pdf(file)
        else:
            raise ValueError("Format de fichier non supporté : " + file_extension)
