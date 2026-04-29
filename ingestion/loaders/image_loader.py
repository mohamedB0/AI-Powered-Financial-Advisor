"""
ingestion/loaders/image_loader.py
Extracts text from images using OCR (pytesseract).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import pytesseract
    from PIL import Image
    _HAS_OCR = True
except ImportError:
    _HAS_OCR = False


class ImageLoader:
    """
    Loads images and extracts text using OCR.
    
    Requires Tesseract OCR engine installed on the system.
    Download from: https://github.com/UB-Mannheim/tesseract/wiki
    """

    def __init__(self, lang: str = "eng"):
        self.lang = lang

    def load(self, file_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Extract text from an image using OCR.
        """
        if not _HAS_OCR:
            raise ImportError("pytesseract and Pillow required: pip install pytesseract pillow")

        path = Path(file_path)
        
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image, lang=self.lang)
        except Exception as e:
            logger.error("ImageLoader: OCR failed for '%s': %s", path.name, e)
            return []

        if not text.strip():
            logger.warning("ImageLoader: No text extracted from '%s'", path.name)
            return []

        # Get image metadata
        width, height = image.size
        
        metadata = {
            "source": path.name,
            "doc_type": "image",
            "file_path": str(path),
            "image_width": width,
            "image_height": height,
        }

        logger.info("ImageLoader: extracted %d chars from '%s'", len(text), path.name)
        return [(text.strip(), metadata)]

    def load_with_boxes(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract text with bounding boxes for more detailed analysis.
        """
        if not _HAS_OCR:
            raise ImportError("pytesseract and Pillow required")

        image = Image.open(file_path)
        data = pytesseract.image_to_data(image, lang=self.lang, output_type=pytesseract.Output.DICT)
        
        results = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            if data['text'][i].strip():
                results.append({
                    "text": data['text'][i],
                    "confidence": data['conf'][i],
                    "bbox": {
                        "x": data['left'][i],
                        "y": data['top'][i],
                        "width": data['width'][i],
                        "height": data['height'][i],
                    }
                })
        
        return results
