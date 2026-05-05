import os
import base64
import requests
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class VisionAnalyzer:
    """
    Uses OpenRouter's vision capabilities to describe images, charts, and OCR scanned text.
    """
    def __init__(self, model_name="openai/gpt-4o-mini"):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model_name = model_name
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not found in env. Vision analysis will fail.")

    def _encode_image(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode('utf-8')

    def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        """Sends image bytes to OpenRouter Vision Model and returns the markdown description."""
        if not self.api_key:
            return "*[Vision API Key missing - Cannot analyze image]*"
        
        base64_img = self._encode_image(image_bytes)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                    ]
                }
            ],
            "temperature": 0.1
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Vision API Error: {e}")
            return f"*[Error analyzing image: {e}]*"

    def ocr_scanned_page(self, image_bytes: bytes) -> str:
        """Fallback for fully scanned pages without native text."""
        prompt = (
            "You are an expert OCR system. Extract all text from this page EXACTLY as written. "
            "If there are tables, format them properly in Markdown. Do not add conversational filler."
        )
        return self.analyze_image(image_bytes, prompt)

    def describe_chart_or_figure(self, image_bytes: bytes) -> str:
        """Describes an embedded chart/diagram."""
        prompt = (
            "This is a chart, graph, or figure extracted from a financial document. "
            "Describe the data trends, key numbers, labels, and the overall meaning of this visual. "
            "Format your response concisely. If there is a table, recreate it in Markdown."
        )
        return self.analyze_image(image_bytes, prompt)
