import os
import base64
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

EXT_TO_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
    "tif": "image/tiff",
    "webp": "image/webp",
    "jp2": "image/jp2",
}


class VisionAnalyzer:
    """
    Uses OpenRouter (primary) or HuggingFace Inference API (fallback)
    for OCR and image analysis. Runs on cloud infrastructure (no local GPU needed).
    """

    HF_MODEL_ID = "google/gemma-4-31B-it"
    OPENROUTER_MODEL = "qwen/qwen-2.5-vl-72b-instruct"

    def __init__(self):
        self.hf_key = os.getenv("HUGGINGFACE_Token")
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not self.hf_key:
            logger.warning("HUGGINGFACE_Token not found in env.")
        if not self.openrouter_key:
            logger.warning("OPENROUTER_API_KEY not found in env.")

    def _encode_image(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")

    @staticmethod
    def _guess_mime(image_bytes: bytes, ext: str | None = None) -> str:
        if ext:
            normalised = ext.lower().lstrip(".")
            if normalised in EXT_TO_MIME:
                return EXT_TO_MIME[normalised]
        if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if image_bytes[:2] == b"\xff\xd8":
            return "image/jpeg"
        if image_bytes[:4] == b"GIF8":
            return "image/gif"
        if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            return "image/webp"
        return "image/png"

    def _call_openrouter(self, image_bytes: bytes, prompt: str, mime: str, max_tokens: int = 2048) -> str | None:
        """Send image + prompt to OpenRouter. Returns None on failure."""
        if not self.openrouter_key:
            return None

        import httpx

        b64 = self._encode_image(image_bytes)
        payload = {
            "model": self.OPENROUTER_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                }
            ],
            "max_tokens": max_tokens,
        }
        try:
            resp = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"OpenRouter Vision API Error: {e}")
            return None

    def _call_hf_api(self, image_bytes: bytes, prompt: str, mime: str, max_tokens: int = 2048) -> str | None:
        """Send image + prompt to HuggingFace Inference API. Returns None on failure."""
        if not self.hf_key:
            return None

        from huggingface_hub import InferenceClient

        client = InferenceClient(token=self.hf_key)
        b64 = self._encode_image(image_bytes)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        ]
        try:
            response = client.chat_completion(
                model=self.HF_MODEL_ID,
                messages=messages,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"HuggingFace Vision API Error: {e}")
            return None

    def _call_vision(self, image_bytes: bytes, prompt: str, mime: str, max_tokens: int = 2048) -> str:
        """Try OpenRouter first, then fall back to HuggingFace."""
        result = self._call_openrouter(image_bytes, prompt, mime, max_tokens)
        if result is not None:
            return result
        logger.info("Falling back to HuggingFace Inference API...")
        result = self._call_hf_api(image_bytes, prompt, mime, max_tokens)
        if result is not None:
            return result
        return "*[All vision API providers failed — check API keys and credits]*"

    def analyze_image(self, image_bytes: bytes, prompt: str, image_ext: str | None = None) -> str:
        """Public interface — sends image bytes to a vision LLM and returns text."""
        mime = self._guess_mime(image_bytes, image_ext)
        return self._call_vision(image_bytes, prompt, mime)

    def ocr_scanned_page(self, image_bytes: bytes) -> str:
        """OCR for fully scanned pages without native text."""
        prompt = (
            "You are an expert OCR system. Extract all text from this page EXACTLY as written. "
            "If there are tables, format them properly in Markdown. "
            "Do not add conversational filler."
        )
        return self.analyze_image(image_bytes, prompt, image_ext="png")

    def describe_chart_or_figure(self, image_bytes: bytes, image_ext: str | None = None) -> str:
        """Describes an embedded chart or diagram."""
        prompt = (
            "This is a chart, graph, or figure extracted from a financial document. "
            "Describe the data trends, key numbers, labels, and the overall meaning of this visual. "
            "Format your response concisely. If there is a table, recreate it in Markdown."
        )
        return self.analyze_image(image_bytes, prompt, image_ext=image_ext)
