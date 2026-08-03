import fitz  # PyMuPDF
import logging
from pathlib import Path
try:
    from OCR_Processing.vision_client import VisionAnalyzer
except ImportError:
    from vision_client import VisionAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _page_has_embedded_images(doc, page) -> bool:
    """Check if a page contains raster images bigger than a trivial icon."""
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        base_image = doc.extract_image(xref)
        if len(base_image["image"]) > 5000:
            return True
    return False


class AdvancedPDFPipeline:
    """
    Advanced PDF Processor:
    - Extracts native text.
    - Uses Vision OCR for scanned pages without text.
    - For pages with both text AND visuals, renders the full page so the vision
      model sees charts/graphics in context with surrounding text.
    - Extracts standalone embedded images and generates descriptions.
    - Outputs a structured Markdown file ready for RAG ingestion.
    """
    def __init__(self):
        self.vision = VisionAnalyzer()
        self.output_dir = Path("OCR_Processing/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _extract_images_standalone(self, doc, page) -> list[str]:
        """Extract and describe individual raster images from a page (no surrounding text context)."""
        sections = []
        images = page.get_images(full=True)
        for img_idx, img_info in enumerate(images):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            if len(image_bytes) > 5000:
                logger.info(f"  Describing standalone image {img_idx + 1}...")
                description = self.vision.describe_chart_or_figure(image_bytes, image_ext=image_ext)
                sections.append(
                    f"\n### Visual Element {img_idx + 1} (Chart/Figure)\n"
                    f"> **AI Vision Description:**\n> {description.replace(chr(10), chr(10) + '> ')}\n"
                )
        return sections

    def process_pdf(self, pdf_path: str) -> str:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Starting advanced extraction on: {pdf_path.name}")
        doc = fitz.open(pdf_path)

        markdown_content = []
        markdown_content.append(f"# Document Metadata: {pdf_path.name}\n")
        markdown_content.append(f"- **Pages**: {len(doc)}")
        markdown_content.append(f"- **Title**: {doc.metadata.get('title', 'Unknown')}")
        markdown_content.append(f"- **Author**: {doc.metadata.get('author', 'Unknown')}\n\n---\n")

        for page_num in range(len(doc)):
            page = doc[page_num]
            logger.info(f"Processing page {page_num + 1}/{len(doc)}")

            markdown_content.append(f"## Page {page_num + 1}\n")

            native_text = page.get_text("text").strip()
            has_images = _page_has_embedded_images(doc, page)

            # --- Strategy selection ---------------------------------------------------
            if len(native_text) < 50 and not has_images:
                # SCANNED PAGE — no usable text, no images → full-page OCR
                logger.info(f"Page {page_num + 1}: scanned/blank. Full-page Vision OCR.")
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("png")
                ocr_text = self.vision.ocr_scanned_page(img_bytes)
                markdown_content.append("*(Scanned Page / OCR Extraction)*\n")
                markdown_content.append(ocr_text + "\n")

            elif len(native_text) >= 50 and has_images:
                # TEXT + VISUALS — render full page so the vision model sees context
                logger.info(f"Page {page_num + 1}: text + visuals detected. Full-page contextual extraction.")
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("png")
                prompt = (
                    "This is a page from a financial document. It contains both text and visual elements "
                    "(charts, graphs, tables, diagrams, or figures).\n\n"
                    "Extract ALL content from this page in order:\n"
                    "1. Transcribe all text paragraphs exactly as written.\n"
                    "2. For each chart/graph/figure, describe it in detail: axes, labels, data points, "
                    "values, trends, legend. If it contains a table, recreate it in Markdown.\n"
                    "3. IMPORTANT: Explain how each visual relates to the surrounding text. "
                    "If a paragraph references a specific chart, note that connection.\n\n"
                    "Format the output as clean Markdown. Be exhaustive — capture every number and label."
                )
                full_page_text = self.vision.analyze_image(img_bytes, prompt, image_ext="png")
                markdown_content.append(full_page_text + "\n")

            elif len(native_text) >= 50:
                # TEXT ONLY — no images
                logger.info(f"Page {page_num + 1}: text only.")
                markdown_content.append(native_text + "\n")

            else:
                # MINIMAL TEXT + IMAGES — render full page for best extraction
                logger.info(f"Page {page_num + 1}: minimal text with images. Full-page Vision extraction.")
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("png")
                prompt = (
                    "This is a page from a financial document. Extract ALL content:\n"
                    "- Transcribe all text exactly as written.\n"
                    "- For charts/graphs/tables, reproduce them in detail with Markdown.\n"
                    "- Describe data trends, key numbers, labels, and values.\n"
                    "Be exhaustive. Output clean Markdown."
                )
                full_page_text = self.vision.analyze_image(img_bytes, prompt, image_ext="png")
                markdown_content.append(full_page_text + "\n")

            markdown_content.append("\n---\n")

        doc.close()

        # Save output Markdown
        output_file = self.output_dir / f"{pdf_path.stem}_extracted.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(markdown_content))

        logger.info(f"Extraction complete! Saved structured Markdown to: {output_file}")
        return str(output_file)
