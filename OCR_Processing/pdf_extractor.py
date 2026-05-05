import fitz  # PyMuPDF
import logging
from pathlib import Path
try:
    from OCR_Processing.vision_client import VisionAnalyzer
except ImportError:
    from vision_client import VisionAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedPDFPipeline:
    """
    Advanced PDF Processor:
    - Extracts native text.
    - Uses Vision OCR for scanned pages without text.
    - Extracts embedded images/charts and generates descriptions using a Vision LLM.
    - Outputs a structured Markdown file ready for RAG ingestion.
    """
    def __init__(self):
        self.vision = VisionAnalyzer()
        self.output_dir = Path("OCR_Processing/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
            
            # 1. Native text extraction
            native_text = page.get_text("text").strip()
            
            # Strategy: If page has virtually no text, treat it as a scanned image.
            if len(native_text) < 50:
                logger.info(f"Page {page_num + 1} has limited native text. Using Vision OCR.")
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                ocr_text = self.vision.ocr_scanned_page(img_bytes)
                markdown_content.append("*(Scanned Page / OCR Extraction)*\n")
                markdown_content.append(ocr_text + "\n")
            else:
                # 2. Add native text
                markdown_content.append(native_text + "\n")
                
                # 3. Extract and describe embedded images / charts
                images = page.get_images(full=True)
                if images:
                    logger.info(f"Found {len(images)} images on page {page_num + 1}.")
                    for img_idx, img_info in enumerate(images):
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Only process reasonably sized images (ignore tiny logos/icons)
                        if len(image_bytes) > 5000:
                            logger.info(f"Generating caption for image {img_idx + 1}...")
                            description = self.vision.describe_chart_or_figure(image_bytes)
                            
                            markdown_content.append(f"\n###  Visual Element {img_idx + 1} (Chart/Figure)\n")
                            markdown_content.append(f"> **AI Vision Description:**\n> {description.replace(chr(10), chr(10) + '> ')}\n")

            markdown_content.append("\n---\n")

        doc.close()
        
        # Save output Markdown
        output_file = self.output_dir / f"{pdf_path.stem}_extracted.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(markdown_content))
            
        logger.info(f"Extraction complete! Saved structured Markdown to: {output_file}")
        return str(output_file)
