import os
import sys

# We'll use one of the smallest PDFs we scraped from AMMC
pdf_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    "../data/collected_data/documents/ammc/ammc_RPI_T2_2018pdf_b3562806.pdf"
))
output_path = "output_extracted.md"

print(f"Testing PyMuPDF on: {pdf_path}")
if not os.path.exists(pdf_path):
    print("Error: The sample PDF was not found. Please make sure the scraper has downloaded AMMC data.")
    sys.exit(1)

try:
    # Attempt to use the dedicated pymupdf4llm markdown extractor (highly recommended for AI)
    import pymupdf4llm
    print("Using 'pymupdf4llm' for advanced Markdown extraction...")
    md_text = pymupdf4llm.to_markdown(pdf_path)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Successfully saved beautifully formatted Markdown to: {output_path}")

except Exception as e:
    print(f"Warning: 'pymupdf4llm' failed (error: {e}).")
    print("Falling back to standard PyMuPDF (fitz)...")
    
    import fitz  # Standard PyMuPDF
    doc = fitz.open(pdf_path)
    md_text = f"# Extracted Text from {os.path.basename(pdf_path)}\n\n"
    
    for page_num, page in enumerate(doc):
        # Some newer versions of fitz support "markdown", otherwise fallback to standard text
        try:
            text = page.get_text("markdown")
        except Exception:
            text = page.get_text("text")
            
        md_text += f"\n\n## Page {page_num + 1}\n\n"
        md_text += text
        
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Successfully extracted {len(doc)} pages and saved to: {output_path}")
