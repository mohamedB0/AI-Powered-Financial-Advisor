import os
import glob
from pathlib import Path
from pdf_extractor import AdvancedPDFPipeline

def get_sample_pdf() -> str:
    # Try finding any PDF in the data directories
    search_patterns = [
        "../data/**/*.pdf",
        "../data/docs/*.pdf",
        "../data/collected_data/**/*.pdf"
    ]
    
    for pattern in search_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
            
    return ""

def main():
    print("=============================================")
    print("  Advanced OCR & Vision Pipeline Test Run ")
    print("=============================================")
    
    sample_pdf = get_sample_pdf()
    
    if not sample_pdf:
        print("Error: Could not find a sample PDF in the ../data/ directory.")
        print("Please place a sample PDF in OCR_Processing/ or update the path.")
        return

    print(f"Sample PDF found: {sample_pdf}")
    print("Initializing API models and PyMuPDF...")
    
    pipeline = AdvancedPDFPipeline()
    
    try:
        output_file = pipeline.process_pdf(sample_pdf)
        print(f"\nSUCCESS! Markdown output generated at:\n{output_file}")
    except Exception as e:
        print(f"\nERROR: Pipeline execution failed:\n{e}")

if __name__ == "__main__":
    main()
