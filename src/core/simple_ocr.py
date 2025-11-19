"""
Simple OCR Module (No YOLO Required)

Fallback OCR implementation that works without YOLO dependencies.
Uses Tesseract directly on full pages.

This is used when YOLO dependencies are not available.

Author: Classmark Development Team
Date: 2025-11-10
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Import PDF image cache and poppler path
try:
    from .classification_scanner import PDFImageCache, POPPLER_PATH
    HAS_IMAGE_CACHE = True
except ImportError:
    HAS_IMAGE_CACHE = False
    POPPLER_PATH = None

# Check for basic OCR dependencies
try:
    from PIL import Image
    import pytesseract
    from pdf2image import convert_from_path
    HAS_SIMPLE_OCR = True
except ImportError:
    HAS_SIMPLE_OCR = False


def extract_text_from_pdf_simple_ocr(
    pdf_path: str,
    dpi: int = 300,
    max_pages: Optional[int] = 10
) -> Tuple[str, bool, float]:
    """
    Extract text from PDF using simple OCR (no YOLO)

    Args:
        pdf_path: Path to PDF file
        dpi: DPI for image conversion
        max_pages: Maximum pages to process

    Returns:
        (extracted_text, success, processing_time)
    """
    if not HAS_SIMPLE_OCR:
        return "", False, 0.0

    start_time = time.time()

    try:
        print(f"Converting PDF to images (DPI: {dpi}, max pages: {max_pages})...")

        # Convert PDF to images (use cache if available)
        if HAS_IMAGE_CACHE:
            images = PDFImageCache.get_or_convert(pdf_path, dpi=dpi, max_pages=max_pages)
        else:
            # Use bundled poppler if available
            kwargs = {'dpi': dpi}
            if POPPLER_PATH:
                kwargs['poppler_path'] = POPPLER_PATH
            images = convert_from_path(pdf_path, **kwargs)
            # Limit pages
            if max_pages and len(images) > max_pages:
                print(f"Limiting to first {max_pages} pages")
                images = images[:max_pages]

        print(f"✓ Converted {len(images)} pages to images")

        # OCR each page
        all_text = []
        for i, image in enumerate(images, 1):
            print(f"OCR page {i}/{len(images)}...", end=' ')

            # Run Tesseract
            text = pytesseract.image_to_string(image, config='--psm 1')

            word_count = len(text.split())
            print(f"✓ {word_count} words")

            if text.strip():
                all_text.append(f"[Page {i}]\n{text}")

        combined_text = '\n\n'.join(all_text)
        processing_time = time.time() - start_time

        print(f"✓ OCR complete: {len(combined_text.split())} total words in {processing_time:.2f}s")

        return combined_text, True, processing_time

    except Exception as e:
        print(f"⚠ OCR error: {e}")
        processing_time = time.time() - start_time
        return "", False, processing_time


def should_use_ocr(text: str, word_threshold: int = 50) -> bool:
    """
    Determine if OCR should be used based on extracted text

    Args:
        text: Text extracted by standard methods
        word_threshold: Minimum word count to skip OCR

    Returns:
        True if OCR should be used
    """
    word_count = len(text.split())
    return word_count < word_threshold


if __name__ == "__main__":
    print("Simple OCR Module (No YOLO)")
    print("=" * 60)
    print(f"Dependencies available: {HAS_SIMPLE_OCR}")

    if HAS_SIMPLE_OCR:
        print("\nThis module provides basic OCR without YOLO.")
        print("For SOTA performance, install full stack: pip install torch ultralytics")
    else:
        print("\nInstall dependencies: pip install pytesseract Pillow pdf2image")
