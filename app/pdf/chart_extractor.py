"""Basic chart/image extraction from RBA PDF documents.

RBA reports contain critical visual data:
- Economic charts (GDP growth, inflation trends)
- Historical comparison graphs
- Risk distribution charts
- International benchmarking visualizations

Why extract charts?
1. Metadata enrichment: Flag chunks containing charts for better retrieval
2. Future multimodal RAG: Extract images for vision LLM processing
3. Context preservation: Charts complement tabular data

Current approach (basic):
- Detect large images (>200x150px) as chart candidates
- Extract image metadata (dimensions, position, format)
- Optional: Save chart images to MinIO for future use

Future enhancements:
- OCR on chart text (axis labels, legends)
- Vision LLM to extract chart data
- Chart type classification (bar, line, pie, etc.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class ChartMetadata:
    """Metadata for an extracted chart/image."""

    page_number: int
    image_index: int  # Index among images on this page
    width: int
    height: int
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    format: str | None  # e.g., "png", "jpeg"
    xref: int  # PyMuPDF cross-reference ID


class ChartExtractor:
    """Extract charts and large images from PDF pages.

    Simple heuristic-based approach:
    - Detect large images (likely charts, not icons/logos)
    - Extract image bytes and metadata
    - Filter by size threshold to avoid small decorative elements

    Limitations:
    - Heuristic-based (not ML classification)
    - May count photos as charts if large
    - Won't detect vector-based charts (rare in RBA PDFs)
    """

    def __init__(self, min_width: int = 200, min_height: int = 150, extract_images: bool = False):
        """Initialize chart extractor.

        Args:
            min_width: Minimum image width in pixels (default: 200)
            min_height: Minimum image height in pixels (default: 150)
            extract_images: Whether to extract actual image bytes (default: False)
                           Set True if you want to save images to storage
        """
        self.min_width = min_width
        self.min_height = min_height
        self.extract_images = extract_images

    def detect_charts(self, page: fitz.Page) -> List[ChartMetadata]:
        """Detect charts and large images on a PDF page.

        Args:
            page: PyMuPDF Page object

        Returns:
            List of ChartMetadata for detected charts

        Detection logic:
        - Extract all images from page
        - Filter by size (width >= min_width AND height >= min_height)
        - Typical RBA chart: 400x300+ pixels
        - Filters out: small icons (<200px), logos, decorative elements
        """
        try:
            image_list = page.get_images(full=True)
            charts: List[ChartMetadata] = []

            for img_index, img_info in enumerate(image_list):
                # img_info structure (PyMuPDF):
                # (xref, smask, width, height, bpc, colorspace, alt_colorspace,
                #  name, filter, referencer)
                xref = img_info[0]
                width = img_info[2]
                height = img_info[3]

                # Filter by size threshold
                if width < self.min_width or height < self.min_height:
                    logger.debug(
                        f"Skipping small image on page {page.number + 1}: "
                        f"{width}x{height}px (below threshold)"
                    )
                    continue

                # Get image bounding box on page
                # Multiple instances of same image may exist; get first occurrence
                bbox = None
                try:
                    img_rects = page.get_image_rects(xref)
                    if img_rects:
                        # Use first rectangle
                        rect = img_rects[0]
                        bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                except Exception as bbox_error:
                    logger.warning(
                        f"Failed to get bbox for image on page {page.number + 1}: {bbox_error}"
                    )
                    # Use default bbox (full page dimensions)
                    bbox = (0, 0, page.rect.width, page.rect.height)

                # Get image format if available
                img_format = None
                try:
                    img_dict = page.parent.extract_image(xref)
                    img_format = img_dict.get("ext", "png")  # default to png
                except Exception as format_error:
                    logger.debug(f"Could not determine image format: {format_error}")

                charts.append(
                    ChartMetadata(
                        page_number=page.number + 1,  # 1-based page numbering
                        image_index=img_index,
                        width=width,
                        height=height,
                        bbox=bbox,
                        format=img_format,
                        xref=xref,
                    )
                )

                logger.info(
                    f"Detected chart on page {page.number + 1}: "
                    f"{width}x{height}px, format={img_format}"
                )

            return charts

        except Exception as e:
            logger.warning(f"Chart detection failed for page {page.number + 1}: {e}")
            return []

    def extract_chart_image(self, doc: fitz.Document, chart: ChartMetadata) -> bytes | None:
        """Extract actual image bytes for a chart.

        Args:
            doc: PyMuPDF Document object
            chart: ChartMetadata with xref

        Returns:
            Image bytes (PNG/JPEG) or None if extraction fails

        Use case:
        - Save chart images to MinIO for future multimodal RAG
        - Vision LLM can analyze chart content
        - OCR on chart text (axis labels, legends, values)
        """
        if not self.extract_images:
            return None

        try:
            img_dict = doc.extract_image(chart.xref)
            return img_dict["image"]

        except Exception as e:
            logger.warning(f"Failed to extract image for chart on page {chart.page_number}: {e}")
            return None

    def extract_page_charts(self, pdf_path: Path, page_num: int) -> List[ChartMetadata]:
        """Extract all charts from a specific PDF page.

        Args:
            pdf_path: Path to PDF file
            page_num: 1-based page number

        Returns:
            List of ChartMetadata for detected charts

        Convenience method combining page loading + chart detection.
        """
        try:
            with fitz.open(pdf_path) as doc:
                # PyMuPDF uses 0-based indexing
                page = doc[page_num - 1]
                return self.detect_charts(page)

        except Exception as e:
            logger.error(f"Failed to extract charts from page {page_num}: {e}")
            return []
