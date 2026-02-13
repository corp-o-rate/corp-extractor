"""
GLM-OCR PDF parser plugin using the GLM-OCR 0.9B vision-language model.

Renders PDF pages to images and uses GLM-OCR for high-quality text extraction
with markdown formatting. Excels at image-heavy PDFs, scanned documents,
tables, and formulas.

Based on https://github.com/zai-org/GLM-OCR — ported to run in-process via
HuggingFace transformers (no external inference server needed).
"""

import logging
import math
import re
from collections import Counter
from typing import Any, Optional

from ..base import BasePDFParserPlugin, PDFParseResult
from ...pipeline.registry import PluginRegistry

logger = logging.getLogger(__name__)

# OCR prompt from GLM-OCR config.yaml (default full-page prompt)
_OCR_PROMPT = (
    "Recognize the text in the image and output in Markdown format. "
    "Preserve the original layout (headings/paragraphs/tables/formulas). "
    "Do not fabricate content that does not exist in the image."
)


# ---------------------------------------------------------------------------
# Ported utilities from glmocr
# ---------------------------------------------------------------------------


def _smart_resize(
    h: int,
    w: int,
    h_factor: int = 28,
    w_factor: int = 28,
    min_pixels: int = 112 * 112,
    max_pixels: int = 14 * 14 * 4 * 15000,
) -> tuple[int, int]:
    """Resize image dimensions for optimal VLM input.

    Ensures height and width are divisible by the model's patch factors (28)
    and total pixels stay within bounds, preserving aspect ratio.

    Ported from glmocr/utils/image_utils.py:smart_resize().
    """
    h_bar = round(h / h_factor) * h_factor
    w_bar = round(w / w_factor) * w_factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = math.floor(h / beta / h_factor) * h_factor
        w_bar = math.floor(w / beta / w_factor) * w_factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / h_factor) * h_factor
        w_bar = math.ceil(w * beta / w_factor) * w_factor

    return h_bar, w_bar


def _find_consecutive_repeat(
    s: str,
    min_unit_len: int = 10,
    min_repeats: int = 10,
) -> Optional[str]:
    """Detect and remove consecutive repeated patterns in text.

    Ported from glmocr/utils/result_postprocess_utils.py:find_consecutive_repeat().
    """
    n = len(s)
    if n < min_unit_len * min_repeats:
        return None
    max_unit_len = n // min_repeats
    if max_unit_len < min_unit_len:
        return None
    pattern = re.compile(
        r"(.{" + str(min_unit_len) + "," + str(max_unit_len) + r"}?)\1{"
        + str(min_repeats - 1)
        + ",}",
        re.DOTALL,
    )
    match = pattern.search(s)
    if match:
        return s[: match.start()] + match.group(1)
    return None


def _clean_repeated_content(
    content: str,
    min_len: int = 10,
    min_repeats: int = 10,
    line_threshold: int = 10,
) -> str:
    """Remove repeated content at both character and line level.

    Ported from glmocr/utils/result_postprocess_utils.py:clean_repeated_content().
    """
    stripped_content = content.strip()
    if not stripped_content:
        return content

    # Check for consecutive character-level repetition
    if len(stripped_content) > min_len * min_repeats:
        result = _find_consecutive_repeat(
            stripped_content, min_unit_len=min_len, min_repeats=min_repeats
        )
        if result is not None:
            return result

    # Check for line-level repetition
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    total_lines = len(lines)
    if total_lines >= line_threshold and lines:
        common, count = Counter(lines).most_common(1)[0]
        if count >= line_threshold and (count / total_lines) >= 0.8:
            for i, line in enumerate(lines):
                if line == common:
                    consecutive = sum(
                        1
                        for j in range(i, min(i + 3, len(lines)))
                        if lines[j] == common
                    )
                    if consecutive >= 3:
                        original_lines = content.split("\n")
                        non_empty_count = 0
                        for idx, orig_line in enumerate(original_lines):
                            if orig_line.strip():
                                non_empty_count += 1
                                if non_empty_count == i + 1:
                                    return "\n".join(original_lines[: idx + 1])
                        break
    return content


def _clean_ocr_output(text: str) -> str:
    """Clean up raw OCR model output.

    Runs repetition removal and strips excessive whitespace.
    """
    text = _clean_repeated_content(text)
    # Strip excessive blank lines (3+ consecutive → 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Device detection (shared pattern from hf_classifier.py)
# ---------------------------------------------------------------------------


def _detect_device() -> "torch.device":
    """Detect the best available device: MPS > CUDA > CPU."""
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


@PluginRegistry.pdf_parser
class GlmOcrParserPlugin(BasePDFParserPlugin):
    """
    PDF parser using the GLM-OCR 0.9B vision-language model.

    Renders each PDF page to an image and runs the VLM for high-quality
    text extraction in markdown format. Model is loaded lazily on first use.

    Best for: scanned documents, image-heavy PDFs, tables, formulas.
    """

    MODEL_ID = "zai-org/GLM-OCR"

    def __init__(self, dpi: int = 200, max_new_tokens: int = 4096) -> None:
        self._dpi = dpi
        self._max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        """Load model and processor on first use."""
        if self._model is not None:
            return

        from transformers import AutoProcessor, AutoModelForImageTextToText

        device = _detect_device()
        logger.info("Loading GLM-OCR model %s on %s", self.MODEL_ID, device)

        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.MODEL_ID, torch_dtype="auto", device_map="auto"
        )
        logger.info("GLM-OCR model loaded")

    # -- Properties ----------------------------------------------------------

    @property
    def name(self) -> str:
        return "glm_ocr_parser"

    @property
    def priority(self) -> int:
        return 200  # pypdf_parser is 100 (default)

    @property
    def description(self) -> str:
        return "PDF parser using GLM-OCR 0.9B vision-language model for high-quality OCR"

    @property
    def supports_ocr(self) -> bool:
        return True

    # -- Core parsing --------------------------------------------------------

    def parse(
        self,
        pdf_bytes: bytes,
        max_pages: int = 500,
        use_ocr: bool = False,
    ) -> PDFParseResult:
        """
        Extract text from a PDF by rendering pages to images and running GLM-OCR.

        The use_ocr flag is ignored — this parser always uses OCR.
        """
        import fitz
        from PIL import Image

        self._ensure_loaded()

        try:
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            return PDFParseResult(pages=[], page_count=0, error=f"Failed to open PDF: {e}")

        total_pages = len(pdf_doc)
        pages_to_process = min(total_pages, max_pages)
        logger.info(
            "GLM-OCR: processing %d/%d pages at %d DPI", pages_to_process, total_pages, self._dpi
        )

        pages: list[str] = []
        for i in range(pages_to_process):
            page = pdf_doc[i]
            pix = page.get_pixmap(dpi=self._dpi)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            # Resize for optimal VLM input
            new_h, new_w = _smart_resize(img.height, img.width)
            if (new_h, new_w) != (img.height, img.width):
                img = img.resize((new_w, new_h))

            text = self._ocr_image(img)
            text = _clean_ocr_output(text)
            pages.append(text)

            if (i + 1) % 5 == 0 or i == pages_to_process - 1:
                logger.info("GLM-OCR: processed %d/%d pages", i + 1, pages_to_process)

        metadata = self._extract_metadata(pdf_doc)
        pdf_doc.close()

        return PDFParseResult(
            pages=pages,
            page_count=total_pages,
            metadata=metadata,
        )

    def _ocr_image(self, image: "Image.Image") -> str:
        """Run GLM-OCR on a single PIL image and return extracted text."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": _OCR_PROMPT},
                ],
            }
        ]

        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_input], images=[image], return_tensors="pt", padding=True
        ).to(self._model.device)

        import torch

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens)

        # Strip the input tokens to get only generated text
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        result = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return result

    @staticmethod
    def _extract_metadata(pdf_doc: Any) -> dict[str, Any]:
        """Extract PDF metadata (same approach as pypdf_parser)."""
        metadata: dict[str, Any] = {}
        try:
            doc_metadata = pdf_doc.metadata
            if doc_metadata:
                field_map = {
                    "title": "title",
                    "author": "author",
                    "subject": "subject",
                    "keywords": "keywords",
                    "creator": "creator",
                    "producer": "producer",
                    "creationDate": "created",
                    "modDate": "modified",
                }
                for pdf_key, our_key in field_map.items():
                    value = doc_metadata.get(pdf_key)
                    if value and isinstance(value, str) and value.strip():
                        metadata[our_key] = value.strip()
        except Exception as e:
            logger.debug("Error extracting metadata: %s", e)
        return metadata
