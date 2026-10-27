"""OpenCV utilities for detecting and cropping a construction-board table."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class CropResult:
    file_name: str
    cropped_image: np.ndarray
    bbox: tuple[int, int, int, int]
    confidence: float
    is_fallback: bool = False


class TableCropper:
    def __init__(self, dilate_iterations=1, erode_iterations=0, padding_px=0, filter_bottom_left=True):
        self.dilate_iterations = dilate_iterations
        self.erode_iterations = erode_iterations
        self.padding_px = padding_px
        self.filter_bottom_left = filter_bottom_left

    def crop(self, image: np.ndarray, file_name: str = "image") -> CropResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        if self.dilate_iterations:
            edges = cv2.dilate(edges, kernel, iterations=self.dilate_iterations)
        if self.erode_iterations:
            edges = cv2.erode(edges, kernel, iterations=self.erode_iterations)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = image.shape[:2]
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area_ratio = (w * h) / (width * height)
            if area_ratio < 0.03 or w / max(h, 1) < 1.1:
                continue
            position = (y + h / 2) / height + (1 - (x + w / 2) / width)
            score = area_ratio * (position if self.filter_bottom_left else 1.0)
            candidates.append((score, x, y, w, h, area_ratio))
        if not candidates:
            return CropResult(file_name, image.copy(), (0, 0, width, height), 0.0, True)
        _, x, y, w, h, area_ratio = max(candidates)
        pad = self.padding_px
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(width, x + w + pad), min(height, y + h + pad)
        return CropResult(file_name, image[y1:y2, x1:x2].copy(), (x1, y1, x2 - x1, y2 - y1), round(area_ratio, 4))


def iter_images(path: Path, recursive: bool = False):
    if path.is_file():
        if path.suffix.lower() in IMAGE_SUFFIXES:
            yield path
        return
    pattern = "**/*" if recursive else "*"
    yield from sorted(p for p in path.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def load_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return image


def save_crop(result: CropResult, output_dir: Path, quality: int = 95, save_metadata: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / result.file_name
    suffix = output_path.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if suffix in {".jpg", ".jpeg"} else []
    ok, encoded = cv2.imencode(suffix or ".jpg", result.cropped_image, params)
    if not ok:
        raise ValueError(f"이미지를 인코딩할 수 없습니다: {output_path}")
    encoded.tofile(output_path)
    if save_metadata:
        metadata = {"source_file": result.file_name, "bbox": list(result.bbox), "confidence": result.confidence, "is_fallback": result.is_fallback}
        output_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
