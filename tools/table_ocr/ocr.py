"""Gemini OCR pipeline for construction-board table images."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import mimetypes
import os
from pathlib import Path
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

from .table_crop import iter_images

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

SYSTEM_INSTRUCTION = """건설 현장 공사 보드판 표 이미지를 보이는 그대로 전사하세요. 특수문자, 숫자, 영문, 한글을 수정·요약·표준화하지 마세요. 보이지 않는 내용은 추정하지 말고 null로 반환하세요. 표의 공사명, 공종, 위치, 내용, 일자 값을 정확히 대응시키세요."""


class BoardTableItem(BaseModel):
    model_config = ConfigDict(json_schema_extra={"required": ["공사명", "공종", "위치", "내용", "일자"]})
    공사명: str | None = Field(default=None)
    공종: str | None = Field(default=None)
    위치: str | None = Field(default=None)
    내용: str | None = Field(default=None)
    일자: str | None = Field(default=None)


@dataclass
class OCRResult:
    source_id: str
    file_name: str
    success: bool
    data: BoardTableItem | None = None
    raw_response: str | None = None
    error_message: str | None = None
    execution_time_sec: float = 0.0
    model_used: str | None = None

    def to_dict(self):
        result = asdict(self)
        result["data"] = self.data.model_dump() if self.data else None
        result["execution_time_sec"] = round(self.execution_time_sec, 3)
        return result


class GeminiOCREngine:
    DEFAULT_MODEL = "gemini-3.5-flash-lite"

    def __init__(self, model: str | None = None):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY 환경변수 또는 프로젝트 루트 .env 설정이 필요합니다.")
        self.model = model or os.environ.get("GEMINI_MODEL") or self.DEFAULT_MODEL
        self.client = genai.Client(api_key=api_key)
        self._api_key = api_key

    def process(self, path: Path) -> OCRResult:
        started = time.perf_counter()
        try:
            mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            response = self.client.models.generate_content(
                model=self.model,
                contents=[types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)],
                config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION, temperature=0, response_mime_type="application/json", response_schema=BoardTableItem),
            )
            raw = response.text or ""
            return OCRResult(str(path), path.name, True, BoardTableItem.model_validate_json(raw), raw, execution_time_sec=time.perf_counter() - started, model_used=self.model)
        except Exception as exc:
            message = str(exc).replace(self._api_key, "***API_KEY_MASKED***")
            return OCRResult(str(path), path.name, False, error_message=message, execution_time_sec=time.perf_counter() - started, model_used=self.model)


def save_result(result: OCRResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{Path(result.file_name).stem}_ocr.json"
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
