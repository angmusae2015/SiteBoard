# 표 크롭·OCR

- [table_crop.py](table_crop.py): OpenCV 표 검출·크롭, 이미지 순회·저장.
- [ocr.py](ocr.py): Gemini 원문 전사, 구조화 응답과 오류·처리 시간 저장.
- [__init__.py](__init__.py): 도구 패키지 경계.

입력은 원본 사진 → 크롭 이미지 → OCR 결과 순입니다. 별도 CLI는 없으며 프로젝트 루트에서 Python API를 호출합니다. 의존성은 루트 requirements.txt로 설치합니다.

```python
from pathlib import Path
from tools.table_ocr.table_crop import TableCropper, iter_images, load_image, save_crop
from tools.table_ocr.ocr import GeminiOCREngine, save_result

cropper = TableCropper()
engine = GeminiOCREngine()
for image in iter_images(Path("dataset/raw/source_files/photos")):
    crop = cropper.crop(load_image(image), image.name)
    cropped_path = save_crop(crop, Path("dataset/raw/source_files/tables"))
    result = engine.process(cropped_path)
    save_result(result, Path("dataset/raw/ocr_results"))
```

실행 전에 기존 출력과 겹치지 않는 경로를 지정하세요. GEMINI_API_KEY는 환경변수 또는 프로젝트 루트 .env에 설정하고 파일에 키를 기록하지 않습니다. GEMINI_MODEL로 모델을 지정할 수 있습니다. OCR 실행은 외부 API 호출입니다.

원본 사진·엑셀·기존 크롭을 보존합니다. 새 디렉터리가 생기면 해당 README를 작성하기 전에 구성안을 평가받고 dataset/README.md도 함께 갱신합니다. OCR 결과는 공종 정답이 아니므로 공통 정제본에 반영하기 전에 사람 검수와 대상 실험의 마스터 기준 검증이 필요합니다.

실제 API 호출과 OpenCV 이미지 처리는 이번 구조 정리에서 실행하지 않았습니다. [원본 자료](../../dataset/raw/source_files/README.md) · [데이터 명세](../../dataset/README.md)
