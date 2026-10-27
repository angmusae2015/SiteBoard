# 재사용 데이터 수집 도구

[table_ocr/](table_ocr/README.md)에 표 검출·크롭과 Gemini 텍스트 추출 코드를 둡니다.
모델 실험과 분리해 신규 사진을 처리할 때 재사용합니다. 원본 자료는 dataset/raw/source_files/에 있습니다.
의존성은 루트 pyproject.toml·requirements.txt에서 관리합니다. 추출 결과의 학습 데이터 반영에는 사람 검수가 필요합니다.
