---
trigger: "always"
description: "Dataset Maintenance and Documentation Sync Rule"
---

# 데이터셋 유지보수

## 변경과 문서

- 공통 원천 데이터는 dataset/raw/dataset_org.json, dataset/processed/dataset.json, dataset/processed/field_corrections.json입니다.
- dataset/의 파일을 추가·수정·삭제하면 [데이터 명세](../../dataset/README.md)의 목록·스키마·이력을 함께 갱신합니다.
- 각 MD는 작성·수정 전에 구성과 내용을 보고하고 평가받습니다. 지시하지 않은 파일·코드 추가와 삭제는 목적·효과를 설명하고 동의를 받습니다.
- 정확한 과거 실행 코드가 없으면 노트북을 추정해 만들지 않습니다. 과거 데이터·결과·가중치는 원형 보존합니다.

## 공종과 분할

- 마스터와 Qwen 변환본·클러스터·분할은 각 실험의 data/에 둡니다. 공통 work_type.json 경로를 사용하지 않습니다.
- 해당 실험의 data/work_type.json을 기준으로 현장별 공종·규격을 전수 검증합니다. (site, image_id) 중복과 train/test 중복·누락을 확인합니다.
- 기존 실험에는 이관 시점 마스터 사본만 보존하고 생성 스크립트를 추가하지 않습니다. 신규 생성 함수의 규칙은 후속 코드 설계에서 정합니다.
- 신규 실험은 공통 정제본에서 출발하고 다른 실험의 코드·생성 데이터를 참조하지 않습니다.

## 실행과 인코딩

- 검사·검증은 일회성 명령으로 수행하며 검증 전용 코드·노트북·테스트 디렉터리·보고서 파일을 남기지 않습니다.
- 학습 실행에 필요한 입력 검사는 실험의 노트북 내부에 포함합니다. 모델 평가 결과는 실험 재현 자료로 보존합니다.
- JSON·JSONL은 UTF-8과 ensure_ascii=False를 사용합니다. 보존 파일은 내용 변경 없이 이동·복사합니다.
- 설정 경로는 프로젝트 루트 기준 상대 경로, 문서 링크는 해당 문서 기준 상대 경로입니다. 공통 실행 방법은 [루트 README](../../README.md)를 따릅니다.
