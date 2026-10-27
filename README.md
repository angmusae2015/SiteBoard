# SiteBoard · 작업 기록 구조화 모델 실험

건설 현장의 작업 텍스트를 `workType`·`spec`·`quantity` 배열로 변환하는 Qwen 모델의 실험 프로젝트입니다. 실험별 코드·데이터·모델·평가 기록을 독립적으로 보존합니다.

## 문제와 입출력

사진대지 보드판의 작업 텍스트에는 공종 축약어·규격·수량·단면·양면과 여러 작업이 함께 기록됩니다. 현장 코드와 텍스트를 입력받아 현장별 공종·규격 규칙에 맞는 작업 내역 JSON 배열을 추출합니다.

출력의 workType은 공종명, spec은 규격·치수 또는 빈 문자열, quantity는 수량입니다. 규격 규칙은 고정 목록·가변 치수·규격 없음으로 구분합니다. 사진의 OCR은 별도 도구이며 실제 시공 상태 판별은 현재 텍스트 모델의 구현 범위에 포함되지 않습니다.

## 먼저 읽을 문서

| 목적 | 문서 |
| --- | --- |
| 실험 조건·결과·재현 상태 | [실험 색인](experiments/README.md) |
| 공통 원천 데이터와 정정 이력 | [데이터 명세](dataset/README.md) |
| 표 크롭·OCR 재사용 | [도구 안내](tools/table_ocr/README.md) |
| 평가 기준·비교 한계 | [실험 색인](experiments/README.md#평가-기준과-비교-한계) |
| 엑셀 필드 매핑 | [원본 자료 명세](dataset/raw/source_files/excel-mapping.md) |
| 입력·출력 예시 | [예시](examples/README.md) |
| 변경 시 작업 규칙 | [AGENTS.md](AGENTS.md) |

## 파일 구조

```text
SiteBoard/
├── dataset/             # 공통 원본·정제본·현장 정정 사례, 원본 엑셀·사진
├── experiments/<ID>/    # 각 실험의 노트북, data, models, results
├── tools/table_ocr/     # 표 검출·크롭·OCR
├── examples/            # 입력·기대 정답
└── pyproject.toml       # 프로젝트 의존성·OCR 도구 패키징
```

각 실험의 README를 먼저 읽고 필요한 자료 폴더로 이동합니다. 과거 실험에는 보유한 자료만 있으며 정확한 당시 코드가 없어 노트북 구성을 보류했습니다. 기존 설정 JSON의 경로는 당시 기록입니다.

## Google Colab 실행

1. 프로젝트 전체를 Colab 또는 Drive에 준비하고 GPU 런타임을 선택합니다. Drive 연결은 Colab UI에서 수행합니다.
2. [06 학습 노트북](experiments/06_weighted_concat_refined/train_colab.ipynb)을 열고 프로젝트 루트를 현재 작업 디렉터리로 지정합니다. 필요하면 첫 설정 셀의 `PROJECT_ROOT`를 실행 환경의 루트로 지정하고 `EXPERIMENT_DIR`을 확인합니다.
3. 첫 설치 셀은 프로젝트 루트의 `requirements-qlora.txt`를 사용합니다. 설치 후 필요하면 런타임을 재시작합니다.
4. 각 노트북 상단의 `RUNTIME_PARAMS`와 `HYPER_PARAMS` 딕셔너리를 검토하고 필요한 값을 수정한 뒤 위에서 아래로 실행합니다. 실행 셀은 두 딕셔너리를 복사해 `config`로 병합합니다. 경로는 프로젝트 루트 기준 상대 경로입니다.
5. [06 평가 노트북](experiments/06_weighted_concat_refined/evaluate_colab.ipynb)은 같은 실험의 완료된 학습 기록·평가 데이터·어댑터·토크나이저를 읽어 결과를 저장합니다.

학습 노트북은 정제본과 해당 실험의 공종 마스터 사본을 읽어 전체 Qwen 메시지·패턴 클러스터·train/test·분할 메타데이터를 만든 뒤 QLoRA 학습을 수행합니다. 마스터 생성 함수는 생성 규칙을 정한 뒤 추가할 후속 작업입니다.

현재 06은 `min_weight=1.0` 등 기존 설정의 학습·증강 값을 유지합니다. 05의 `0.01` 조건과 혼동하지 않습니다. 입력 해시·실제 설정·라이브러리 버전·확보 가능한 기본 모델 revision을 기록하며 기존 산출물은 덮어쓰지 않습니다. 보존할 새 실행은 새 실험 ID와 독립된 파일 사본을 사용합니다.

Colab의 임시 런타임을 종료하기 전에 실험 디렉터리 전체를 Drive 등 지속 저장소에 보존해야 합니다. 과거 기본 모델의 정확한 revision은 확인되지 않았으며, 06의 실제 GPU 학습·평가는 아직 검증하지 않았습니다.
