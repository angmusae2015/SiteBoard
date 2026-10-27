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

## Git 관리

개인 GitHub의 SiteBoard 비공개 저장소에서 기본 브랜치 `main`을 사용합니다. 코드, 노트북, 문서, 의존성, 프로젝트 설정과 공통 JSON, 실험별 데이터, 분할, 설정, 평가 기록을 추적합니다.

모델 가중치와 토크나이저, 원본 엑셀과 사진은 Git에서 제외합니다. 기존 제외 규칙의 예외인 모델 README와 `adapter_config.json`, 원본 자료 README와 엑셀 매핑 문서는 추적합니다. 가상환경, 캐시, `.env`와 `.env.*`, 노트북 자동 체크포인트, `.presentation-build/`는 제외합니다.

독립적인 변경은 최신 `main`에서 짧은 작업 브랜치를 만들어 진행합니다. `git status`와 `git diff`로 확인한 뒤 대상 경로를 지정해 스테이징하고, `git diff --cached`로 검토하여 목적별로 커밋합니다. 원격에 올린 변경은 검토 후 `main`에 반영합니다.

새 실험은 새 ID를 사용하고 설정, 입력 데이터와 분할, 평가 기록을 해당 실행에 연결해 보존합니다. 과거 자료를 덮어쓰거나 일괄 포맷하지 않습니다.

새로 복제한 저장소에는 제외 자료가 없습니다. 보유 자료를 원래의 프로젝트 상대 경로에 복원하고 파일 해시로 확인한 뒤 필요한 실행을 진행합니다. 외부 백업 위치는 후속 결정 사항이며, Git 도입은 전체 자료의 백업 완료를 뜻하지 않습니다.

## Google Colab 실행

1. 데이터 준비는 프로젝트 루트에서 `experiments/06`으로 이동한 뒤 `cluster.py`, `partition.py`, `build_dataset.py` 순서로 수행합니다. 기존 결과를 보존해야 하는 재실행은 새 실험 ID를 사용합니다. 자세한 입출력은 [06 실험 안내](experiments/06/README.md)를 따릅니다.
2. 프로젝트와 필요한 자료를 Colab 또는 Drive에 준비하고 GPU 런타임을 선택합니다. [학습 노트북](experiments/06/train_colab.ipynb)의 Drive 연결 셀을 실행합니다. 평가만 별도로 진행할 때도 Drive를 먼저 연결합니다.
3. 학습 노트북의 `RUNTIME_PARAMS["prefix_path"]`를 실행 환경의 자료 위치로 지정합니다. 경로 설정값은 이 위치를 기준으로 해석됩니다. 실험 디렉터리를 기준으로 사용할 경우 `train_file`과 `test_file`은 각각 `data/train.jsonl`, `data/test.jsonl`로 지정합니다. 모델과 결과 저장 경로도 기존 자료와 겹치지 않게 확인합니다.
4. 설치 셀은 현재 노트북에 직접 선언된 의존성을 설치합니다. `RUNTIME_PARAMS`와 `HYPER_PARAMS`를 검토하고 위에서 아래로 실행합니다. 학습 노트북은 준비된 JSONL을 읽어 문장 결합 증강과 QLoRA 학습을 수행합니다.
5. [평가 노트북](experiments/06/evaluate_colab.ipynb)도 `prefix_path`, 평가 데이터, 어댑터와 결과 경로를 지정하고 같은 실험의 학습된 어댑터와 토크나이저로 평가합니다.

06은 Qwen 3B의 균등 문장 결합 QLoRA 실험입니다. `min_weight=1.0`으로 파트너 선택 가중치가 균등하며 05의 `0.01` 조건과 구분합니다. [가중치 분석 노트북](experiments/06/weight_test.ipynb)은 역빈도 가중치가 동반 공종 노출과 레코드 반복에 주는 영향을 별도로 보여줍니다. 입력 해시·실제 설정·라이브러리 버전·확보 가능한 기본 모델 revision을 기록하며 기존 산출물은 덮어쓰지 않습니다. 보존할 새 실행은 새 실험 ID와 독립된 파일 사본을 사용합니다.

Colab의 임시 런타임을 종료하기 전에 실험 디렉터리 전체를 Drive 등 지속 저장소에 보존해야 합니다. 06은 학습과 평가를 수행했으며 사용자 요청으로 해당 모델과 원본 실행 및 평가 산출물을 삭제했습니다. 현재 보유 상태와 역빈도 가중 증강의 한계는 [06 실험 안내](experiments/06/README.md)를 따릅니다.

## 데스크톱 실행과 분석

06의 [학습 노트북](experiments/06/train_desktop.ipynb)과 [평가 노트북](experiments/06/evaluate_desktop.ipynb)은 프로젝트의 `.venv/Scripts/python.exe` 커널과 루트 [requirements.txt](requirements.txt)를 사용합니다. GPU 학습에는 CUDA가 필요합니다. 설치와 경로 설정은 각 노트북과 [06 실행 안내](experiments/06/README.md#학습-실행)를 참고하세요.

[weight_test.ipynb](experiments/06/weight_test.ipynb)는 GPU나 모델 없이 train 데이터만으로 현장별 파트너 선택 확률과 공종별 에포크당 예상 노출을 시각화합니다. 희소 공종을 포함한 레코드를 우선 선택하는 방식이 공종 전체의 빈도 균형을 보장하지 않음을 확인할 수 있습니다.
