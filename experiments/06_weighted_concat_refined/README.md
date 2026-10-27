# 06: 정제 데이터 패턴 클러스터링, 분할과 Qwen 데이터 생성

공통 정제본의 텍스트 패턴을 클러스터로 묶고, 클러스터 기준으로 학습용 및 평가용 원본 인덱스를 결정한 뒤 Qwen 파인튜닝용 메시지로 변환하는 데이터 준비 코드입니다. [cluster.py](cluster.py), [partition.py](partition.py), [build_dataset.py](build_dataset.py)는 로컬에서 순서대로 실행하는 독립 스크립트이며 다른 실험의 코드를 참조하지 않습니다.

## 스크립트와 입출력

경로와 알고리즘 파라미터는 각 스크립트 상단의 전역변수에 직접 작성합니다. 세 스크립트는 Python 표준 라이브러리만 사용합니다. 모든 입력과 출력 파일에는 상대경로로 접근하며, 경로의 기준은 스크립트 파일 위치가 아니라 **실행 시 현재 작업 디렉터리인 이 실험 디렉터리**입니다. `../../dataset/processed/dataset.json`은 공통 정제본을, `data/`는 이 실험의 데이터 디렉터리를 가리킵니다.

| 스크립트 | 입력 | 출력 | 목적 |
| --- | --- | --- | --- |
| [cluster.py](cluster.py) | `../../dataset/processed/dataset.json` | `data/cluster.json` | 유사한 텍스트 패턴을 묶고 원본 레코드와 클러스터의 대응 관계를 저장 |
| [partition.py](partition.py) | `data/cluster.json`, `../../dataset/processed/dataset.json` | `data/split_meta.json` | 클러스터별 분할 규칙을 적용하고 원본 인덱스와 통계를 저장 |
| [build_dataset.py](build_dataset.py) | `../../dataset/processed/dataset.json`, `data/work_type.json`, `data/cluster.json`, `data/split_meta.json` | `data/train.jsonl`, `data/test.jsonl` | 저장된 분할에 따라 원본 레코드를 Qwen 파인튜닝용 메시지로 변환 |

입력 데이터의 구조는 [공통 데이터 명세](../../dataset/README.md)를 따릅니다. 클러스터링, 분할, 메시지 변환은 각각 별도 스크립트에서 수행합니다.

## cluster.py: 텍스트 패턴 클러스터링

학습 노트북에서 분리한 `PatternClusterer`의 구현과 주석을 보존합니다. 원본 레코드의 `x`에서 번호, 치수, 수량 등의 표현을 정규화한 문자열을 추출하고, `difflib.SequenceMatcher`로 대표 문자열과의 유사도를 계산합니다. 정답 `y`는 클러스터링 기준으로 사용하지 않습니다.

정규화 문자열의 레코드 빈도가 높은 순서로 대표 패턴을 선택하고, 대표와의 유사도가 `SIMILARITY_THRESHOLD = 0.85` 이상인 미배정 패턴을 흡수합니다. 완성된 클러스터는 소속 레코드 수 내림차순으로 정렬하며, 정렬 후 0부터 `cluster_id`를 부여합니다.

`cluster.json`은 `fit()` 완료 후의 `to_dict()` 전체 필드와 입력 파일의 SHA-256을 포함합니다.

| 필드 | 내용 |
| --- | --- |
| `total_samples` | 전체 원본 레코드 수 |
| `total_clusters` | 생성된 클러스터 수 |
| `similarity_threshold` | 적용한 유사도 임계값 |
| `clusters` | 클러스터 ID, 대표 문자열, 소속 정규화 문자열, 레코드 수, 전체 대비 비율(%), 원본 인덱스 목록 |
| `sample_to_cluster` | 원본 인덱스에서 클러스터 ID로의 매핑. JSON 객체의 키는 문자열로 저장됨 |
| `dataset_sha256` | 읽은 `dataset.json` 파일 바이트의 SHA-256 |

`sample_indices`와 `sample_to_cluster`의 인덱스는 입력 JSON 배열의 0부터 시작하는 위치입니다. 현장별 사진 번호인 `image_id`와 구분합니다.

## partition.py: 클러스터 기준 분할

`partition_clusters()`는 저장된 클러스터를 순서대로 순회하며, 독립적인 `random.Random(SEED)`로 각 클러스터의 원본 인덱스 목록을 섞습니다. 다음 우선순위로 평가 유형과 학습 인덱스를 결정하고, 최종 인덱스 목록은 오름차순으로 저장합니다.

| 우선순위 | 조건 | 배정 |
| --- | --- | --- |
| 1 | ID가 `HOLDOUT_CLUSTER_IDS`에 포함됨 | 클러스터 전체를 `unseen_complex_holdout` 평가 유형에 배정 |
| 2 | 홀드아웃이 아니며 소속 레코드가 1개임 | 해당 레코드를 `unseen_rare` 평가 유형에 배정 |
| 3 | 나머지 일반 클러스터 | 일부를 `seen_patterns` 평가 유형에, 나머지를 학습셋에 배정 |

현재 파라미터는 다음과 같습니다.

| 전역변수 | 값 |
| --- | --- |
| `SEED` | `42` |
| `TEST_RATIO` | `0.18` |
| `HOLDOUT_CLUSTER_IDS` | `[20, 22, 35, 36, 42, 65]` |

일반 클러스터의 크기를 `n`이라 할 때 평가 건수는 `min(n - 1, max(1, round(n * TEST_RATIO)))`입니다. 일반 클러스터마다 학습과 평가에 최소 1건씩 배정하므로, `TEST_RATIO`는 전체 데이터의 최종 평가 비율과 다릅니다.

홀드아웃 목록은 기존 분할 기준을 유지한 값입니다. `unseen_complex_holdout`은 지정된 클러스터를 뜻하며, 모든 레코드가 다중 작업이라는 의미는 아닙니다. 22번과 36번 클러스터도 그대로 포함됩니다. 분할은 현장 전체를 격리하는 방식이 아닙니다.

### split_meta.json 구조

`build_split_meta()`는 01번 실험의 `data/split_meta.json` 구조에서 `test.cluster_coverage`를 제외한 통계와 인덱스를 만듭니다. `main()`에서 실제로 읽은 데이터셋의 `dataset_sha256`을 최상위 필드로 추가합니다. 실행 중 01번 실험의 파일을 읽지는 않습니다.

- 최상위: `total_samples`, `total_clusters`, `seed`, `train`, `test`, `dataset_sha256`.
- `train`: 레코드 수, 전체 대비 비율, 클러스터 수, 다중 작업 레코드 수와 학습셋 내 비율, 원본 인덱스 목록.
- `test`: 전체 평가 레코드 수와 비율, 다중 작업 레코드 수와 평가셋 내 비율, 유형별 `subsets`, 전체 평가 원본 인덱스 목록.
- `test.subsets`: `seen_patterns`, `unseen_complex_holdout`, `unseen_rare` 각각의 레코드 수, 클러스터 수, 원본 인덱스 목록. 홀드아웃 유형에는 `cluster_ids`도 포함.

다중 작업은 원본 레코드의 `len(y) >= 2`로 집계합니다. 비율은 백분율을 소수 둘째 자리까지 반올림합니다. `partition.py`는 이 통계를 계산하기 위해 원본 `dataset.json`도 읽습니다. 두 산출물에는 각각 입력 데이터셋의 해시가 기록됩니다.

## build_dataset.py: Qwen 파인튜닝용 메시지 생성

`build_dataset.py`는 로컬에서 학습용 `train.jsonl`과 평가용 `test.jsonl`을 준비하는 스크립트입니다. `split_meta.json`의 학습 및 평가 인덱스 순서대로 공통 정제본의 레코드를 선택하고, `cluster.json`에서 해당 레코드의 클러스터 정보를 가져옵니다. 기존 분할을 그대로 사용하며 클러스터링과 분할을 다시 수행하지 않습니다.

시스템 프롬프트는 스크립트 상단의 `SYSTEM_PROMPT`에 정의합니다. 현장별 허용 공종과 규격 규칙은 이 실험의 `data/work_type.json`에서 읽습니다. 프롬프트와 메시지 변환 코드는 스크립트 안에 있으며 노트북을 읽거나 호출하지 않습니다.

JSONL의 각 줄에는 `messages`와 `metadata`를 가진 객체 하나를 저장합니다.

| 필드 | 내용 |
| --- | --- |
| `messages` | `system`, `user`, `assistant` 순서의 Qwen ChatML용 메시지 목록 |
| `messages[0].content` | 시스템 프롬프트 |
| `messages[1].content` | 해당 현장의 허용 공종 및 규격 규칙과 원본 `x` 텍스트 |
| `messages[2].content` | 원본 `y` 배열을 들여쓰기 2칸으로 직렬화한 JSON 문자열 |
| `metadata` | `site`, `image_id`, `source_index`, `multi_label`, `y_count`, `cluster_id`, `rep_signature`, `split_group` |
| `metadata.test_type` | 평가 레코드에만 포함되는 평가 유형 |

`split_group`은 학습 레코드에서 `train`, 평가 레코드에서 `seen_patterns`, `unseen_complex_holdout`, `unseen_rare` 중 하나입니다. 평가 레코드의 `test_type`은 `split_group`과 같습니다. `source_index`는 원본 JSON 배열의 인덱스이며, `multi_label`은 정답 작업이 2개 이상인지 나타냅니다.

메시지는 역할과 본문을 가진 객체로 저장하며, `<|im_start|>` 등의 특수 토큰을 붙인 문자열로 변환하거나 토큰화하지 않습니다. 학습 중 문장 결합 증강도 이 스크립트에서 수행하지 않습니다. 전체 레코드의 별도 변환본인 `qwen_finetune.jsonl`은 생성하지 않습니다.

## 실행과 재현

프로젝트 루트에서 다음 순서로 실행합니다.

```bash
cd experiments/06_weighted_concat_refined
python cluster.py
python partition.py
python build_dataset.py
```

출력 디렉터리는 필요하면 생성합니다. JSON과 JSONL은 UTF-8, `ensure_ascii=False`로 저장합니다. `cluster.json`과 `split_meta.json`은 들여쓰기 2칸의 JSON이며, `train.jsonl`과 `test.jsonl`은 한 줄에 레코드 하나씩 저장합니다.

동일한 입력 파일과 배열 순서, 코드, 파라미터, Python 버전을 재현 조건으로 삼습니다. 클러스터링의 동률 정렬은 원본 등장 순서를 유지하므로 입력 배열 순서도 결과에 영향을 줍니다. 빈도순 정렬과 ID 부여, 클러스터 순회 순서는 홀드아웃 대상과 난수 사용 순서에 연결되어 있습니다.

`dataset_sha256`은 JSON을 다시 직렬화한 값이 아니라 파일 바이트의 해시이므로, 공백이나 줄바꿈만 바뀌어도 달라집니다. 현재 파라미터의 기준은 각 스크립트의 전역변수입니다. `split_meta.json`에는 시드와 홀드아웃 ID가 기록되며 `TEST_RATIO`는 별도 필드로 저장되지 않습니다.

## 학습·평가 노트북 설정

[학습 노트북](train_colab.ipynb)과 [평가 노트북](evaluate_colab.ipynb)은 각각 상단의 `RUNTIME_PARAMS` 딕셔너리에 모델, 파일 경로와 실행 장치 설정을, `HYPER_PARAMS` 딕셔너리에 학습, 증강, LoRA, 양자화와 평가 설정을 선언합니다. 각 노트북에서 필요한 값을 직접 수정하며, 실행 셀은 두 딕셔너리를 복사해 `config`로 병합합니다. 노트북의 상대 경로는 프로젝트 루트 기준입니다.

실행 정보의 파일 해시는 학습·평가 노트북을 대상으로 수집합니다.

## 확인된 결과

현재 공통 정제본 2,123건을 대상으로 세 스크립트를 순서대로 실행해 이 실험의 `data/`에 결과물을 생성했습니다.

| 구분 | 레코드 수 |
| --- | ---: |
| 전체 | 2,123 |
| 학습 | 1,568 |
| 평가 전체 | 555 |
| 평가: `seen_patterns` | 392 |
| 평가: `unseen_complex_holdout` | 74 |
| 평가: `unseen_rare` | 89 |

총 231개 클러스터를 생성했으며, `data/cluster.json`, `data/split_meta.json`, `data/train.jsonl`, `data/test.jsonl`을 저장했습니다. `data/work_type.json`은 이 실험의 현장별 공종 마스터 사본입니다.

[실험 색인](../README.md) | [평가 기준과 비교 한계](../README.md#평가-기준과-비교-한계)
