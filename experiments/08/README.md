# 08: Qwen3-8B 비추론 비증강 QLoRA

07의 문장 결합 증강이 정확도에 미친 영향을 확인하기 위한 대조 실험입니다. 모델, 프롬프트, 원본 데이터, 분할, 학습과 평가 설정을 유지하고 학습 증강 확률만 `0.5`에서 `0.0`으로 바꿉니다. 오타 보정을 포함한 기존 정답을 유지하며 채점 기준을 완화하거나 정답을 수정하지 않습니다.

학습과 평가를 완료했으며 기존 엄격한 채점은 **87.93% (488/555)**입니다. 07 대비 −14건, −2.52%p이고 그중 11건의 차이는 실리콘마감 홀드아웃 클러스터 22에 집중됩니다. 같은 공종과 규격의 총수량이 같으면 합산과 분리를 인정하는 추가 기준에서는 **89.01% (494/555)**입니다. 원래 채점과 JSON은 보존하며 유형별 점수와 신뢰도, 07과의 대응 비교 및 추가 기준은 [결과 분석](results/README.md)에 기록합니다.

## 비교 설계

비교 경로는 07에서 실제 사용한 `train_desktop_unsloth.ipynb`와 `evaluate_desktop.ipynb`입니다. 08도 같은 기본 모델에서 새 어댑터를 학습하며 07의 가중치나 체크포인트를 이어 사용하지 않습니다. 학습과 평가 실행 중 다른 실험의 코드, 데이터 및 결과를 읽지 않습니다.

| 항목 | 07과 08의 고정 조건 |
| --- | --- |
| 기본 모델 | `Qwen/Qwen3-8B` |
| 모델 revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| 채팅 템플릿 | 공식 토크나이저, `enable_thinking=False` |
| 프롬프트와 정답 | 07과 동일한 시스템 지시문, 현장 규칙, 원문과 정답 배열 |
| 데이터 | 학습 1,568건, 평가 555건, 같은 분할 인덱스와 평가 유형 |
| 학습 백엔드 | Unsloth 2026.10.9와 `transformers.Trainer` |
| 양자화 | NF4 4비트, 이중 양자화, 병합하지 않은 LoRA |
| 정밀도와 장치 | 비교 실행은 RTX 4070의 BF16, 고정 임베딩과 출력 계층도 BF16 |
| 최적화 | 3에포크, 배치 1, 누적 8, 학습률 0.0002, 시드 42, paged AdamW 8비트 |
| 스케줄 | cosine, warmup 10스텝, 전체 588 optimizer 스텝 |
| LoRA | rank 16, alpha 32, dropout 0.05, Q/K/V/O와 MLP 세 투영 계층 |
| 학습 길이와 손실 | 최대 1,280토큰, JSON 응답과 EOS만 손실 적용, 길이 초과 시 중단 |
| 체크포인팅 | `use_gradient_checkpointing='unsloth'`, 에포크별 저장, 최대 2개 유지 |
| 평가 | 4비트 기본 모델과 어댑터, 배치 1, greedy, 출력 상한 512토큰 |
| 채점 | `strict_schema_multiset_v1`, 작업 순서 무시, 중복 행 개수 유지 |

유일한 학습 파라미터 변경은 `HYPER_PARAMS["augmentation"]["probability"]=0.0`입니다. 최대 결합 수와 구분자 및 기존 Dataset 구현은 비교를 위해 남겨 두지만 증강 분기는 실행되지 않습니다. 원본 학습 레코드의 입력과 정답을 그대로 사용합니다. 평가 코드는 07의 대응 노트북과 같습니다.

증강을 끄면 같은 에포크와 스텝에서도 학습에 노출되는 작업 수와 토큰 수가 줄어듭니다. 이를 보상하는 추가 학습은 하지 않습니다. 이번 비교는 같은 원본 데이터와 학습 스케줄에서 기존 문장 결합 증강을 사용할 때의 효과를 확인하며, 토큰 노출량을 맞춘 비교는 아닙니다. 단일 시드 결과이므로 작은 차이를 일반적인 효과로 확정하지 않습니다.

## 파일과 데이터 준비

| 경로 | 역할 |
| --- | --- |
| [cluster.py](cluster.py) | 공통 정제본에서 패턴 클러스터 독립 생성 |
| [partition.py](partition.py) | 같은 설정으로 분할 인덱스와 유형 생성 |
| [build_dataset.py](build_dataset.py) | 같은 프롬프트와 정답으로 JSONL 생성 |
| [train_desktop_unsloth.ipynb](train_desktop_unsloth.ipynb) | 07과의 비교에 사용할 학습 |
| [evaluate_desktop.ipynb](evaluate_desktop.ipynb) | 07과 같은 엄격한 생성 평가 |
| [train_desktop.ipynb](train_desktop.ipynb), [train_colab.ipynb](train_colab.ipynb), [evaluate_colab.ipynb](evaluate_colab.ipynb) | 07의 대응 구성에서 증강을 끈 별도 실행 경로 |
| [data/](data/README.md) | 마스터, 클러스터, 분할과 학습 및 평가 JSONL |
| [data/models/](data/models/README.md) | 어댑터, 토크나이저와 체크포인트 저장 위치 |
| [data/result/](data/result/README.md) | 실행 설정, 학습 이력과 평가 JSON 저장 위치 |

공통 정제본 `dataset/processed/dataset.json`에서 준비 스크립트로 데이터를 독립 생성했습니다. 마스터는 07에 보존된 이관 시점 마스터를 내용 변경 없이 복사했으며 생성 코드를 추가하지 않습니다. 준비한 클러스터와 분할, 메시지 및 메타데이터를 07과 일회성 명령으로 전수 대조했고 데이터 파일 5개의 바이트와 SHA-256이 같습니다. 공종과 규격, 식별자 중복, train/test 교집합 및 원천 인덱스 누락 검사도 통과했습니다.

데이터 준비 명령은 프로젝트 루트에서 다음 순서로 실행합니다. 현재 준비된 파일을 덮어쓰는 재실행은 중단하며 보존할 재실행은 새 실험 ID를 사용합니다.

```powershell
.\.venv\Scripts\python.exe experiments/08/cluster.py
.\.venv\Scripts\python.exe experiments/08/partition.py
.\.venv\Scripts\python.exe experiments/08/build_dataset.py
```

## 학습과 평가 실행

1. 07과 같은 프로젝트 가상환경과 루트 의존성을 사용합니다. 비교 실행 전 라이브러리 버전과 GPU 및 dtype을 07의 보존 실행과 맞춥니다. 무조건 최신 버전으로 갱신하지 않습니다.
2. `train_desktop_unsloth.ipynb`의 커널을 `.venv/Scripts/python.exe`로, 작업 디렉터리를 `experiments/08`로 설정합니다. 상단 `RUNTIME_PARAMS`와 `HYPER_PARAMS`를 확인합니다. `prefix_path='./data/'`의 나머지 경로는 08 데이터 디렉터리를 기준으로 해석합니다.
3. 위에서 아래로 실행합니다. 원본 1,568건을 에포크마다 사용하고 파트너 결합은 하지 않습니다. 기존 어댑터, 체크포인트와 결과가 있으면 중단합니다.
4. 학습 후 커널을 재시작하고 `evaluate_desktop.ipynb`를 실행합니다. `prefix_path='data'`가 08을 가리키는지 확인합니다. 같은 기본 모델 revision과 dtype으로 08의 어댑터 및 토크나이저를 불러옵니다.
5. 실행 설정, 실제 패키지 버전, 입력 해시와 학습 이력 및 평가 기록을 08 내부에 보존합니다.

공통 Colab 절차는 [루트 안내](../../README.md#google-colab-실행)를 따릅니다. 08 Colab 노트북의 `prefix_path` 기본값은 `/drive/MyDrive/SiteBoard/08/`이며 실제 Drive 위치에 맞춥니다. 일반 학습과 Colab 구성은 Unsloth 데스크톱 비교 경로와 백엔드 또는 환경이 다를 수 있으므로 해당 결과 차이를 증강 하나의 효과로 해석하지 않습니다.

## 결과 비교 방법

07의 기준 결과는 **502/555, 90.45%**이며 유형별로 `seen_patterns` 369/392, `unseen_complex_holdout` 64/74, `unseen_rare` 69/89입니다. 출처는 [07 평가 원본](../07/data/result/evaluation.json)입니다.

08 평가 후 다음 항목을 일회성 명령으로 확인했고 [결과 분석](results/README.md)에 기록했습니다. 검증용 코드나 별도 집계 JSON은 추가하지 않았습니다.

1. 양쪽 평가의 `(site, image_id)`와 `source_index`를 대응하고 원문, 정답과 유형 및 평가 데이터 해시가 같은지 검증합니다. 케이스 번호나 배열 위치만으로 대응하지 않습니다.
2. 전체와 유형별 건수, 정답 수, 완전 일치율 및 케이스별 `value_min_confidence`의 최솟값을 집계합니다. 신뢰도는 JSON 문법 토큰을 제외한 workType, spec, quantity 값 토큰 기준이며 정답 확률로 해석하지 않습니다.
3. 07 오답에서 08 정답으로 바뀐 건수와 07 정답에서 08 오답으로 바뀐 건수를 함께 집계합니다. 전체 정답 수 차이가 두 전환 건수의 차이와 같은지 확인합니다.
4. 행 구성, 조건 범위, 규격과 수량, 작업 누락 및 추가, 오타 보정 실패를 검토합니다. 정답과 중복 개수 및 기존 채점은 유지합니다.
5. GPU, dtype, 라이브러리 버전, 설정, 학습 스텝과 실행 시간을 확인합니다. 환경이나 설정이 추가로 달라졌으면 비교 한계에 기록합니다.

학습 중 loss 평가에도 같은 test.jsonl을 사용한다는 07의 조건을 유지합니다. 에포크별 loss만으로 생성 정확도나 최적 에포크를 판단하지 않습니다. 실제 기록에서 학습 하이퍼파라미터는 증강 확률만 다르고 평가 하이퍼파라미터는 같습니다. 모델 revision과 패키지 버전, GPU와 dtype은 같지만 Windows 플랫폼 빌드는 07이 26200, 08이 26300입니다. 단일 시드의 차이를 일반화하지 않으며 09의 시드 43, 44, 45 평균에 이 실행을 합치지 않습니다.

[실험 색인](../README.md) | [데이터 안내](data/README.md)
