# 07 원본 실행 기록

Unsloth 데스크톱 학습과 일반 데스크톱 평가에서 생성된 JSON을 원본 내용 그대로 보존합니다. 평가 통계와 사례 해석은 [결과 분석](../../results/README.md)을 따릅니다.

| 파일 | 내용 |
| --- | --- |
| [run_config.json](run_config.json) | 실제 파라미터, 입력 해시, 실행 환경, training_completed 상태와 실행 시간 |
| [training_history.json](training_history.json) | 588스텝과 3에포크의 Trainer 이력, 학습 loss와 에포크별 eval_loss |
| [evaluation.json](evaluation.json) | 555건 원문 예측과 정답, 식별자와 유형, 채점, 값 토큰 신뢰도, 파일 해시와 실행 환경 |

## 실제 실행 환경

Windows, Python 3.10.11, NVIDIA GeForce RTX 4070, CUDA 12.8, torch 2.11.0+cu128, transformers 4.57.6, PEFT 0.18.1, accelerate 1.15.0, bitsandbytes 0.50.2입니다. 학습은 Unsloth 및 unsloth_zoo 2026.10.9와 transformers.Trainer, trl 0.24.0, triton-windows 3.6.0.post26, xformers 0.0.35를 기록했습니다. 평가는 Unsloth 학습 어댑터를 일반 PEFT 경로로 연결했습니다.

| 항목 | 학습 | 평가 |
| --- | ---: | ---: |
| 기록된 실행 시간 | 4,257.84초, 70.96분 | 4,170.75초, 69.51분 |
| 최대 할당 GPU 메모리 | 6.45 GiB | 6.44 GiB |
| 최대 예약 GPU 메모리 | 6.72 GiB | 6.62 GiB |
| 연산, 입력 임베딩, 출력 임베딩 dtype | BF16 | BF16 |

메모리는 PyTorch의 실행 중 최대 할당량과 예약량이며 전체 장치 사용량과 구분합니다. 평가 시간은 전체 기록 기준 건당 약 7.51초로, 순수 생성 지연 시간이나 별도 벤치마크가 아닙니다. 학습 Trainer 자체 train_runtime은 4,235.05초로 상위 실행 시간과 측정 범위가 다릅니다.

## 학습 경과와 보존

| 에포크 | 스텝 | eval_loss |
| --- | ---: | ---: |
| 1 | 196 | 0.01234669 |
| 2 | 392 | 0.01036039 |
| 3 | 588 | 0.01018180 |

loss는 감소했지만 에포크별 생성 정확도는 없어 최적 에포크나 추가 학습의 효과를 확정할 수 없습니다. test.jsonl은 학습 중 loss 평가에도 사용했으므로 최종 생성 평가만을 위한 별도 미사용 검증셋은 아닙니다.

평가 기록에 저장된 test.jsonl, split_meta.json, run_config.json, 어댑터 설정과 가중치, 토크나이저 JSON 두 파일의 해시가 분석 시점 로컬 자료와 일치했습니다. 실행 JSON에 남은 경로 문자열은 당시 원문이며 수정하지 않습니다. JSON은 Git으로 보존하고 가중치 및 토크나이저는 [모델 안내](../models/README.md)의 제외 범위를 따릅니다.

원본 작업 파일과 Git에 보존한 실행 JSON의 바이트는 대조했습니다. 사용자 결정에 따라 .gitattributes는 추가하지 않았고 저장소의 core.autocrlf 설정은 유지합니다. 재복제나 체크아웃 시 줄바꿈 변환만으로 파일 해시가 달라질 수 있으므로, 기록된 실행의 정확한 바이트를 복원하려면 Git blob과 로컬 원본을 확인해야 합니다. 해시를 맞추기 위해 과거 실행 JSON의 값을 바꾸지 않습니다.
