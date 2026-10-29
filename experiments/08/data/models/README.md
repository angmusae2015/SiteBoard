---
base_model: Qwen/Qwen3-8B
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen3-8B
- lora
- transformers
- unsloth
---
# 08 모델 저장 위치

08 학습 노트북이 이 디렉터리에 병합하지 않은 PEFT 어댑터와 공식 토크나이저를 저장합니다. 기본 모델은 고정 revision의 `Qwen/Qwen3-8B`이며 07의 어댑터를 이어 사용하지 않습니다.

시드 42, 증강 확률 0.0의 학습은 3에포크, 588스텝까지 완료했습니다. 기존 엄격한 생성 평가는 **488/555, 87.93%**, 합산 및 분리 허용 추가 기준은 **494/555, 89.01%**입니다. 실제 평가 배치는 1이고 전체 값 토큰 최소 신뢰도는 0.350750입니다. 신뢰도는 정답과 오답을 모두 포함한 케이스별 value_min_confidence의 min이며 평균이나 정답 확률이 아닙니다.

[학습 설정](../result/run_config.json), [학습 이력](../result/training_history.json), [원본 평가](../result/evaluation.json)를 보존합니다. 유형별 점수와 07 대비 비교 및 추가 기준의 정의는 [결과 분석](../../results/README.md)을 따릅니다. 원래 점수와 가중치는 변경하지 않았습니다.

`checkpoints/checkpoint-*/`에는 에포크별 Trainer 체크포인트를 최대 2개 유지합니다. 기존 모델 또는 체크포인트가 있으면 새 학습을 중단합니다. 별도 실행의 보존은 새 실험 ID를 사용합니다.

루트 Git 제외 규칙에 따라 가중치, 토크나이저와 체크포인트는 추적하지 않으며 README와 `adapter_config.json`은 추적 대상입니다. 평가에서는 같은 실행의 어댑터와 토크나이저를 사용하고 4비트 기본 모델에 연결합니다.

[실험 안내](../../README.md) | [실행 기록](../result/README.md)
### Framework versions

- PEFT 0.18.1
