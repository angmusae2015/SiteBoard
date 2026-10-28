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
# 07 보유 모델

[Unsloth 데스크톱 학습](../../train_desktop_unsloth.ipynb)으로 생성한 Qwen3-8B의 병합하지 않은 PEFT LoRA 어댑터와 토크나이저를 로컬에 보존합니다.

| 자료 | 보유 상태 |
| --- | --- |
| adapter_config.json | LoRA rank 16, alpha 32, dropout 0.05, 기본 모델과 revision, PEFT 0.18.1 |
| adapter_model.safetensors | 로컬 어댑터 가중치 |
| tokenizer.json, tokenizer_config.json 및 보조 파일 | 같은 실행의 토크나이저와 채팅 템플릿 |
| checkpoints/checkpoint-392/ | 2에포크 체크포인트 |
| checkpoints/checkpoint-588/ | 3에포크 체크포인트 |

기본 모델은 `Qwen/Qwen3-8B`, revision은 `b968826d9c46dd6066d109eabc6255188de91218`입니다. [평가 노트북](../../evaluate_desktop.ipynb)은 동일 revision의 NF4 4비트 기본 모델에 이 어댑터를 연결하고 BF16과 비추론 모드로 평가했습니다. FP16 전체 모델로 병합하지 않았습니다. 다른 실험의 모델로 대체하지 않습니다.

Git에는 이 README와 adapter_config.json을 추적합니다. 가중치, 토크나이저와 체크포인트는 루트 .gitignore에 따라 제외합니다. 새로 복제한 저장소에서 평가하려면 제외된 로컬 자료를 별도로 복원하고 [평가 기록](../result/evaluation.json)의 SHA-256과 대조해야 합니다. Git 커밋이 모델 파일의 백업 완료를 뜻하지 않습니다.

기존 모델과 실행 결과를 덮어쓰지 않습니다. 보존할 새 실행은 새 실험 ID와 독립 경로를 사용합니다. [실험 안내](../../README.md) | [실행 기록](../result/README.md)
