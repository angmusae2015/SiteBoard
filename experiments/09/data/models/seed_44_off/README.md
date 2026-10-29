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
# seed_44_off 모델

학습 시드 44, 문장 결합 증강 off, 증강 확률 0.0의 독립 어댑터와 토크나이저 저장 위치입니다. 같은 기본 모델 revision에서 독립 학습을 완료했습니다.

adapter_model.safetensors는 LoRA 가중치, adapter_config.json은 어댑터 설정, tokenizer.json과 tokenizer_config.json은 평가에 사용할 토크나이저입니다. checkpoints/에는 에포크별 체크포인트가 최대 2개 저장됩니다.

입력은 [공유 데이터](../../README.md), 설정과 평가 JSON은 [실행 결과](../../result/README.md)의 seed_44_off 접두사 파일입니다. 가중치와 토크나이저 및 체크포인트는 Git에서 제외하고 이 README와 최종 어댑터 설정만 추적합니다.
## 완료 상태와 평가

학습은 3에포크, 588스텝까지 완료했습니다. 기존 엄격한 채점은 **490/555, 88.29%**, 합산 및 분리 허용 추가 기준은 **491/555, 88.47%**입니다. 실제 생성 평가 배치는 4입니다. 전체 값 토큰 최소 신뢰도는 0.282567이며 정답과 오답을 모두 포함한 케이스별 value_min_confidence의 min입니다. 평균이나 정답 확률이 아닙니다.

[학습 설정](../../result/seed_44_off.run_config.json), [학습 이력](../../result/seed_44_off.training_history.json), [원본 평가](../../result/seed_44_off.evaluation.json)를 보존합니다. 유형별 점수, 추가 채점 기준과 시드 및 평가 배치 차이는 [결과 분석](../../../results/README.md)을 따릅니다. 원래 점수와 가중치는 변경하지 않았습니다.

### Framework versions

- PEFT 0.18.1
