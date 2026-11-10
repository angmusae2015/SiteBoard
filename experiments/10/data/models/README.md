# 10 최종 어댑터

최종 어댑터는 [43 on](seed_43_on/README.md), [43 off](seed_43_off/README.md), [44 on](seed_44_on/README.md), [44 off](seed_44_off/README.md), [45 on](seed_45_on/README.md), [45 off](seed_45_off/README.md), [46 on](seed_46_on/README.md), [46 off](seed_46_off/README.md) 디렉터리에 보존합니다.

Git은 조건별 모델 README와 `adapter_config.json`을 추적합니다. 평가에 사용할 때는 같은 실행의 `adapter_model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`를 해당 어댑터 디렉터리에 복원하세요. 중간 학습 체크포인트는 각 조건의 `checkpoints/`에 로컬 보존합니다.

평가 보고서의 `file_hashes`에서 최종 어댑터 설정, 가중치, 토크나이저 JSON 두 파일의 SHA-256을 확인할 수 있습니다. [실험 안내](../../README.md)에 기본 모델 revision과 실행 방법이 있습니다.
