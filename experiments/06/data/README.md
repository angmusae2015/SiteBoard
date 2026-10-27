# 06 데이터와 분할

Qwen 3B 균등 문장 결합 QLoRA 학습과 역빈도 가중치 분석에 사용하는 보존 입력입니다.

| 파일 | 내용 | 생성 주체 |
| --- | --- | --- |
| [work_type.json](work_type.json) | 현장별 허용 공종과 규격 규칙 | 이관 시점 공통 마스터 사본, 생성 코드 없음 |
| [cluster.json](cluster.json) | 2,123건, 231개 패턴 클러스터와 원본 인덱스 | 실험의 cluster.py |
| [split_meta.json](split_meta.json) | 학습 1,568건, 평가 555건의 유형과 인덱스 | 실험의 partition.py |
| [train.jsonl](train.jsonl) | 학습 메시지와 원본 식별자 | 실험의 build_dataset.py |
| [test.jsonl](test.jsonl) | 평가 메시지, 원본 식별자와 평가 유형 | 실험의 build_dataset.py |

세 스크립트의 실행 위치와 입력은 [실험 안내](../README.md#데이터-준비)를 따릅니다. 공종과 규격은 이 디렉터리의 마스터로 검증하고, (site, image_id) 중복과 학습 및 평가 간 중복과 누락을 확인합니다. 문장 결합 증강은 데이터 생성 시가 아니라 학습 중 수행합니다.

학습 및 평가 결과인 `result/`와 모델 파일은 사용자 요청으로 삭제했습니다. 모델 보유 상태는 [models/README.md](models/README.md)를 참고하세요. 이 디렉터리의 데이터 원문과 분할은 변경하지 않았습니다.
