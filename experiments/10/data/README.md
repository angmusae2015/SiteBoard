# 10 실험 데이터

[공통 정제본](../../../dataset/processed/dataset.json)의 2,123건을 사용했습니다. 보존된 [클러스터](cluster.json)는 원문을 정규화한 패턴을 SequenceMatcher 유사도 0.85 기준으로 묶은 231개 군집입니다. [공종 마스터](work_type.json)는 현장별 고정 규격, 가변 규격, 규격 없는 공종을 정의합니다.

[분할 메타데이터](split_meta.json)는 시드 42와 복합 홀드아웃 클러스터 20, 22, 35, 36, 42, 65를 기록합니다. 홀드아웃과 단일 레코드 군집은 평가에 배정하고, 나머지 군집은 목표 비율 0.18을 적용하되 학습과 평가에 각각 최소 한 건을 둡니다.

| 자료 | 건수 | 구성 |
| --- | ---: | --- |
| [train.jsonl](train.jsonl) | 1,568 | 학습 레코드 |
| [test.jsonl](test.jsonl) | 555 | `seen_patterns` 392건, `unseen_complex_holdout` 74건, `unseen_rare` 89건 |

원천과 학습 및 평가의 `(site, image_id)`, 원문과 정답을 전수 대조했습니다. 식별자 중복, 분할 간 겹침, 원천 누락은 없었습니다. 각 작업의 공종과 규격을 해당 현장의 마스터 규칙과 대조했습니다. 클러스터, 마스터, 분할 메타데이터와 두 JSONL은 09에 보존된 대응 파일과 바이트 단위로 같습니다. 10의 준비 코드는 공통 정제본과 이 디렉터리의 파일을 읽습니다.

데이터 준비 순서는 [cluster.py](../cluster.py), [partition.py](../partition.py), [build_dataset.py](../build_dataset.py)입니다. 보존본을 유지하려면 새 실험 경로에서 생성하세요.
