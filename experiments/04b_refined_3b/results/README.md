# 실행·평가 결과

- [eval_results_3b.json](eval_results_3b.json)
- [metrics.json](metrics.json)

## 평가 요약

| 유형 | 평가 건수 | 정답 수 | 일치율 | 값 토큰 최소 신뢰도 (0~1) |
| --- | ---: | ---: | ---: | ---: |
| 전체 | 555 | 496 | 89.37% | 0.3494 |
| seen · 학습에서 본 패턴 | 392 | 368 | 93.88% | 0.4049 |
| unseen_complex · 미학습 복합 패턴 | 74 | 60 | 81.08% | 0.3494 |
| unseen_rare · 희소 패턴 | 89 | 68 | 76.40% | 0.4345 |

최소 신뢰도는 해당 행의 정답·오답 케이스에 기록된 value_min_confidence의 최솟값입니다. workType·spec·quantity의 값 구간에 해당하는 토큰을 대상으로 하며, 전체 JSON 토큰의 최소 신뢰도나 케이스별 최소 신뢰도의 평균과 구분합니다. 저장된 소수 4자리 값을 그대로 집계했습니다.

유형은 이 실험의 [test.jsonl](../data/test.jsonl)에 있는 metadata.test_type을 평가 케이스의 input_user와 대응해 확인했습니다. 모든 입력이 일치하고 유형 대응이 유일한지 검사했습니다. 일치 여부는 평가 JSON의 is_exact_match를 사용합니다.

예측·정답·점수·설정과 기존 경로 문자열은 원문 그대로 보존합니다.
metrics.json은 상세 결과에서 개별 cases를 제외한 기존 요약입니다.

해석과 재현 한계는 [실험 안내](../README.md), 지표는 [평가 문서](../../README.md#평가-기준과-비교-한계)를 따릅니다.
