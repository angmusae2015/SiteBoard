# 10 Qwen3.5-4B 문장 결합 증강 비교

Qwen/Qwen3.5-4B의 비추론 모드에서 시드 43, 44, 45, 46마다 문장 결합 증강 on(확률 0.5)과 off(0.0)를 비교했습니다. 8조건 모두 Unsloth 4비트 QLoRA로 3에포크(588스텝)를 학습하고, 최종 어댑터를 동일한 555건과 평가 배치 8로 평가했습니다. 엄격한 완전 일치율의 시드 평균은 on 90.27%, off 88.65%입니다. [평가 결과](results/README.md)에서 시드별 차이와 유형별 집계를 확인할 수 있습니다.

## 보존 자료

| 위치 | 내용 |
| --- | --- |
| [data/](data/README.md) | 공통 정제본에서 구성한 클러스터, 공종 마스터, 학습 1,568건과 평가 555건 |
| [data/models/](data/models/README.md) | 조건별 최종 어댑터 메타데이터와 로컬 모델 복원 방법 |
| [results/](results/README.md) | 조건별 학습 설정, 학습 이력, 평가 보고서 |

기본 모델 revision은 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`입니다. 보존된 실행은 Windows, Python 3.10.11, PyTorch 2.11.0+cu128, Transformers 5.17.0, PEFT 0.18.1을 사용했고 학습에는 Unsloth 2026.10.9를 사용했습니다. 조건별 실제 라이브러리 버전과 설정은 결과 JSON에 기록되어 있습니다. 8조건의 평가 구현 해시는 현재 평가 코드와 일치합니다. 보존된 학습 구현 해시는 현재 학습 스크립트와 공통 설정 파일로 계산한 해시와 다르므로 당시 학습 구현의 정확한 재현 여부는 확인되지 않았습니다.

## 데스크톱 실행

프로젝트 루트에서 실행합니다. 먼저 보존된 실행 환경과 [공통 설정](experiment_common.py)을 확인하고 평가에 필요한 모델 파일을 복원하세요. 기존 산출물을 보존해야 하는 학습 재실행은 독립된 새 실험 경로에서 진행합니다.

```powershell
.\.venv\Scripts\python.exe experiments/10/run_experiment.py --start 43 --end 46
```

[실행 스크립트](run_experiment.py)는 지정한 시드 범위의 on/off 학습을 순서대로 처리한 뒤 평가하고 비교 집계를 저장합니다. 완료된 학습과 이미 기록된 평가는 건너뜁니다. 최종 어댑터가 있는 조건 중 평가 보고서가 없는 조건만 평가하려면 [evaluate_all_desktop.py](evaluate_all_desktop.py)를 실행합니다. 단일 조건은 [학습 스크립트](train_desktop_unsloth.py) 또는 [평가 스크립트](evaluate_desktop.py)에 `--seed 43 --augmentation on`과 같이 지정합니다.
