# 실험 11 문자 N-gram 검색 예시 비교

작업 원문과 비슷한 train 기록을 프롬프트 예시로 제공하고, 검색 조건에 따른 완전 일치율과 토큰 사용량을 비교합니다. Gemini는 `gemini-3.5-flash-lite`, GPT는 `gpt-6.1-terra`를 사용합니다. 두 모델 모두 추론 수준은 `low`입니다.

| 항목 | 값 |
| --- | --- |
| 문자 N-gram 길이 | 2, 3, 4, 5 |
| 유사도 | Jaccard, 출현 횟수 Cosine, TF-IDF Cosine |
| 검색 예시 수 | 5, 10, 20, 30 |
| 분할 시드 | 42, 43, 44, 45, 46 |

48개 검색 조건을 5개 시드에서 각각 비교합니다. 논리 배치는 총 240개이고 배치당 test 428건으로 최초 요청은 102,720건입니다. [데이터 안내](data/README.md)에 분할과 공종 마스터 검증 기준이 있습니다.

시드 42는 Gemini `seed42-01`과 GPT `seed42-02`에서 각각 48개 조건의 평가를 완료했습니다. 두 모델 모두 N=2, TF-IDF Cosine, Top_K=30에서 가장 높은 완전 일치율을 기록했습니다. 시드 43~46의 결과는 미기록입니다. [실험 결과](results/README.md)

## 검색과 프롬프트

- 검색용 문자열 양쪽에서 숫자, 문장부호와 기호를 삭제합니다. 문자 표기, 대소문자, 공백을 통일한 뒤 길이 `N`의 연속 문자 조각을 만듭니다.
- 해당 시드의 전체 train을 검색합니다. Jaccard는 조각의 집합을, Cosine은 출현 횟수를 비교합니다. TF-IDF Cosine은 train에서 계산한 가중치를 사용합니다. 공통 조각이 없으면 점수는 0이며 동점은 현장 코드와 사진 번호 순으로 정렬합니다. 같은 검색 순위의 앞부분을 각 Top_K에 사용합니다.
- [Gemini 배치 준비 코드](gemini_prepare_batches.py) 상단의 `PROMPT_TEMPLATE`에서 공통 지시문을 확인할 수 있습니다. 예시에는 작업 원문과 정답 JSON만 들어갑니다. 입력 작업 원문과 해당 현장의 공종 마스터 및 적용 규칙을 함께 제공합니다. [GPT 배치 준비 코드](gpt_prepare_batches.py)는 같은 작업 배열을 구조화 출력의 `items` 필드에 담도록 요청합니다.

## Gemini 실행

프로젝트 루트에서 실행합니다. API 키는 `GEMINI_API_KEY` 환경 변수에 설정합니다. 프로젝트 루트에 정의된 `gemini` 선택 의존성을 설치합니다.

```powershell
python -m pip install -e '.[gemini]'
```

먼저 한 조건의 두 건으로 JSONL 형식, Gemini 응답, 채점과 토큰 집계를 확인할 수 있습니다.

```powershell
python experiments/11/gemini_prepare_batches.py pilot --seed 42 --n 3 --method cosine --top-k 5 --limit 2
python experiments/11/gemini_run_batches.py watch pilot
python experiments/11/gemini_evaluate_batches.py pilot
```

전체 실험은 별도 실행 ID로 시작합니다.

```powershell
python experiments/11/gemini_prepare_batches.py full-01
python experiments/11/gemini_run_batches.py watch full-01
python experiments/11/gemini_evaluate_batches.py full-01
```

`watch`는 제출, 상태 조회, 응답 저장을 반복합니다. `submit`은 한 차례 제출하고, `status`는 한 차례 조회하고 완료 응답을 저장합니다. 동시에 활성화할 배치는 `--max-active`로 지정하며 기본은 4개입니다. API가 한도 초과를 반환하면 다음 조회 주기에 제출을 다시 시도합니다. 요청 오류와 누락은 기본 한 번 재시도하며 `--retries`로 조정할 수 있습니다. 같은 실행 ID로 `watch`를 다시 실행하면 저장된 배치 상태에서 이어갑니다. 이미 `watch`가 동작 중이면 해당 터미널에서 진행률을 확인합니다.

실행 목록과 요청, 원격 응답, 평가 파일은 [results](results/README.md) 아래 실행 ID별로 저장됩니다. 중단된 평가를 다시 수행하려면 `--report`에 새 이름을 지정합니다.

## GPT 실행

프로젝트 루트에서 GPT 선택 의존성을 설치합니다.

```powershell
python -m pip install -e '.[gpt]'
```

실행 환경에 `OPENAI_API_KEY`를 설정합니다. 아래 명령은 새 실행 ID로 시드 42의 48개 조건을 생성하고 실행합니다. 이미 요청 파일이 준비되어 있으면 해당 실행 ID로 `watch`부터 실행합니다. 2건으로 확인할 때는 준비 명령에 `--n 3 --method cosine --top-k 5 --limit 2`를 붙이고 별도 실행 ID를 사용합니다.

```powershell
python experiments/11/gpt_prepare_batches.py gpt-seed42-new --seed 42
python experiments/11/gpt_run_batches.py watch gpt-seed42-new
```

`watch`는 대기 배치를 제출하고, 접수된 배치의 상태를 주기적으로 조회해 종료 시 출력 파일과 오류 파일을 저장합니다. 동시 실행 수는 기본 4개이고 조회 주기는 60초입니다. 각각 `--max-active`와 `--poll-seconds`로 조정합니다. `submit`은 한 번 조회하고 빈 자리에 배치를 제출합니다. 같은 실행 ID로 `watch`를 다시 실행하면 저장한 배치 ID로 이어서 처리합니다.

`status`는 한 번 조회합니다. `watch`가 실행 중이면 다른 터미널의 `status`는 저장된 진행 상태를 표시합니다. 모든 배치가 종료되고 다운로드가 끝나면 평가를 실행합니다.

```powershell
python experiments/11/gpt_run_batches.py status gpt-seed42-new
python experiments/11/gpt_evaluate_batches.py gpt-seed42-new
```
