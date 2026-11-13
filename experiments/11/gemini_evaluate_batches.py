import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from gemini_prepare_batches import DATA, MODEL, RESULTS, json_text, request_key, sha256
from gemini_run_batches import response_rows


def reject_duplicates(pairs):
    """JSON 객체의 키와 값을 사전으로 변환합니다. 필드가 반복되면 어떤 값을 채점할지 모호하므로 거부합니다."""
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError('JSON 객체에 중복 필드가 있습니다.')
    return result


def item_counts(items):
    """작업 항목의 필드와 값 종류를 검사한 뒤 항목별 횟수를 반환합니다. 배열 순서가 달라도 중복 개수는 비교합니다."""
    if not isinstance(items, list):
        raise ValueError('응답은 JSON 배열이어야 합니다.')
    counts = Counter()
    for item in items:
        if not isinstance(item, dict) or item.keys() != {'workType', 'spec', 'quantity'}:
            raise ValueError('작업 항목에 필드가 빠졌거나 추가되었습니다.')
        work, spec, quantity = item['workType'], item['spec'], item['quantity']
        if not (isinstance(work, str) and isinstance(spec, str)
                and (type(quantity) is int or type(quantity) is float and math.isfinite(quantity))):
            raise ValueError('공종, 규격 또는 수량의 값 종류가 올바르지 않습니다.')
        counts[work, spec, quantity] += 1
    return counts


def score_response(response, expected):
    """Gemini 응답에서 출력 텍스트를 꺼내 정답 배열과 비교합니다. 비정상 종료와 잘못된 JSON은 오답 사유로 기록합니다."""
    candidates = response.get('candidates') or []
    if not candidates or candidates[0].get('finishReason') != 'STOP':
        reason = candidates[0].get('finishReason') if candidates else None
        return False, None, f'출력 종료 상태: {reason or "응답 후보 없음"}'
    parts = candidates[0].get('content', {}).get('parts', [])
    text = ''.join(part.get('text', '') for part in parts if not part.get('thought'))
    return score_text(text, expected)


def score_text(text, expected, *, array_field=None):
    """출력 JSON을 읽고 정답 항목의 개수와 비교합니다. 지정한 배열 필드가 있으면 그 안의 배열을 채점합니다."""
    try:
        prediction = json.loads(text, object_pairs_hook=reject_duplicates)
        if array_field:
            if not isinstance(prediction, dict) or prediction.keys() != {array_field}:
                raise ValueError(f'응답 객체에 {array_field} 필드 하나가 필요합니다.')
            prediction = prediction[array_field]
        predicted = item_counts(prediction)
    except ValueError as error:
        return False, None, str(error)
    return predicted == expected, prediction, None


def summarize(cases):
    """예정, 평가, 정답과 미평가 건수를 집계합니다. 토큰 통계에는 API가 총 토큰 수를 기록한 응답을 사용합니다."""
    evaluated = [case for case in cases if case['correct'] is not None]
    correct = sum(case['correct'] for case in evaluated)
    token_counts = [case['usage'].get('totalTokenCount') for case in evaluated]
    recorded = [count for count in token_counts if type(count) is int]
    return {'expected': len(cases), 'evaluated': len(evaluated), 'correct': correct,
            'unevaluated': len(cases) - len(evaluated),
            'accuracy': correct / len(evaluated) if evaluated else None,
            'token_records': len(recorded), 'token_missing': len(evaluated) - len(recorded),
            'total_tokens': sum(recorded) if recorded else None,
            'mean_tokens': statistics.mean(recorded) if recorded else None,
            'value_min_confidence': None}


def evaluate(run_dir, report_id, *, read_responses=response_rows, scorer=score_response,
             model=MODEL, sdk_field='google_genai_version'):
    """실행 목록, 원본 test와 저장된 응답을 대조합니다. 케이스 결과와 조건별 요약을 새 파일로 저장합니다."""
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['model'] != model:
        raise ValueError(f'{model} 실행 목록을 지정하세요.')
    state = json.loads((run_dir / 'state.json').read_text(encoding='utf-8'))
    tests = {}
    for seed in {batch['seed'] for batch in manifest['batches']}:
        path = DATA / f'{seed}_test.json'
        if sha256(path) != manifest['input_sha256'][path.name]:
            raise ValueError(f'준비 이후 정답 파일이 변경되었습니다: {path.name}')
        tests[seed] = json.loads(path.read_text(encoding='utf-8'))[:manifest['test_limit']]

    summary_path = run_dir / f'{report_id}.summary.json'
    cases_path = run_dir / f'{report_id}.cases.jsonl'
    if summary_path.exists() or cases_path.exists():
        raise FileExistsError('이미 사용한 결과 이름입니다. 다른 이름을 지정하세요.')
    by_batch, by_condition = [], defaultdict(list)
    with cases_path.open('x', encoding='utf-8', newline='\n') as stream:
        for batch in manifest['batches']:
            attempts = state['batches'].get(batch['id'], [])
            responses, errors = read_responses(run_dir, attempts)
            records = {request_key(batch['id'], record): record for record in tests[batch['seed']]}
            # 알 수 없는 응답 키가 있으면 다른 실행 결과가 섞였을 수 있습니다.
            if len(records) != batch['count'] or (responses.keys() | errors.keys()) - records.keys():
                raise ValueError(f'요청 키 또는 예상 건수 불일치: {batch["id"]}')
            cases = []
            for key, record in records.items():
                response = responses.get(key)
                if response is None:
                    correct, prediction, error = None, None, errors.get(key, '응답 누락')
                else:
                    correct, prediction, error = scorer(response, item_counts(record['y']))
                case = {'key': key, 'site': record['site'], 'image_id': record['image_id'],
                        'correct': correct, 'prediction': prediction, 'expected': record['y'],
                        'error': error, 'usage': (response or {}).get('usageMetadata') or {}}
                stream.write(json_text(case) + '\n')
                cases.append(case)
            metrics = summarize(cases)
            condition = batch['n'], batch['method'], batch['top_k']
            by_batch.append({**batch, **metrics})
            by_condition[condition].extend(cases)

    conditions = []
    for (n, method, top_k), cases in sorted(by_condition.items()):
        matching = [batch for batch in by_batch if (batch['n'], batch['method'], batch['top_k']) == (n, method, top_k)]
        # 각 시드가 모두 평가된 경우에만 정확도 시드 평균과 표준편차를 계산합니다.
        accuracies = [batch['accuracy'] for batch in matching if batch['unevaluated'] == 0]
        complete = len(accuracies) == len(matching)
        conditions.append({'n': n, 'method': method, 'top_k': top_k, **summarize(cases),
                           'seeds': [batch['seed'] for batch in matching], 'complete_seeds': len(accuracies),
                           'seed_mean_accuracy': statistics.mean(accuracies) if complete else None,
                           'seed_accuracy_sd': statistics.stdev(accuracies) if complete and len(accuracies) > 1 else None})
    with summary_path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps({'run_id': manifest['run_id'], 'model': manifest['model'],
                                 sdk_field: state.get(sdk_field),
                                 'batches': by_batch, 'conditions': conditions}, ensure_ascii=False, indent=2) + '\n')
    print(f'{summary_path}: 조건 {len(conditions)}개, 미평가 {sum(row["unevaluated"] for row in conditions)}건')


def main(evaluate_run=evaluate):
    """명령줄에서 실행과 결과 이름을 읽고 평가를 시작합니다. 새 결과 이름을 지정해 기존 평가를 보존합니다."""
    parser = argparse.ArgumentParser(description='실험 11 배치 응답 평가')
    parser.add_argument('run_id')
    parser.add_argument('--report', default='evaluation')
    args = parser.parse_args()
    for name in (args.run_id, args.report):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,29}', name):
            parser.error('이름에는 영문, 숫자, 밑줄, 하이픈을 사용해 30자 이내로 지정하세요.')
    evaluate_run(Path(RESULTS) / args.run_id, args.report)


if __name__ == '__main__':
    main()
