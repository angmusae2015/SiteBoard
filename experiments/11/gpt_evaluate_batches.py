import json

from gemini_evaluate_batches import evaluate, main, score_text
from gpt_prepare_batches import MODEL


def response_rows(run_dir, job):
    """OpenAI의 원본 출력 및 오류 파일을 읽어 요청 키로 연결합니다. HTTP 오류는 미평가 사유로 반환합니다."""
    responses, errors = {}, {}
    for field in ('result', 'error_file'):
        if field not in job:
            continue
        with (run_dir / job[field]).open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row['custom_id']
                reply = row.get('response') or {}
                body = reply.get('body') or {}
                if reply.get('status_code') != 200 or row.get('error') or body.get('error'):
                    errors[key] = row.get('error') or body.get('error') or f'HTTP {reply.get("status_code")}'
                    continue
                if key in responses:
                    raise ValueError(f'같은 요청에 성공 응답이 여러 개 있습니다: {key}')
                usage = body.get('usage') or {}
                # 공통 집계가 읽는 필드로 사용량을 연결합니다. 추론 토큰은 출력 토큰에 이미 포함됩니다.
                responses[key] = {**body, 'usageMetadata': {
                    'totalTokenCount': usage.get('total_tokens'),
                    'input_tokens': usage.get('input_tokens'),
                    'output_tokens': usage.get('output_tokens'),
                    'reasoning_tokens': (usage.get('output_tokens_details') or {}).get('reasoning_tokens'),
                }}
    return responses, errors


def score_response(response, expected):
    """완료된 GPT 메시지의 텍스트에서 작업 배열을 꺼내 공통 완전 일치 기준으로 채점합니다."""
    if response.get('status') != 'completed':
        return False, None, f'출력 종료 상태: {response.get("status")}'
    texts = []
    for output in response.get('output') or []:
        if output.get('type') == 'message':
            texts.extend(part['text'] for part in output.get('content') or [] if part.get('type') == 'output_text')
    return score_text(''.join(texts), expected, array_field='items')


def evaluate_run(run_dir, report_id):
    """GPT 원본 결과와 실험 11의 정답 파일을 비교해 공통 형식으로 케이스별 결과 및 집계를 저장합니다."""
    evaluate(run_dir, report_id, read_responses=response_rows, scorer=score_response,
             model=MODEL, sdk_field='openai_version')


if __name__ == '__main__':
    main(evaluate_run)
