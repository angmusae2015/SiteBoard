from gemini_prepare_batches import PROMPT_TEMPLATE as SHARED_PROMPT
from gemini_prepare_batches import RESPONSE_SCHEMA, format_prompt, main, prepare


MODEL = 'gpt-6.1-terra'
THINKING = 'low'
ENDPOINT = '/v1/responses'

# OpenAI 구조화 출력은 최상위 객체가 필요하므로 동일한 작업 배열을 items 필드에 넣습니다.
PROMPT_TEMPLATE = SHARED_PROMPT.replace(
    '응답은 JSON 배열 하나로 작성하세요.',
    '응답은 items 필드 하나를 가진 JSON 객체로 작성하고, items에 작업 배열을 넣으세요.',
)
OUTPUT_FORMAT = {
    'type': 'json_schema', 'name': 'siteboard_work_items', 'strict': True,
    'schema': {'type': 'object', 'properties': {'items': RESPONSE_SCHEMA},
               'required': ['items'], 'additionalProperties': False},
}


def make_request(key, query, examples, master):
    """기존 검색 예시와 현장 마스터를 OpenAI Batch 요청으로 변환합니다. custom_id로 케이스를 연결합니다."""
    return {'custom_id': key, 'method': 'POST', 'url': ENDPOINT, 'body': {
        'model': MODEL, 'input': format_prompt(query, examples, master, PROMPT_TEMPLATE),
        'reasoning': {'effort': THINKING}, 'text': {'format': OUTPUT_FORMAT}, 'store': False,
    }}


def prepare_run(run_id, seeds, n_values, methods, top_k_values, limit=None):
    """Gemini와 같은 입력 검사와 검색 순위를 사용해 GPT용 배치 파일 및 실행 목록을 생성합니다."""
    prepare(run_id, seeds, n_values, methods, top_k_values, limit,
            request_factory=make_request, model=MODEL, thinking=THINKING, template=PROMPT_TEMPLATE)


if __name__ == '__main__':
    main(prepare_run, '실험 11 GPT 배치 입력 준비')
