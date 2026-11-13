import argparse
import hashlib
import heapq
import itertools
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'experiments/11/data'
RESULTS = ROOT / 'experiments/11/results'
SEEDS = (42, 43, 44, 45, 46)
N_VALUES = (2, 3, 4, 5)
METHODS = ('jaccard', 'cosine', 'tfidf_cosine')
TOP_K_VALUES = (5, 10, 20, 30)
MODEL = 'gemini-3.5-flash-lite'
THINKING = 'low'

PROMPT_TEMPLATE = """작업 기록을 읽고 공종, 규격, 수량을 추출하세요.
공종은 아래 현장 공종 마스터에 등록된 이름 그대로 사용하세요.
규격 목록에 값이 있으면 그중 하나를 문자열로 출력하세요.
규격 목록이 비어 있으면 빈 문자열을 출력하세요.
규격 규칙이 null이면 기록의 가로와 세로를 숫자*숫자 형식으로 출력하세요.
예시와 현장 공종 마스터가 다르면 현장 공종 마스터를 따르세요.
예시는 해석 방법을 참고하는 자료입니다. 입력 기록에서 확인되는 작업만 추출하세요.
workType과 spec은 문자열, quantity는 숫자로 출력하세요. 응답은 JSON 배열 하나로 작성하세요.

현장 공종 마스터
{master}

검색 예시 목록. 각 예시의 x는 작업 원문, y는 정답 JSON입니다.
{examples}

입력 작업 원문
{work_text}"""

RESPONSE_SCHEMA = {
    'type': 'array',
    'items': {
        'type': 'object',
        'properties': {'workType': {'type': 'string'}, 'spec': {'type': 'string'},
                       'quantity': {'type': 'number'}},
        'required': ['workType', 'spec', 'quantity'],
        'additionalProperties': False,
    },
}


def json_text(value):
    """값을 JSON 한 줄로 변환합니다. 한글을 그대로 기록하고 JSON으로 표현할 수 없는 숫자는 거부합니다."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def sha256(path):
    """파일 내용을 조금씩 읽어 SHA-256 해시를 반환합니다. 실행 이후 입력 변경을 확인하는 데 사용합니다."""
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def record_key(record):
    """레코드에서 현장 코드와 사진 번호를 꺼냅니다. 사진 번호는 현장과 함께 사용해야 고유합니다."""
    return record['site'], record['image_id']


def index_records(records):
    """레코드 목록을 현장과 사진 번호로 찾을 수 있는 사전으로 바꿉니다. 중복 ID는 입력 오류로 처리합니다."""
    indexed = {record_key(record): record for record in records}
    if len(indexed) != len(records):
        raise ValueError('현장과 사진 번호가 같은 레코드가 있습니다.')
    return indexed


def validate_split(train, test, source, master):
    """시드별 분할을 실험 11 원본과 전수 대조합니다. 중복, 누락, 내용 변경이나 공종 및 규격 오류가 있으면 중단합니다."""
    train_index, test_index = index_records(train), index_records(test)
    if train_index.keys() & test_index.keys() or train_index | test_index != source:
        raise ValueError('분할에 교집합, 누락 또는 원본과 다른 레코드가 있습니다.')
    for record in itertools.chain(train, test):
        site = record['site']
        if site not in master or not isinstance(record['x'], str) or not record['x'].strip():
            raise ValueError(f'현장 또는 작업 원문 오류: {record_key(record)}')
        for item in record['y']:
            work, spec = item['workType'], item['spec']
            if work not in master[site] or not isinstance(spec, str):
                raise ValueError(f'공종 또는 규격 오류: {record_key(record)}, {item}')
            rule = master[site][work]
            # 마스터의 고정 규격은 숫자로 저장되어 있으므로 문자열로 비교합니다.
            if rule is None:
                valid = re.fullmatch(r'[0-9]+\*[0-9]+', spec) is not None
            else:
                valid = spec in {str(value) for value in rule} if rule else spec == ''
            if not valid:
                raise ValueError(f'허용 규격과 일치하지 않습니다: {record_key(record)}, {item}')


def normalize(text):
    """검색용 문자열에서 모든 숫자, 문장부호와 기호를 삭제합니다. 글자 표기와 공백을 통일하고 원문은 보존합니다."""
    normalized = unicodedata.normalize('NFC', text).casefold()
    return ' '.join(''.join(char for char in normalized
                            if unicodedata.category(char)[0] not in 'NPS').split())


def grams(text, n):
    """검색용 문자열을 길이 n의 연속 문자 조각으로 나눠 출현 횟수를 반환합니다. 짧은 문자열은 빈 결과가 됩니다."""
    return Counter(text[start:start + n] for start in range(len(text) - n + 1))


def build_index(train_texts, n):
    """train의 문자 조각별 레코드 목록과 각 레코드의 비교 기준을 만듭니다. TF-IDF 가중치도 train에서 계산합니다."""
    vectors = [grams(text, n) for text in train_texts]
    postings = defaultdict(list)
    for index, vector in enumerate(vectors):
        for gram, count in vector.items():
            postings[gram].append((index, count))
    # 자주 등장하는 문자 조각의 영향은 줄이고 드문 조각의 영향은 높입니다.
    weights = {gram: 1 + math.log((1 + len(vectors)) / (1 + len(entries)))
               for gram, entries in postings.items()}
    sizes = [(len(vector), math.hypot(*vector.values()),
              math.hypot(*(count * weights[gram] for gram, count in vector.items())))
             for vector in vectors]
    return postings, weights, sizes


def rank_examples(query, search_index, method, limit):
    """한 질의와 train의 유사도를 계산해 상위 레코드 위치를 반환합니다. 점수가 같으면 앞서 정렬한 현장과 사진 번호 순서를 따릅니다."""
    postings, weights, sizes = search_index
    shared = defaultdict(list)
    # 공통 문자 조각이 없는 후보는 점수가 0입니다. 공통 조각의 기여만 모아 계산합니다.
    for gram, count in query.items():
        for index, train_count in postings.get(gram, ()):
            product = count * train_count
            shared[index].append(product * weights[gram] ** 2 if method == 'tfidf_cosine' else product)
    scores = {}
    if method == 'jaccard':
        for index, products in shared.items():
            intersection = len(products)
            scores[index] = intersection / (len(query) + sizes[index][0] - intersection)
    else:
        weighted = method == 'tfidf_cosine'
        # TF-IDF에서는 train에 등장한 조각만 질의 벡터에 포함합니다.
        norm = (math.hypot(*(count * weights[gram] for gram, count in query.items() if gram in weights))
                if weighted else math.hypot(*query.values()))
        for index, products in shared.items():
            scores[index] = math.fsum(products) / (norm * sizes[index][2 if weighted else 1])
    return heapq.nsmallest(limit, range(len(sizes)), key=lambda index: (-scores.get(index, 0), index))


def request_key(batch_id, record):
    """배치 조건과 레코드 ID를 이어 붙여 요청 키를 반환합니다. 결과 파일의 순서와 관계없이 응답을 찾을 수 있습니다."""
    return f'{batch_id}-{record["site"]}-{record["image_id"]}'


def format_prompt(query, examples, master, template=PROMPT_TEMPLATE):
    """질의 현장 마스터, 검색 예시, 작업 원문을 넣어 프롬프트를 만듭니다. 예시에는 원문과 정답 배열만 담습니다."""
    return template.format(
        master=json_text(master[query['site']]),
        examples=json_text([{'x': record['x'], 'y': record['y']} for record in examples]),
        work_text=json_text(query['x']),
    )


def make_request(key, query, examples, master):
    """공통 프롬프트를 Gemini 요청에 넣고 응답 형식과 사고 수준을 지정합니다."""
    prompt = format_prompt(query, examples, master)
    return {'key': key, 'request': {
        'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
        'generationConfig': {'thinkingConfig': {'thinkingLevel': THINKING},
                             'responseMimeType': 'application/json', 'responseJsonSchema': RESPONSE_SCHEMA},
    }}


def prepare(run_id, seeds, n_values, methods, top_k_values, limit=None, *,
            request_factory=make_request, model=MODEL, thinking=THINKING, template=PROMPT_TEMPLATE):
    """입력을 검증하고 지정한 API의 형식으로 요청 파일을 만듭니다. limit은 사전 확인용 test 건수입니다."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,29}', run_id):
        raise ValueError('실행 ID는 영문, 숫자, 밑줄, 하이픈을 사용해 30자 이내로 지정하세요.')
    master_path, source_path = DATA / 'work_type.json', DATA / 'dataset.json'
    master = json.loads(master_path.read_text(encoding='utf-8'))
    source = index_records(json.loads(source_path.read_text(encoding='utf-8')))
    splits = {}
    hashes = {path.name: sha256(path) for path in (master_path, source_path)}
    # 모든 시드의 원본 분할을 먼저 검사합니다. 사전 확인용 제한은 검사가 끝난 후 적용합니다.
    for seed in seeds:
        train_path, test_path = (DATA / f'{seed}_{name}.json' for name in ('train', 'test'))
        train, test = (json.loads(path.read_text(encoding='utf-8')) for path in (train_path, test_path))
        validate_split(train, test, source, master)
        splits[seed] = sorted(train, key=record_key), test[:limit]
        hashes.update({path.name: sha256(path) for path in (train_path, test_path)})

    run_dir = RESULTS / run_id
    run_dir.mkdir(exist_ok=False)
    manifest = {'model': model, 'thinking': thinking, 'run_id': run_id, 'test_limit': limit,
                'prompt_sha256': hashlib.sha256(template.encode('utf-8')).hexdigest(),
                'input_sha256': hashes, 'batches': []}
    for seed, (train, test) in splits.items():
        train_texts = [normalize(record['x']) for record in train]
        test_texts = [normalize(record['x']) for record in test]
        for n in n_values:
            search_index = build_index(train_texts, n)
            queries = [grams(text, n) for text in test_texts]
            for method in methods:
                batches = []
                with ExitStack() as files:
                    outputs = {}
                    for top_k in top_k_values:
                        batch_id = f's{seed}-n{n}-{method}-k{top_k}'
                        name = f'{batch_id}.requests.jsonl'
                        batches.append({'id': batch_id, 'seed': seed, 'n': n, 'method': method,
                                        'top_k': top_k, 'requests': name, 'count': len(test)})
                        outputs[top_k] = files.enter_context((run_dir / name).open('x', encoding='utf-8', newline='\n'))
                    # 같은 질의의 순위를 한 번 계산해 Top_K 간 예시 순서가 일치하도록 합니다.
                    for record, query in zip(test, queries):
                        ranking = rank_examples(query, search_index, method, max(top_k_values))
                        examples = [train[index] for index in ranking]
                        for batch in batches:
                            top_k = batch['top_k']
                            request = request_factory(request_key(batch['id'], record), record, examples[:top_k], master)
                            outputs[top_k].write(json_text(request) + '\n')
                for batch in batches:
                    batch['request_sha256'] = sha256(run_dir / batch['requests'])
                    manifest['batches'].append(batch)
    # 실행 목록은 요청 파일이 모두 준비된 뒤에만 공개합니다.
    (run_dir / 'manifest.json').write_text(json_text(manifest) + '\n', encoding='utf-8')
    print(f'{run_dir}: 배치 {len(manifest["batches"])}개, 요청 {sum(b["count"] for b in manifest["batches"])}건')


def main(prepare_run=prepare, description='실험 11 Gemini 배치 입력 준비'):
    """명령줄 조건을 읽어 준비를 실행합니다. 선택 옵션을 생략하면 전체 시드와 모든 검색 조합을 생성합니다."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('run_id')
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--n', type=int, choices=N_VALUES)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--top-k', type=int, choices=TOP_K_VALUES)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit은 1 이상이어야 합니다.')
    prepare_run(args.run_id, (args.seed,) if args.seed else SEEDS, (args.n,) if args.n else N_VALUES,
            (args.method,) if args.method else METHODS, (args.top_k,) if args.top_k else TOP_K_VALUES, args.limit)


if __name__ == '__main__':
    main()
