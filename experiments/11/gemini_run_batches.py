import argparse
import json
import re
import time
import uuid
from collections import Counter
from importlib.metadata import version

from gemini_prepare_batches import MODEL, RESULTS, json_text, sha256


TERMINAL = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED',
            'JOB_STATE_EXPIRED', 'JOB_STATE_PARTIALLY_SUCCEEDED'}


def read_rows(path):
    """JSONL 파일을 읽어 요청 키와 해당 행을 연결합니다. 같은 키가 반복되면 결과를 잘못 연결하지 않도록 중단합니다."""
    rows = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                key = row['key']
                if key in rows:
                    raise ValueError(f'{path.name}: 요청 키 중복 {key}')
                rows[key] = row
    return rows


def response_rows(run_dir, attempts):
    """시도별 응답 파일에서 모델이 반환한 응답을 모읍니다. 재시도 대상인 요청 오류는 별도로 돌려줍니다."""
    responses, errors = {}, {}
    for attempt in attempts:
        if 'result' not in attempt:
            continue
        for key, row in read_rows(run_dir / attempt['result']).items():
            if isinstance(row.get('response'), dict):
                if key in responses:
                    raise ValueError(f'두 배치에서 같은 요청에 응답했습니다: {key}')
                responses[key] = row['response']
            else:
                errors[key] = row.get('error', row.get('status', row))
    return responses, errors


def save_state(path, state):
    """상태를 임시 파일에 먼저 기록하고 교체합니다. 기록 중 중단되어도 이전 상태를 다시 읽을 수 있습니다."""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json_text(state) + '\n', encoding='utf-8')
    temporary.replace(path)


def refresh(client, run_dir, attempts, state_path, state):
    """원격 작업 상태를 갱신하고 완료 파일을 저장합니다. 생성 응답이 유실된 작업은 고유하게 지정한 표시 이름으로 찾아 연결합니다."""
    for attempt in attempts:
        if attempt.get('settled'):
            continue
        if 'name' not in attempt:
            # 생성 요청을 다시 보내기 전에 실제 작업이 이미 접수되었는지 확인합니다.
            matches = [job for job in client.batches.list()
                       if job.display_name == attempt['display_name']]
            if len(matches) > 1:
                raise ValueError(f'표시 이름이 같은 원격 작업이 여러 개입니다: {attempt["display_name"]}')
            if not matches:
                print(f'제출 확인 대기: {attempt["display_name"]}', flush=True)
                continue
            attempt['name'] = matches[0].name
            save_state(state_path, state)
        job = client.batches.get(name=attempt['name'])
        attempt['remote_state'] = job.state.name if job.state else 'SUBMITTED'
        if attempt['remote_state'] in TERMINAL:
            if job.dest and job.dest.file_name:
                result = f'{attempt["display_name"]}.responses.jsonl'
                target = run_dir / result
                if not target.exists():
                    temporary = target.with_suffix('.tmp')
                    temporary.write_bytes(client.files.download(file=job.dest.file_name))
                    temporary.replace(target)
                attempt['result'] = result
            if job.error:
                attempt['error'] = job.error.model_dump(mode='json', exclude_none=True)
            attempt['settled'] = True
        save_state(state_path, state)


def submit(client, api_errors, run_dir, batch, attempts, manifest, state_path, state):
    """첫 실행에는 전체 요청을, 재시도에는 응답이 없는 요청만 업로드합니다. 접수 전에 시도 정보를 저장합니다."""
    original = run_dir / batch['requests']
    if sha256(original) != batch['request_sha256']:
        raise ValueError(f'준비 이후 요청 파일이 변경되었습니다: {original.name}')
    attempt_id = f'{batch["id"]}-a{len(attempts) + 1}'
    # 다른 실행에서도 같은 검색 조건을 제출할 수 있으므로 원격 표시 이름에는 무작위 식별자를 붙입니다.
    display_name = f'{attempt_id}-{uuid.uuid4().hex[:16]}'
    upload = original
    if attempts:
        requests = read_rows(original)
        responses, _ = response_rows(run_dir, attempts)
        pending = requests.keys() - responses.keys()
        upload = run_dir / f'{attempt_id}.retry.jsonl'
        if not upload.exists():
            temporary = upload.with_suffix('.tmp')
            with temporary.open('w', encoding='utf-8', newline='\n') as stream:
                for key in requests:
                    if key in pending:
                        stream.write(json_text(requests[key]) + '\n')
            temporary.replace(upload)
    try:
        uploaded = client.files.upload(file=upload, config={'mime_type': 'jsonl', 'display_name': display_name})
    except api_errors.APIError as error:
        if error.code == 429:
            return False
        raise
    attempt = {'display_name': display_name, 'source_file': uploaded.name}
    attempts.append(attempt)
    save_state(state_path, state)
    try:
        job = client.batches.create(model=manifest['model'], src=uploaded.name,
                                    config={'display_name': display_name})
    except api_errors.APIError as error:
        # 한도 초과는 접수 거절이 명확하므로, 다음 조회 주기에 다시 제출할 수 있습니다.
        if error.code == 429:
            attempts.pop()
            save_state(state_path, state)
            client.files.delete(name=uploaded.name)
            return False
        # 다른 4xx 응답도 거절된 요청입니다. 요청을 고친 뒤 같은 실행을 이어갈 수 있게 합니다.
        if 400 <= error.code < 500 and error.code != 408:
            attempts.pop()
            save_state(state_path, state)
        raise
    attempt.update(name=job.name, remote_state=job.state.name if job.state else 'SUBMITTED')
    save_state(state_path, state)
    print(f'제출: {display_name} ({job.name})', flush=True)
    return True


def progress(run_dir, batches, state, retries):
    """배치별로 대기, 실행, 완료, 실패를 판정합니다. 다음에 제출할 조건과 전체 응답 건수를 반환합니다."""
    counts, ready = Counter(), []
    collected = 0
    for batch in batches:
        attempts = state['batches'].get(batch['id'], [])
        responses, _ = response_rows(run_dir, attempts)
        collected += len(responses)
        if attempts and not attempts[-1].get('settled'):
            counts['active'] += 1
        elif len(responses) == batch['count']:
            counts['complete'] += 1
        elif len(attempts) <= retries:
            counts['queued'] += 1
            ready.append(batch)
        else:
            counts['failed'] += 1
    return counts, ready, collected


def run(args, run_dir, manifest):
    """상태 파일과 원격 작업을 연결합니다. 명령에 따라 한 번 조회하거나 배치가 모두 끝날 때까지 제출과 조회를 반복합니다."""
    from google import genai
    from google.genai import errors, types

    # SDK에서 생성 요청을 자동으로 반복하면 같은 배치가 두 번 접수될 수 있습니다.
    client = genai.Client(http_options=types.HttpOptions(
        api_version='v1beta', retry_options=types.HttpRetryOptions(attempts=1)))
    state_path = run_dir / 'state.json'
    state = (json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists()
             else {'batches': {}})
    state['google_genai_version'] = version('google-genai')
    batches = manifest['batches']
    while True:
        try:
            for attempts in state['batches'].values():
                refresh(client, run_dir, attempts, state_path, state)
            counts, ready, collected = progress(run_dir, batches, state, args.retries)
            print(f'배치 {len(batches)}개 / 대기 {counts["queued"]} / 실행 {counts["active"]} / '
                  f'완료 {counts["complete"]} / 실패 {counts["failed"]} / '
                  f'수집 응답 {collected}/{sum(batch["count"] for batch in batches)}', flush=True)
            if args.action == 'status' or not (counts['active'] or ready):
                return 1 if counts['failed'] and not (counts['active'] or ready) else 0
            slots = max(0, args.max_active - counts['active'])
            for batch in ready[:slots]:
                attempts = state['batches'].setdefault(batch['id'], [])
                if not submit(client, errors, run_dir, batch, attempts, manifest, state_path, state):
                    print('API 한도에 도달했습니다. 다음 조회에서 다시 확인합니다.', flush=True)
                    break
            if args.action == 'submit':
                return 0
        except errors.APIError as error:
            if args.action != 'watch' or (error.code != 429 and error.code < 500):
                raise
            print(f'일시적인 API 응답 오류 {error.code}. 다음 조회에서 다시 확인합니다.', flush=True)
        time.sleep(args.poll_seconds)


def main():
    """명령줄 옵션을 검증하고 지정된 실행을 엽니다. 잠금 파일을 사용해 같은 실행에 대한 동시 제출을 막습니다."""
    parser = argparse.ArgumentParser(description='실험 11 Gemini 배치 제출 및 진행 확인')
    parser.add_argument('action', choices=('submit', 'status', 'watch'))
    parser.add_argument('run_id')
    parser.add_argument('--max-active', type=int, default=4)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--retries', type=int, default=1)
    args = parser.parse_args()
    if args.max_active < 1 or args.poll_seconds < 1 or args.retries < 0:
        parser.error('동시 실행 수와 조회 주기는 1 이상, 재시도 횟수는 0 이상이어야 합니다.')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,29}', args.run_id):
        parser.error('올바른 실행 ID를 지정하세요.')
    run_dir = RESULTS / args.run_id
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['model'] != MODEL:
        parser.error('Gemini 실행 목록을 지정하세요.')
    lock = run_dir / 'runner.lock'
    try:
        with lock.open('x', encoding='utf-8'):
            pass
    except FileExistsError:
        parser.error(f'실행 잠금이 있습니다: {lock}')
    try:
        return run(args, run_dir, manifest)
    finally:
        lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
