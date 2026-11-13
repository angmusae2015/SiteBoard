import argparse
import json
import re
import time
from collections import Counter
from importlib.metadata import version
from uuid import uuid4

from gemini_prepare_batches import RESULTS, sha256
from gemini_run_batches import save_state as save_state_file
from gpt_prepare_batches import ENDPOINT, MODEL


TERMINAL = {'completed', 'failed', 'expired', 'cancelled'}

def save_state(path, state):
    """상태를 저장합니다. Windows가 일시적으로 파일 교체를 막으면 잠시 기다렸다 다시 시도합니다."""
    for attempt in range(5):
        try:
            save_state_file(path, state)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(1)


def print_progress(manifest, state):
    """저장된 상태로 대기, 접수, 완료 및 실패 수를 표시합니다. OpenAI에 상태를 조회하지 않습니다."""
    counts = Counter(job['status'] for job in state['batches'].values())
    waiting = len(manifest['batches']) - len(state['batches']) + counts['waiting']
    active = counts['submitted'] + counts['submitting']
    failed = sum(counts[name] for name in ('failed', 'expired', 'cancelled'))
    received = sum((job.get('request_counts') or {}).get('completed', 0) for job in state['batches'].values())
    expected = sum(batch['count'] for batch in manifest['batches'])
    print(f'배치 {len(manifest["batches"])}개 / 대기 {waiting} / 접수 또는 확인 중 {active} / '
          f'완료 {counts["completed"]} / 실패 {failed} / 성공 응답 {received}/{expected}', flush=True)


def submit_available(client, run_dir, manifest, state, max_active):
    """현재 실행 가능한 수만큼 대기 배치를 제출합니다. 접수 여부가 불명확한 생성 요청은 다시 보내지 않습니다."""
    from openai import APIStatusError

    jobs = state['batches']
    slots = max_active - sum(job['status'] in {'submitting', 'submitted'} for job in jobs.values())
    for batch in manifest['batches']:
        if slots <= 0:
            break
        job = jobs.get(batch['id'])
        if job and job['status'] != 'waiting':
            continue
        if job and job.get('wait_for'):
            active_ids = {other.get('batch_id') for other in jobs.values()
                          if other['status'] in {'submitting', 'submitted'}}
            if set(job['wait_for']) <= active_ids:
                continue
        request_path = run_dir / batch['requests']
        if sha256(request_path) != batch['request_sha256']:
            raise ValueError(f'준비 이후 요청 파일이 변경되었습니다: {request_path.name}')
        if not job:
            uploaded = client.files.create(file=request_path, purpose='batch')
            job = jobs[batch['id']] = {'status': 'waiting', 'file_id': uploaded.id, 'submission_id': uuid4().hex}
        # 원격 응답을 받기 전에 식별자를 보존합니다. 응답이 유실돼도 원격 배치 목록에서 해당 제출을 찾습니다.
        job['status'] = 'submitting'
        save_state(run_dir / 'state.json', state)
        try:
            remote = client.with_options(max_retries=0).batches.create(
                input_file_id=job['file_id'], endpoint=ENDPOINT, completion_window='24h',
                metadata={'siteboard_submission': job['submission_id']},
            )
        except APIStatusError as error:
            if error.status_code in {400, 401, 403, 404, 422, 429}:
                job['status'] = 'waiting'
                save_state(run_dir / 'state.json', state)
            raise
        job.update(status='submitted', batch_id=remote.id)
        save_state(run_dir / 'state.json', state)
        print(f'제출 {batch["id"]} ({remote.id})', flush=True)
        slots -= 1


def refresh(client, run_dir, state):
    """접수된 배치를 조회해 종료된 작업의 출력 및 오류 파일을 저장합니다."""
    jobs = state['batches']
    uncertain = {job['submission_id']: job for job in jobs.values() if job['status'] == 'submitting'}
    if uncertain:
        # 생성 응답이 유실된 경우 원격 목록에서 제출 식별자를 대조합니다.
        for remote in client.batches.list():
            submission = (remote.metadata or {}).get('siteboard_submission')
            if submission in uncertain:
                job = uncertain.pop(submission)
                job.update(status='submitted', batch_id=remote.id)
                save_state(run_dir / 'state.json', state)
                if not uncertain:
                    break
        for submission in uncertain:
            print(f'원격 접수 확인 대기: {submission}', flush=True)

    for key, job in jobs.items():
        if job['status'] != 'submitted':
            continue
        remote = client.batches.retrieve(job['batch_id'])
        if remote.status not in TERMINAL:
            continue
        # 결과를 모두 저장한 뒤 종료 상태를 기록합니다. 재개 시 이미 내려받은 파일을 사용합니다.
        for file_id, field, suffix in ((remote.output_file_id, 'result', 'responses'),
                                       (remote.error_file_id, 'error_file', 'errors')):
            if file_id:
                target = run_dir / f'{key}.{suffix}.jsonl'
                if not target.exists():
                    temporary = target.with_suffix('.tmp')
                    client.files.content(file_id).write_to_file(temporary)
                    temporary.replace(target)
                job[field] = target.name
        job.update(status=remote.status, settled=True,
                   request_counts=remote.request_counts.model_dump() if remote.request_counts else None,
                   batch_errors=remote.errors.model_dump() if remote.errors else None)
        save_state(run_dir / 'state.json', state)

    # 대기 토큰 한도로 접수 직후 거절된 작업은 기존 기록을 보관하고 다시 제출합니다.
    for job in jobs.values():
        errors = (job.get('batch_errors') or {}).get('data') or []
        if job['status'] == 'failed' and any(error.get('code') == 'token_limit_exceeded' for error in errors):
            history = job.get('history', [])
            history.append({key: value for key, value in job.items() if key != 'history'})
            file_id = job['file_id']
            wait_for = [other['batch_id'] for other in jobs.values()
                        if other.get('batch_id') and other['status'] == 'submitted']
            job.clear()
            job.update(status='waiting', file_id=file_id, submission_id=uuid4().hex,
                       history=history, wait_for=wait_for)
            save_state(run_dir / 'state.json', state)


def run(run_dir, manifest, state, args):
    """명령에 따라 한 번 조회하거나, 모든 조건이 종료될 때까지 조회와 제출을 반복합니다."""
    from openai import APIConnectionError, APIStatusError, OpenAI

    state['openai_version'] = version('openai')
    save_state(run_dir / 'state.json', state)
    with OpenAI(timeout=30) as client:
        while True:
            try:
                refresh(client, run_dir, state)
                print_progress(manifest, state)
                jobs = state['batches']
                done = len(jobs) == len(manifest['batches']) and all(
                    job['status'] in TERMINAL for job in jobs.values())
                if args.action == 'status' or done:
                    return int(done and any(job['status'] != 'completed' for job in jobs.values()))
                submit_available(client, run_dir, manifest, state, args.max_active)
                if args.action == 'submit':
                    return 0
            except APIStatusError as error:
                if args.action != 'watch' or (error.status_code not in {408, 429} and error.status_code < 500):
                    raise
                print(f'일시적인 API 응답 HTTP {error.status_code}. 다음 주기에 다시 확인합니다.', flush=True)
            except APIConnectionError:
                if args.action != 'watch':
                    raise
                print('API 연결 오류. 다음 주기에 다시 확인합니다.', flush=True)
            time.sleep(args.poll_seconds)


def main():
    """실행 폴더를 선택합니다. 배치 조회와 제출 중 지정한 작업을 실행합니다."""
    parser = argparse.ArgumentParser(description='실험 11 GPT 배치 제출 및 상태 조회')
    parser.add_argument('action', choices=('submit', 'status', 'watch'))
    parser.add_argument('run_id')
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--max-active', type=int, default=4)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,29}', args.run_id) or min(args.max_active, args.poll_seconds) < 1:
        parser.error('실행 ID, 동시 실행 수와 조회 주기를 확인하세요.')
    run_dir = RESULTS / args.run_id
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['model'] != MODEL:
        parser.error('GPT 실행 목록을 지정하세요.')
    state_path = run_dir / 'state.json'
    lock = run_dir / 'gpt-runner.lock'
    try:
        with lock.open('x', encoding='utf-8'):
            pass
    except FileExistsError:
        if args.action == 'status':
            state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'batches': {}}
            print_progress(manifest, state)
            return 0
        parser.error(f'다른 실행이 상태를 관리하고 있습니다: {lock}')
    try:
        state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'batches': {}}
        return run(run_dir, manifest, state, args)
    finally:
        lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
