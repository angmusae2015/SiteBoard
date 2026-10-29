"""학습과 평가를 각각 별도 프로세스로 실행합니다."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from experiment_common import (
    EXECUTION_PARAMS, EXPERIMENT_PARAMS, PROJECT_ROOT, SCRIPT_DIR,
    build_comparison, condition_settings, display_comparison, read_json,
    runtime_path, training_completed, write_json,
)


STAGE_SCRIPTS = {
    "train": "train_desktop_unsloth.py",
    "evaluate": "evaluate_desktop.py",
}


def execution_steps(all_conditions):
    steps = [
        (seed, mode, stage)
        for seed in EXPERIMENT_PARAMS["seeds"]
        for mode in EXPERIMENT_PARAMS["augmentation_modes"]
        for stage in ("train", "evaluate")
    ]
    if all_conditions:
        return steps
    start = next(
        i for i, (seed, mode, stage) in enumerate(steps)
        if f"seed_{seed}_{mode}" == EXECUTION_PARAMS["start_run"]
        and stage == EXECUTION_PARAMS["start_stage"]
    )
    return steps[start:]


def run_child(seed, mode, stage):
    command = [
        sys.executable, "-u", str(SCRIPT_DIR / STAGE_SCRIPTS[stage]),
        "--seed", str(seed), "--augmentation", mode,
    ]
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    # stdout과 stderr를 상속해 Trainer와 tqdm 진행률을 바로 표시합니다.
    child = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env)
    try:
        returncode = child.wait()
    except KeyboardInterrupt:
        print("\n중단 요청: 실행 중인 자식 프로세스 종료를 기다립니다.", flush=True)
        try:
            child.wait(timeout=5)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=5)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                if child.poll() is None:
                    child.kill()
                child.wait()
        raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def main():
    parser = argparse.ArgumentParser(description="09 시드별 증강 on/off 학습과 평가")
    parser.add_argument("--all", action="store_true", help="시작 위치 제한 없이 전체 조건 확인")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    steps = execution_steps(args.all)
    for index, (seed, mode, stage) in enumerate(steps, start=1):
        label = f"[{index}/{len(steps)}] seed_{seed}_{mode} {stage}"
        train_runtime, _, eval_runtime = condition_settings(seed, mode)
        completed = training_completed(train_runtime)
        if stage == "train" and completed:
            print(f"{label}: 완료된 학습 건너뛰기", flush=True)
            continue
        if stage == "evaluate":
            if not completed:
                raise RuntimeError(f"{label}: 완료된 학습이 없습니다.")
            if Path(runtime_path(eval_runtime, "report_file")).exists():
                print(f"{label}: 완료된 평가 건너뛰기", flush=True)
                continue
        print(f"{label}: 시작", flush=True)
        started = time.perf_counter()
        try:
            run_child(seed, mode, stage)
        except subprocess.CalledProcessError as error:
            print(f"{label}: 실패, 종료 코드 {error.returncode}", flush=True)
            return max(1, error.returncode)
        print(f"{label}: 프로세스 종료, 코드 0, {time.perf_counter() - started:.1f}초", flush=True)

    print("전체 6조건 평가 결과 집계", flush=True)
    destination = Path(EXPERIMENT_PARAMS["summary_file"])
    if destination.exists():
        print(f"기존 비교 집계를 보존합니다: {destination}", flush=True)
        comparison = read_json(destination)
    else:
        comparison = build_comparison()
        write_json(destination, comparison)
        print(f"비교 집계 저장: {destination}", flush=True)
    display_comparison(comparison)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("실행을 중단했습니다. 후속 단계는 실행하지 않습니다.", flush=True)
        sys.exit(130)
