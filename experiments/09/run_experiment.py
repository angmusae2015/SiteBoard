"""학습과 평가를 각각 별도 프로세스로 실행합니다."""

import argparse
import os
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

from experiment_common import (
    EXPERIMENT_PARAMS, PROJECT_ROOT, SCRIPT_DIR,
    build_comparison, condition_settings, display_comparison, read_json,
    runtime_path, training_completed, write_json,
)


STAGE_SCRIPTS = {
    "train": "train_desktop_unsloth.py",
    "evaluate": "evaluate_desktop.py",
}


def execution_steps(args: Namespace):
    steps = [
        (seed, mode, stage)
        for stage in ("train", "evaluate")
        for seed in range(args.start, args.end + 1)
        for mode in EXPERIMENT_PARAMS["augmentation_modes"]
    ]

    return steps


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


def step_briefing(steps):
    print(f"실행 예정 스텝: 총 {len(steps)}개", flush=True)
    for index, (seed, mode, stage) in enumerate(steps, start=1):
        augmentation = "적용" if mode == "on" else "미적용"
        stage_name = {"train": "학습", "evaluate": "평가"}[stage]
        print(
            f"[{index}/{len(steps)}] 시드: {seed}, "
            f"증강: {augmentation}, 단계: {stage_name}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description="09 시드별 증강 on/off 학습과 평가")
    parser.add_argument("--start", help="시작 시드", type=int, required=True)
    parser.add_argument("--end", help="마지막 시드", type=int, required=True)
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("--start는 --end 이하여야 합니다.")

    os.chdir(PROJECT_ROOT)
    steps = execution_steps(args)
    step_briefing(steps)

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

    seeds = list(range(args.start, args.end + 1))
    destination = Path(EXPERIMENT_PARAMS["summary_file"])
    if destination.exists():
        comparison = read_json(destination)
        if comparison.get("seeds") == seeds:
            print(f"기존 비교 집계를 보존합니다: {destination}", flush=True)
            display_comparison(comparison)
            return 0

    destination = destination.with_stem(f"{destination.stem}_{args.start}_{args.end}")
    if destination.exists():
        print(f"기존 비교 집계를 보존합니다: {destination}", flush=True)
        comparison = read_json(destination)
    else:
        comparison = build_comparison(seeds)
        write_json(destination, comparison)
        print(f"비교 집계 저장: {destination}", flush=True)
    display_comparison(comparison)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("실행을 중단했습니다.", flush=True)
        sys.exit(130)
