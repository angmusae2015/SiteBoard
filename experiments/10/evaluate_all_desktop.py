"""10번 실험의 최종 어댑터를 각각 별도 프로세스에서 평가합니다."""

import os
import re
import subprocess
from pathlib import Path

from experiment_common import PROJECT_ROOT, TRAIN_RUNTIME_TEMPLATE, condition_settings, runtime_path
from run_experiment import run_child


def main():
    os.chdir(PROJECT_ROOT)
    models_dir = Path(TRAIN_RUNTIME_TEMPLATE["prefix_path"]) / TRAIN_RUNTIME_TEMPLATE["adapter_path"]
    conditions = []
    for weights in models_dir.glob("seed_*_*/adapter_model.safetensors"):
        match = re.fullmatch(r"seed_(\d+)_(on|off)", weights.parent.name)
        if match and weights.is_file():
            conditions.append((int(match[1]), match[2]))

    if not conditions:
        print(f"최종 어댑터 가중치를 찾을 수 없습니다: {models_dir}")
        return 1

    failed = []
    for index, (seed, mode) in enumerate(sorted(conditions), start=1):
        run_id = f"seed_{seed}_{mode}"
        label = f"[{index}/{len(conditions)}] {run_id}"
        _, _, eval_runtime = condition_settings(seed, mode)
        if Path(runtime_path(eval_runtime, "report_file")).exists():
            print(f"{label}: 기존 평가 결과 보존", flush=True)
            continue

        print(f"{label}: 평가 시작", flush=True)
        try:
            run_child(seed, mode, "evaluate")
        except subprocess.CalledProcessError as error:
            print(f"{label}: 평가 실패 (종료 코드 {error.returncode})", flush=True)
            failed.append(run_id)
        else:
            print(f"{label}: 평가 프로세스 종료", flush=True)

    if failed:
        print(f"평가 실패 조건: {', '.join(failed)}", flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("평가를 중단했습니다.", flush=True)
        raise SystemExit(130) from None
