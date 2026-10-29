"""09 실험 설정, 파일 입출력과 결과 집계. GPU 라이브러리는 가져오지 않습니다."""

import argparse
import hashlib
import json
import statistics
from collections import Counter
from copy import deepcopy
from pathlib import Path

EXPERIMENT_PARAMS = {'experiment_dir': 'experiments/09',
 'seeds': [43, 44, 45],
 'augmentation_modes': {'on': 0.5, 'off': 0.0},
 'summary_file': 'experiments/09/data/result/comparison.json',
 'expected_train_count': 1568,
 'expected_test_count': 555,
 'split_seed': 42,
 'excluded_cluster': 22}

TRAIN_RUNTIME_TEMPLATE = {'prefix_path': 'experiments/09/data',
 'model_name': 'Qwen/Qwen3-8B',
 'train_file': 'train.jsonl',
 'test_file': 'test.jsonl',
 'output_dir': 'models/checkpoints',
 'adapter_path': 'models',
 'run_config_file': 'result/run_config.json',
 'training_history_file': 'result/training_history.json',
 'report_file': 'result/evaluation.json',
 'device': 'cuda',
 'model_revision': 'b968826d9c46dd6066d109eabc6255188de91218'}

TRAIN_HYPER_TEMPLATE = {'augmentation': {'probability': 0.0,
                  'max_concat': 2,
                  'delimiters': [' ', '  ', ' / ', ' . ', '\n', ' , ', ' + ']},
 'chat_template': {'enable_thinking': False},
 'training': {'epochs': 3,
              'batch_size': 1,
              'grad_accum': 8,
              'learning_rate': 0.0002,
              'max_seq_length': 1280,
              'seed': 43},
 'lora': {'r': 16,
          'lora_alpha': 32,
          'lora_dropout': 0.05,
          'task_type': 'CAUSAL_LM',
          'bias': 'none',
          'target_modules': ['q_proj',
                             'k_proj',
                             'v_proj',
                             'o_proj',
                             'gate_proj',
                             'up_proj',
                             'down_proj']},
 'quantization': {'load_in_4bit': True,
                  'bnb_4bit_quant_type': 'nf4',
                  'bnb_4bit_use_double_quant': True},
 'precision': {'compute_dtype': 'auto', 'frozen_embedding_dtype': 'auto'},
 'model_loading': {'device_map': {'': 0},
                   'attn_implementation': 'sdpa',
                   'low_cpu_mem_usage': True,
                   'use_exact_model_name': True,
                   'fix_tokenizer': False,
                   'offload_embedding': False},
 'tokenizer': {'padding_side': 'right'},
 'kbit_training': {'use_gradient_checkpointing': 'unsloth'},
 'trainer': {'per_device_eval_batch_size': 1,
             'lr_scheduler_type': 'cosine',
             'warmup_steps': 10,
             'logging_steps': 10,
             'save_strategy': 'epoch',
             'eval_strategy': 'epoch',
             'save_total_limit': 2,
             'optim': 'paged_adamw_8bit',
             'dataloader_num_workers': 0,
             'dataloader_pin_memory': True,
             'report_to': 'none',
             'remove_unused_columns': False,
             'prediction_loss_only': True,
             'label_names': ['labels']},
 'collator': {'pad_to_multiple_of': 8, 'return_tensors': 'pt', 'padding': True}}

EVAL_RUNTIME_TEMPLATE = {'prefix_path': 'experiments/09/data',
 'model_name': 'Qwen/Qwen3-8B',
 'model_revision': 'b968826d9c46dd6066d109eabc6255188de91218',
 'adapter_path': 'models',
 'test_file': 'test.jsonl',
 'split_meta_file': 'split_meta.json',
 'run_config_file': 'result/run_config.json',
 'report_file': 'result/evaluation.json',
 'device': 'cuda'}

EVAL_HYPER_PARAMS = {'chat_template': {'enable_thinking': False},
 'quantization': {'load_in_4bit': True,
                  'bnb_4bit_quant_type': 'nf4',
                  'bnb_4bit_use_double_quant': True},
 'precision': {'compute_dtype': 'auto', 'frozen_embedding_dtype': 'auto'},
 'model_loading': {'device_map': {'': 0}, 'attn_implementation': 'sdpa', 'low_cpu_mem_usage': True},
 'tokenizer': {'padding_side': 'left'},
 'evaluation': {'batch_size': 4, 'max_new_tokens': 512, 'do_sample': False}}

# --all 옵션을 사용하면 이 시작 위치를 무시하고 전체 조건을 확인합니다.
EXECUTION_PARAMS = {"start_run": "seed_44_off", "start_stage": "evaluate"}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent


def runtime_path(params, field):
    return (Path(params["prefix_path"]) / params[field]).as_posix()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value, mode="x"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_sha256(*names):
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update((SCRIPT_DIR / name).read_bytes())
    return digest.hexdigest()


def input_hashes():
    base = Path(TRAIN_RUNTIME_TEMPLATE["prefix_path"])
    return {(base / name).as_posix(): file_sha256(base / name) for name in
            ("work_type.json", "cluster.json", "split_meta.json", "train.jsonl", "test.jsonl")}


def artifact_hashes(adapter):
    return {name: file_sha256(Path(adapter) / name) for name in
            ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json")}


def condition_settings(seed, mode):
    run_id = f"seed_{seed}_{mode}"
    train_runtime = deepcopy(TRAIN_RUNTIME_TEMPLATE)
    train_hyper = deepcopy(TRAIN_HYPER_TEMPLATE)
    train_hyper["training"]["seed"] = seed
    train_hyper["augmentation"]["probability"] = EXPERIMENT_PARAMS["augmentation_modes"][mode]
    train_runtime.update(
        adapter_path=f"models/{run_id}", output_dir=f"models/{run_id}/checkpoints",
        run_config_file=f"result/{run_id}.run_config.json",
        training_history_file=f"result/{run_id}.training_history.json",
        report_file=f"result/{run_id}.evaluation.json",
    )
    eval_runtime = deepcopy(EVAL_RUNTIME_TEMPLATE)
    eval_runtime.update({key: train_runtime[key] for key in
        ("prefix_path", "model_name", "model_revision", "adapter_path", "test_file",
         "run_config_file", "report_file", "device")})
    return train_runtime, train_hyper, eval_runtime


def run_identity(seed, mode, script):
    return {"run_id": f"seed_{seed}_{mode}", "seed": seed, "augmentation": mode,
            "implementation_sha256": implementation_sha256(script, "experiment_common.py")}


def condition_arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, choices=EXPERIMENT_PARAMS["seeds"], required=True)
    parser.add_argument("--augmentation", choices=EXPERIMENT_PARAMS["augmentation_modes"], required=True)
    return parser.parse_args()


def training_completed(params):
    path = Path(runtime_path(params, "run_config_file"))
    if not path.exists():
        return False
    if read_json(path).get("status") != "training_completed":
        raise RuntimeError(f"미완료 학습은 자동 재개하거나 덮어쓰지 않습니다: {path}")
    return True


def check_training_outputs(params):
    for key in ("run_config_file", "training_history_file", "report_file"):
        path = Path(runtime_path(params, key))
        if path.exists():
            raise FileExistsError(f"기존 산출물이 있습니다: {path}")
    adapter = Path(runtime_path(params, "adapter_path"))
    if adapter.exists() and any(path.name != "README.md" for path in adapter.iterdir()):
        raise FileExistsError(f"기존 모델 또는 체크포인트가 있습니다: {adapter}")

def summarize_cases(cases):
    values = [case["value_min_confidence"] for case in cases if case["value_min_confidence"] is not None]
    averages = [case["value_avg_confidence"] for case in cases if case["value_avg_confidence"] is not None]
    correct = sum(case["is_exact_match"] for case in cases)
    return {
        "total": len(cases),
        "correct": correct,
        "accuracy": 100.0 * correct / len(cases) if cases else None,
        "value_min_confidence": min(values) if values else None,
        "value_avg_confidence": sum(averages) / len(averages) if averages else None,
        "confidence_count": len(values),
        "confidence_missing_count": len(cases) - len(values),
    }


def case_key(case):
    return tuple(case[key] for key in ("site", "image_id", "source_index"))


def compare_group(on, off, keys):
    left = summarize_cases([on[key] for key in keys])
    right = summarize_cases([off[key] for key in keys])
    transitions = Counter((on[key]["is_exact_match"], off[key]["is_exact_match"]) for key in keys)
    return {"on": left, "off": right, "correct_delta": left["correct"] - right["correct"],
            "accuracy_delta_pp": left["accuracy"] - right["accuracy"] if keys else None,
            "both_correct": transitions[True, True], "on_only_correct": transitions[True, False],
            "off_only_correct": transitions[False, True], "both_incorrect": transitions[False, False]}


def build_comparison():
    base = Path(TRAIN_RUNTIME_TEMPLATE["prefix_path"])
    samples = read_jsonl(base / "test.jsonl")
    index = {case_key(sample["metadata"]): sample for sample in samples}
    groups = {"overall": list(index)}
    for name in sorted({x["metadata"]["test_type"] for x in index.values()}):
        groups[name] = [key for key, x in index.items() if x["metadata"]["test_type"] == name]
    groups["without_cluster_22"] = [key for key, x in index.items() if x["metadata"]["cluster_id"] != EXPERIMENT_PARAMS["excluded_cluster"]]
    groups["one_task"] = [key for key, x in index.items() if len(json.loads(x["messages"][2]["content"])) == 1]
    groups["multiple_tasks"] = [key for key, x in index.items() if len(json.loads(x["messages"][2]["content"])) >= 2]
    for cluster in sorted({x["metadata"]["cluster_id"] for x in index.values()}):
        groups[f"cluster_{cluster}"] = [key for key, x in index.items() if x["metadata"]["cluster_id"] == cluster]
    paired, summaries, evaluation_hashes = [], [], {}
    for seed in EXPERIMENT_PARAMS["seeds"]:
        pair = {}
        for mode in EXPERIMENT_PARAMS["augmentation_modes"]:
            run_id = f"seed_{seed}_{mode}"
            path = base / "result" / f"{run_id}.evaluation.json"
            report = read_json(path)
            pair[mode] = {case_key(case): case for case in report["cases"]}
            evaluation_hashes[run_id] = file_sha256(path)
            summaries.append({"run_id": run_id, "seed": seed, "augmentation": mode,
                              "overall": summarize_cases(report["cases"]), "by_test_type": report["by_test_type"]})
        paired.append({"seed": seed, "groups": {name: compare_group(pair["on"], pair["off"], keys) for name, keys in groups.items()}})
    aggregate = {}
    for name in groups:
        values = [row["groups"][name]["accuracy_delta_pp"] for row in paired]
        aggregate[name] = {"seed_count": len(values), "mean_accuracy_delta_pp": statistics.mean(values),
                           "sample_std_accuracy_delta_pp": statistics.stdev(values),
                           "min_accuracy_delta_pp": min(values), "max_accuracy_delta_pp": max(values),
                           "on_wins": sum(v > 0 for v in values), "ties": sum(v == 0 for v in values),
                           "off_wins": sum(v < 0 for v in values)}
    return {"experiment": "09", "seeds": list(EXPERIMENT_PARAMS["seeds"]),
            "implementation_sha256": implementation_sha256("run_experiment.py", "experiment_common.py"),
            "input_file_hashes": input_hashes(), "evaluation_sha256": evaluation_hashes,
            "confidence_definition": "min of known case value_min_confidence including correct and incorrect cases; JSON syntax excluded; missing remains null",
            "interpretation": "동일 스케줄의 증강 효과와 시드별 반복성. 토큰 노출량 대조 없음. 시드 42 제외. 통계적 유의성 자동 판정 없음.",
            "runs": summaries, "paired": paired, "across_seeds": aggregate}

def display_comparison(comparison):
    def table(headers, rows):
        def fmt(value):
            if value is None:
                return "미확인"
            return f"{value:.6f}" if isinstance(value, float) else str(value)
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
        print("\n".join(lines), flush=True)
    print("### 실행별 전체 및 유형별 결과", flush=True)
    rows = []
    for run in comparison["runs"]:
        for group, x in {"overall": run["overall"], **run["by_test_type"]}.items():
            rows.append([run["run_id"], group, x["total"], x["correct"], x["accuracy"], x["value_min_confidence"], x["confidence_missing_count"]])
    table(["실행", "유형", "건수", "정답", "일치율 (%)", "값 토큰 최소 신뢰도", "신뢰도 미확인"], rows)
    print("### 시드별 on − off 및 정답 전환", flush=True)
    rows = [[pair["seed"], name, g["correct_delta"], g["accuracy_delta_pp"], g["on_only_correct"], g["off_only_correct"], g["both_correct"], g["both_incorrect"]]
            for pair in comparison["paired"] for name, g in pair["groups"].items() if not name.startswith("cluster_")]
    table(["시드", "범위", "정답 차이", "일치율 차이 (%p)", "on만 정답", "off만 정답", "모두 정답", "모두 오답"], rows)
    print("### 시드 간 차이 집계", flush=True)
    rows = [[name, x["mean_accuracy_delta_pp"], x["sample_std_accuracy_delta_pp"], x["min_accuracy_delta_pp"], x["max_accuracy_delta_pp"], x["on_wins"], x["ties"], x["off_wins"]]
            for name, x in comparison["across_seeds"].items() if not name.startswith("cluster_")]
    table(["범위", "평균 (%p)", "표본 표준편차", "최소", "최대", "on 우세", "동률", "off 우세"], rows)
    print("### 평가 클러스터별 비교", flush=True)
    rows = [[pair["seed"], name, g["on"]["total"], g["on"]["correct"], g["off"]["correct"], g["accuracy_delta_pp"]]
            for pair in comparison["paired"] for name, g in pair["groups"].items() if name.startswith("cluster_")]
    table(["시드", "클러스터", "평가 건수", "on 정답", "off 정답", "차이 (%p)"], rows)
    print("값 토큰 최소 신뢰도는 정답과 오답을 포함한 케이스별 최솟값의 min입니다. 평균이나 보정된 정답 확률이 아닙니다. 3개 시드의 평균이 양수여도 통계적 유의성을 자동 판정하지 않습니다.", flush=True)
