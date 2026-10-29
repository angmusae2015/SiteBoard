"""저장된 한 조건의 어댑터를 평가하고 종료합니다."""

import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

from experiment_common import (
    EVAL_HYPER_PARAMS as HYPER_PARAMS, PROJECT_ROOT, condition_arguments,
    condition_settings, file_sha256, read_jsonl, run_identity, runtime_path,
    summarize_cases, training_completed, write_json,
)

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def path_for(field):
    return runtime_path(RUNTIME_PARAMS, field)

def resolve_dtype(name):
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return getattr(torch, name)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"중복 JSON 키: {key}")
        result[key] = value
    return result


def parse_prediction(text):
    """필드와 값 타입을 검사하고, 유효하지 않은 출력은 None으로 반환합니다."""
    try:
        data = json.loads(text, object_pairs_hook=unique_object)
        if not isinstance(data, list):
            return None
        for item in data:
            if not isinstance(item, dict) or set(item) != {"workType", "spec", "quantity"}:
                return None
            if not isinstance(item["workType"], str) or not item["workType"].strip():
                return None
            if not isinstance(item["spec"], str):
                return None
            item["workType"].encode("utf-8")
            item["spec"].encode("utf-8")
            if type(item["quantity"]) not in (int, float) or not math.isfinite(item["quantity"]):
                return None
        return data
    except (ValueError, TypeError, OverflowError, RecursionError):
        return None


def parse_and_normalize(text):
    data = parse_prediction(text)
    if data is None:
        return None
    return [(item["workType"].strip(), item["spec"].strip(), round(item["quantity"], 4)) for item in data]


def build_token_spans(tokenizer, token_ids, probabilities, pred_text):
    """Qwen 바이트 토큰을 복원해 한글 분할 토큰에도 같은 값 구간을 연결합니다."""
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    missing = [value for value in range(256) if value not in visible]
    byte_decoder = dict(zip(map(chr, visible + list(range(256, 256 + len(missing)))), visible + missing))
    spans = []
    restored = bytearray()
    try:
        for token, probability in zip(tokenizer.convert_ids_to_tokens(token_ids), probabilities):
            start = len(restored)
            restored.extend(byte_decoder[char] for char in token)
            spans.append((start, len(restored), probability))
    except KeyError:
        return None
    if len(spans) != len(token_ids) or bytes(restored) != pred_text.encode("utf-8"):
        return None
    return spans


def extract_values_with_confidence(pred_text, token_spans):
    """값과 겹치는 생성 토큰을 집계하며 대응이 미확인인 값은 None으로 둡니다."""
    data = parse_prediction(pred_text)
    if data is None:
        return [], None, None
    predictions = [dict(item) for item in data]
    for item in predictions:
        for field in ("workType", "spec", "quantity"):
            item[f"{field}_confidence"] = None
    if token_spans is None:
        return predictions, None, None

    # JSON 문자열 전체를 먼저 매칭해 값 내부의 키처럼 보이는 문자열을 제외합니다.
    string = r'"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"'
    number = r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?'
    fields = list(re.finditer(rf'({string})\s*:\s*({string}|{number})|{string}', pred_text))
    fields = [match for match in fields if match.group(1) is not None]
    if len(fields) != 3 * len(data):
        return predictions, None, None
    byte_offsets = [0]
    for char in pred_text:
        byte_offsets.append(byte_offsets[-1] + len(char.encode("utf-8")))
    value_indices = set()
    for index, match in enumerate(fields):
        field = json.loads(match.group(1))
        start, end = match.span(2)
        if pred_text[start] == '"':
            start, end = start + 1, end - 1
        selected = {i for i, (left, right, _) in enumerate(token_spans)
                    if max(byte_offsets[start], left) < min(byte_offsets[end], right)}
        if selected:
            probabilities = [token_spans[i][2] for i in selected]
            predictions[index // 3][f"{field}_confidence"] = sum(probabilities) / len(probabilities)
            value_indices.update(selected)
    probabilities = [token_spans[i][2] for i in value_indices]
    return (predictions, sum(probabilities) / len(probabilities), min(probabilities)) if probabilities else (predictions, None, None)


def collect_generated_tokens(outputs, prompt_length):
    """선택된 토큰의 확률을 GPU에서 모으고 배치 단위로 CPU에 전달합니다."""
    generated = outputs.sequences[:, prompt_length:]
    # 전체 어휘 점수를 쌓지 않고 각 스텝에서 선택된 토큰의 확률만 보관합니다.
    probabilities = torch.stack([
        torch.softmax(scores.float(), dim=-1)
        .gather(1, generated[:, step:step + 1]).squeeze(1)
        for step, scores in enumerate(outputs.scores)
    ], dim=1)
    return generated.tolist(), probabilities.tolist()


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        path_for("adapter_path"), **HYPER_PARAMS["tokenizer"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model():
    dtype = resolve_dtype(HYPER_PARAMS["precision"]["compute_dtype"])
    embedding_dtype = resolve_dtype(HYPER_PARAMS["precision"]["frozen_embedding_dtype"])
    model = AutoModelForCausalLM.from_pretrained(
        RUNTIME_PARAMS["model_name"], revision=RUNTIME_PARAMS["model_revision"],
        dtype=dtype,
        quantization_config=BitsAndBytesConfig(
            bnb_4bit_compute_dtype=dtype, **HYPER_PARAMS["quantization"],
        ),
        **HYPER_PARAMS["model_loading"],
    )
    for module in model.modules():
        if "norm" in type(module).__name__.lower():
            module.to(dtype=torch.float32)
    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        module.to(dtype=embedding_dtype)
    model = PeftModel.from_pretrained(model, path_for("adapter_path"), is_trainable=False)
    model.config.use_cache = True
    model.eval()
    return model


def prepare_evaluation(tokenizer):
    samples = read_jsonl(path_for("test_file"))
    prompts = [tokenizer.apply_chat_template(
        sample["messages"][:2], tokenize=False, add_generation_prompt=True,
        **HYPER_PARAMS["chat_template"],
    ) for sample in samples]
    paths = [path_for(key) for key in ("test_file", "split_meta_file", "run_config_file")]
    adapter = Path(path_for("adapter_path"))
    paths.extend(str(adapter / name) for name in
                 ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json"))
    return samples, prompts, {path: file_sha256(path) for path in paths}


def collect_execution_metadata(model=None):
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(), "python_executable": sys.executable,
        "platform": platform.platform(), "packages": packages,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda": torch.version.cuda,
        "base_model": RUNTIME_PARAMS["model_name"],
        "base_model_revision": getattr(getattr(model, "config", None), "_commit_hash", None)
            or RUNTIME_PARAMS["model_revision"],
        "compute_dtype": str(resolve_dtype(HYPER_PARAMS["precision"]["compute_dtype"])),
        "input_embedding_dtype": str(model.get_input_embeddings().weight.dtype) if model else None,
        "output_embedding_dtype": str(model.get_output_embeddings().weight.dtype) if model else None,
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
    }


def evaluate():
    report_path = Path(path_for("report_file"))
    if report_path.exists():
        raise FileExistsError(f"기존 평가 결과가 있습니다: {report_path}")
    settings = HYPER_PARAMS["evaluation"]
    tokenizer = load_tokenizer()
    samples, prompts, file_hashes = prepare_evaluation(tokenizer)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    print("평가 기본 모델과 어댑터 로딩", flush=True)
    model = load_model()
    eos_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    eos_ids.discard(None)
    case_results = []
    print(f"평가 {len(samples)}건 / GPU: {torch.cuda.get_device_name(0)}")

    for start in tqdm(range(0, len(samples), settings["batch_size"]), desc="Evaluation"):
        batch = samples[start:start + settings["batch_size"]]
        inputs = tokenizer(prompts[start:start + len(batch)], return_tensors="pt", padding=True,
                           truncation=False, add_special_tokens=False).to(model.device)
        with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=resolve_dtype(HYPER_PARAMS["precision"]["compute_dtype"])):
            outputs = model.generate(
                **inputs,
                max_new_tokens=settings["max_new_tokens"],
                do_sample=settings["do_sample"],
                temperature=None, top_p=None, top_k=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=sorted(eos_ids),
                return_dict_in_generate=True,
                output_scores=True,
            )
            batch_token_ids, batch_probabilities = collect_generated_tokens(
                outputs, inputs["input_ids"].shape[1],
            )
        # CPU에서 채점하기 전에 GPU의 전체 어휘 점수와 입력을 해제합니다.
        del outputs, inputs
        for offset, sample in enumerate(batch):
            token_ids = batch_token_ids[offset]
            end = next((i for i, token_id in enumerate(token_ids)
                        if token_id in eos_ids or token_id == tokenizer.pad_token_id), len(token_ids))
            terminated = end < len(token_ids) and token_ids[end] in eos_ids
            token_ids = token_ids[:end]
            pred_text = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            probabilities = batch_probabilities[offset][:end]
            token_spans = build_token_spans(tokenizer, token_ids, probabilities, pred_text)
            prediction, avg_confidence, min_confidence = extract_values_with_confidence(pred_text, token_spans)
            ground_truth = sample["messages"][2]["content"]
            norm_pred = parse_and_normalize(pred_text)
            is_match = norm_pred is not None and Counter(norm_pred) == Counter(parse_and_normalize(ground_truth))
            metadata = sample["metadata"]
            case_results.append({
                "case_id": start + offset + 1,
                **{key: metadata[key] for key in ("site", "image_id", "source_index", "test_type")},
                "input_user": sample["messages"][1]["content"],
                "ground_truth_raw": ground_truth,
                "prediction_raw": pred_text,
                "prediction": prediction,
                "is_valid_json_schema": norm_pred is not None,
                "is_exact_match": is_match,
                "output_limit_reached": not terminated and end >= settings["max_new_tokens"],
                "value_avg_confidence": avg_confidence,
                "value_min_confidence": min_confidence,
            })

    overall = summarize_cases(case_results)
    by_type = {name: summarize_cases([case for case in case_results if case["test_type"] == name])
               for name in sorted({case["test_type"] for case in case_results})}
    report = {
        "experiment": deepcopy(ACTIVE_RUN_IDENTITY),
        "process": {"pid": os.getpid(), "script": "experiments/09/evaluate_desktop.py"},
        "model": RUNTIME_PARAMS["model_name"],
        "adapter": path_for("adapter_path"),
        "device": torch.cuda.get_device_name(0),
        "dtype": str(model.dtype),
        "quantized": True, "adapter_merged": False,
        "duration_seconds": time.perf_counter() - started,
        "test_file": path_for("test_file"),
        "runtime_params": deepcopy(RUNTIME_PARAMS),
        "hyper_params": deepcopy(HYPER_PARAMS),
        "batch_size": settings["batch_size"],
        "scoring_version": "strict_schema_multiset_v1",
        "confidence_definition": "Generated tokens overlapping workType/spec/quantity value bytes; unique tokens per case. Group min includes correct and incorrect cases with known confidence; null cases are excluded and counted. Avg is the mean of known case averages.",
        "file_hashes": file_hashes,
        "execution": collect_execution_metadata(model),
        **overall,
        "correct_value_avg_confidence": summarize_cases([case for case in case_results if case["is_exact_match"]])["value_avg_confidence"],
        "incorrect_value_avg_confidence": summarize_cases([case for case in case_results if not case["is_exact_match"]])["value_avg_confidence"],
        "by_test_type": by_type,
        "cases": case_results,
    }
    print("평가 보고서 저장", flush=True)
    write_json(report_path, report)
    print(f"완전 일치율: {overall['accuracy']:.2f}% ({overall['correct']}/{overall['total']})")
    print(f"평가 결과 저장: {report_path}")
    return report



def main():
    global RUNTIME_PARAMS, ACTIVE_RUN_IDENTITY
    args = condition_arguments("09 단일 조건 생성 평가")
    os.chdir(PROJECT_ROOT)
    train_runtime, _, RUNTIME_PARAMS = condition_settings(args.seed, args.augmentation)
    ACTIVE_RUN_IDENTITY = run_identity(args.seed, args.augmentation, "evaluate_desktop.py")
    if not training_completed(train_runtime):
        raise RuntimeError(f"완료된 학습이 없습니다: {ACTIVE_RUN_IDENTITY['run_id']}")
    if Path(path_for("report_file")).exists():
        print(f"[{ACTIVE_RUN_IDENTITY['run_id']}] 완료된 평가를 건너뜁니다.", flush=True)
        return
    evaluate()


if __name__ == "__main__":
    main()
