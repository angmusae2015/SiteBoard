"""한 시드와 증강 조건을 Unsloth로 학습하고 종료합니다."""

import json
import os
import platform
import random
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

from experiment_common import (
    PROJECT_ROOT, artifact_hashes, check_training_outputs, condition_arguments,
    condition_settings, file_sha256, input_hashes, read_jsonl, run_identity,
    runtime_path, training_completed, write_json,
)

os.environ.setdefault(
    "UNSLOTH_COMPILE_LOCATION",
    str(Path(tempfile.gettempdir()) / "siteboard_unsloth_compiled_cache"),
)
from unsloth import FastLanguageModel
import torch
from transformers import (
    AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq,
    Trainer, TrainingArguments, set_seed,
)


def path_for(field):
    return runtime_path(RUNTIME_PARAMS, field)

def resolve_dtype(name):
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return getattr(torch, name)


def prepare_training_data():
    train_records = read_jsonl(path_for("train_file"))
    valid_records = read_jsonl(path_for("test_file"))
    return train_records, valid_records, {
        "train_count": len(train_records), "test_count": len(valid_records),
    }


class QwenChatDataset:
    """같은 현장의 레코드를 랜덤 결합하고 JSON 응답 구간만 학습합니다."""

    def __init__(self, records, tokenizer, max_seq_length, is_train,
                 aug_prob, aug_max_concat, delimiters):
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.is_train = is_train
        self.aug_prob = aug_prob
        self.aug_max_concat = aug_max_concat
        self.delimiters = delimiters
        self.site_to_indices = {}
        self.components = []
        for index, record in enumerate(records):
            self.site_to_indices.setdefault(record["metadata"]["site"], []).append(index)
            messages = record["messages"]
            rules, text = messages[1]["content"].split("[보드판 텍스트]\n", 1)
            self.components.append((messages[0]["content"], rules, text,
                                    json.loads(messages[2]["content"])))

    def __len__(self):
        return len(self.records)

    def encode_messages(self, messages):
        # 비추론 프롬프트와 JSON 응답을 분리해 경계 토큰까지 정확히 마스킹합니다.
        prompt = self.tokenizer.apply_chat_template(
            messages[:2], tokenize=False, add_generation_prompt=True,
            **HYPER_PARAMS["chat_template"],
        )
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(messages[2]["content"], add_special_tokens=False)
        response_ids.append(self.tokenizer.eos_token_id)
        input_ids = prompt_ids + response_ids
        if len(input_ids) > self.max_seq_length:
            raise ValueError(f"학습 입력과 정답 {len(input_ids)}토큰이 상한 {self.max_seq_length}을 넘습니다. 잘라서 학습하지 않습니다.")
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids),
                "labels": [-100] * len(prompt_ids) + response_ids}

    def __getitem__(self, index):
        messages = self.records[index]["messages"]
        if self.is_train and random.random() < self.aug_prob:
            system, rules, text, answers = self.components[index]
            texts, combined = [text], list(answers)
            pool = self.site_to_indices[self.records[index]["metadata"]["site"]]
            for _ in range(random.randint(2, self.aug_max_concat) - 1):
                # 06과 같이 자기 자신도 포함한 동일 현장 풀에서 복원 추출합니다.
                partner = random.choice(pool)
                _, _, partner_text, partner_answers = self.components[partner]
                texts.append(partner_text)
                combined.extend(partner_answers)
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": rules + "[보드판 텍스트]\n" + random.choice(self.delimiters).join(texts)},
                {"role": "assistant", "content": json.dumps(combined, ensure_ascii=False, indent=2)},
            ]
        return self.encode_messages(messages)


def build_dataset(records, tokenizer, is_train):
    settings = HYPER_PARAMS["augmentation"]
    return QwenChatDataset(
        records, tokenizer, HYPER_PARAMS["training"]["max_seq_length"], is_train,
        settings["probability"], settings["max_concat"], settings["delimiters"],
    )


def load_tokenizer():
    # 기존 마스킹과 평가에 사용하는 공식 템플릿을 그대로 읽습니다.
    tokenizer = AutoTokenizer.from_pretrained(
        RUNTIME_PARAMS["model_name"], revision=RUNTIME_PARAMS["model_revision"],
        **HYPER_PARAMS["tokenizer"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_qlora_model():
    dtype = resolve_dtype(HYPER_PARAMS["precision"]["compute_dtype"])
    # 반환된 토크나이저 대신 위에서 읽은 공식 토크나이저로 학습하고 저장합니다.
    model, _ = FastLanguageModel.from_pretrained(
        model_name=RUNTIME_PARAMS["model_name"], revision=RUNTIME_PARAMS["model_revision"],
        max_seq_length=HYPER_PARAMS["training"]["max_seq_length"], dtype=dtype,
        load_in_4bit=HYPER_PARAMS["quantization"]["load_in_4bit"],
        quantization_config=BitsAndBytesConfig(
            bnb_4bit_compute_dtype=dtype, **HYPER_PARAMS["quantization"],
        ),
        use_gradient_checkpointing=HYPER_PARAMS["kbit_training"]["use_gradient_checkpointing"],
        random_state=HYPER_PARAMS["training"]["seed"],
        **HYPER_PARAMS["model_loading"],
    )
    actual_revision = getattr(model.config, "_commit_hash", None)
    if actual_revision is not None and actual_revision != RUNTIME_PARAMS["model_revision"]:
        raise ValueError(f"요청한 기본 모델 revision과 다릅니다: {actual_revision}")
    model.config.use_cache = False
    return model


def add_lora_adapter(model):
    settings = dict(HYPER_PARAMS["lora"])
    if settings.pop("task_type") != "CAUSAL_LM":
        raise ValueError("이 학습 스크립트는 CAUSAL_LM LoRA 학습만 지원합니다.")
    # Unsloth API가 k비트 준비와 LoRA 패치를 함께 수행합니다.
    model = FastLanguageModel.get_peft_model(
        model,
        random_state=HYPER_PARAMS["training"]["seed"],
        **settings, **HYPER_PARAMS["kbit_training"],
    )
    dtype = resolve_dtype(HYPER_PARAMS["precision"]["frozen_embedding_dtype"])
    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        if module is not None and not module.weight.requires_grad:
            module.to(dtype=dtype)
    for config in model.peft_config.values():
        config.base_model_name_or_path = RUNTIME_PARAMS["model_name"]
        config.revision = RUNTIME_PARAMS["model_revision"]
    model.config.use_cache = False
    model.print_trainable_parameters()
    return model


def build_trainer(model, tokenizer, train_records, valid_records):
    settings = HYPER_PARAMS["training"]
    train_dataset = build_dataset(train_records, tokenizer, is_train=True)
    valid_dataset = build_dataset(valid_records, tokenizer, is_train=False)
    dtype = resolve_dtype(HYPER_PARAMS["precision"]["compute_dtype"])
    args = TrainingArguments(
        disable_tqdm=False,
        output_dir=path_for("output_dir"), num_train_epochs=settings["epochs"],
        per_device_train_batch_size=settings["batch_size"],
        gradient_accumulation_steps=settings["grad_accum"],
        learning_rate=settings["learning_rate"], seed=settings["seed"],
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        **HYPER_PARAMS["trainer"],
    )
    # Dataset을 직접 전달해 매 접근 시점의 랜덤 결합과 labels를 유지합니다.
    trainer = Trainer(
        model=model, args=args, processing_class=tokenizer,
        train_dataset=train_dataset, eval_dataset=valid_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, **HYPER_PARAMS["collator"]),
    )
    return trainer


def collect_execution_metadata(model=None):
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes",
                 "unsloth", "unsloth_zoo", "trl", "triton", "triton-windows",
                 "torchvision", "xformers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(), "python_executable": sys.executable,
        "platform": platform.platform(), "packages": packages,
        "training_backend": "unsloth", "trainer": "transformers.Trainer",
        "gradient_checkpointing": HYPER_PARAMS["kbit_training"]["use_gradient_checkpointing"],
        "unsloth_compile_location": os.environ.get("UNSLOTH_COMPILE_LOCATION"),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda": torch.version.cuda,
        "base_model": RUNTIME_PARAMS["model_name"],
        "base_model_revision": getattr(getattr(model, "config", None), "_commit_hash", None)
            or RUNTIME_PARAMS["model_revision"],
        "compute_dtype": str(resolve_dtype(HYPER_PARAMS["precision"]["compute_dtype"])),
        "input_embedding_dtype": str(model.get_input_embeddings().weight.dtype) if model is not None else None,
        "output_embedding_dtype": str(model.get_output_embeddings().weight.dtype) if model is not None else None,
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
    }



def run_training():
    check_training_outputs(RUNTIME_PARAMS)
    set_seed(HYPER_PARAMS["training"]["seed"])
    print("학습 데이터 읽기", flush=True)
    train_records, valid_records, prepared_data = prepare_training_data()
    started = time.perf_counter()
    config = {
        "experiment": deepcopy(ACTIVE_RUN_IDENTITY),
        "runtime_params": deepcopy(RUNTIME_PARAMS), "hyper_params": deepcopy(HYPER_PARAMS),
        "prepared_data": prepared_data, "input_file_hashes": input_hashes(),
        "prepared_file_hashes": {path_for(key): file_sha256(path_for(key)) for key in ("train_file", "test_file")},
        "process": {"pid": os.getpid(), "script": "experiments/09/train_desktop_unsloth.py"},
        "status": "training_started",
    }
    write_json(path_for("run_config_file"), config)
    trainer = None
    try:
        print("학습 토크나이저와 기본 모델 로딩", flush=True)
        tokenizer = load_tokenizer()
        torch.cuda.reset_peak_memory_stats()
        model = add_lora_adapter(load_qlora_model())
        trainer = build_trainer(model, tokenizer, train_records, valid_records)
        print(f"학습 시작: {prepared_data['train_count']}건, {HYPER_PARAMS['training']['epochs']}에포크", flush=True)
        trainer.train()
        print("어댑터와 토크나이저 저장", flush=True)
        destination = path_for("adapter_path")
        model.save_pretrained(destination)
        tokenizer.save_pretrained(destination)
        write_json(path_for("training_history_file"), trainer.state.log_history)
        config.update(artifact_hashes=artifact_hashes(destination),
                      training_history_sha256=file_sha256(path_for("training_history_file")),
                      global_step=trainer.state.global_step, epoch=trainer.state.epoch,
                      execution=collect_execution_metadata(model), status="training_completed",
                      duration_seconds=time.perf_counter() - started)
        write_json(path_for("run_config_file"), config, mode="w")
        print(f"학습 완료: {destination}", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        config.update(status="training_interrupted" if isinstance(error, KeyboardInterrupt) else "training_failed",
                      error={"type": type(error).__name__, "message": str(error)},
                      duration_seconds=time.perf_counter() - started)
        write_json(path_for("run_config_file"), config, mode="w")
        if trainer is not None and not Path(path_for("training_history_file")).exists():
            write_json(path_for("training_history_file"), trainer.state.log_history)
        raise


def main():
    global RUNTIME_PARAMS, HYPER_PARAMS, ACTIVE_RUN_IDENTITY
    args = condition_arguments("09 단일 조건 Unsloth 학습")
    os.chdir(PROJECT_ROOT)
    RUNTIME_PARAMS, HYPER_PARAMS, _ = condition_settings(args.seed, args.augmentation)
    ACTIVE_RUN_IDENTITY = run_identity(args.seed, args.augmentation, "train_desktop_unsloth.py")
    if training_completed(RUNTIME_PARAMS):
        print(f"[{ACTIVE_RUN_IDENTITY['run_id']}] 완료된 학습을 건너뜁니다.", flush=True)
        return
    run_training()


if __name__ == "__main__":
    main()
