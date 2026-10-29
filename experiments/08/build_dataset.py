"""08 실험의 공통 정제본과 저장된 분할을 Qwen 메시지로 변환합니다."""

import json
from pathlib import Path


DATASET_FILE = "dataset/processed/dataset.json"
WORK_TYPES_FILE = "experiments/08/data/work_type.json"
CLUSTER_FILE = "experiments/08/data/cluster.json"
SPLIT_META_FILE = "experiments/08/data/split_meta.json"
TRAIN_FILE = "experiments/08/data/train.jsonl"
TEST_FILE = "experiments/08/data/test.jsonl"
SYSTEM_PROMPT = '보드판 텍스트에서 현장별 허용 공종과 규격 규칙에 맞는 표준 작업 내역을 추출하세요. 설명과 마크다운 없이 JSON 배열만 출력하세요. 각 객체의 필드는 workType(표준 공종명 문자열), spec(규격 문자열, 규격 없으면 ""), quantity(수량 숫자, 부동소수점)입니다.'


def to_messages(record, master, source_index, cluster, split_group):
    rules = json.dumps(master[record["site"]], ensure_ascii=False, separators=(",", ":"))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"[현장 공종 및 규격 규칙]\n{rules}\n\n[보드판 텍스트]\n{record.get('x', '')}"},
            {"role": "assistant", "content": json.dumps(record["y"], ensure_ascii=False, indent=2)},
        ],
        "metadata": {
            "site": record["site"], "image_id": record["image_id"], "source_index": source_index,
            "multi_label": len(record["y"]) >= 2, "y_count": len(record["y"]),
            "cluster_id": cluster["cluster_id"], "rep_signature": cluster["rep_signature"],
            "split_group": split_group,
            **({"test_type": split_group} if split_group != "train" else {}),
        },
    }


def write_datasets(train_records, test_records):
    for filename, records in ((TRAIN_FILE, train_records), (TEST_FILE, test_records)):
        path = project_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def project_path(path):
    return Path(__file__).resolve().parents[2] / path


def main():
    records = json.loads(project_path(DATASET_FILE).read_text(encoding="utf-8"))
    master = json.loads(project_path(WORK_TYPES_FILE).read_text(encoding="utf-8"))
    cluster_data = json.loads(project_path(CLUSTER_FILE).read_text(encoding="utf-8"))
    split_meta = json.loads(project_path(SPLIT_META_FILE).read_text(encoding="utf-8"))
    by_index = {
        index: cluster
        for cluster in cluster_data["clusters"]
        for index in cluster["sample_indices"]
    }
    test_types = {
        index: name
        for name, subset in split_meta["test"]["subsets"].items()
        for index in subset["indices"]
    }
    train_records = [
        to_messages(records[index], master, index, by_index[index], "train")
        for index in split_meta["train"]["indices"]
    ]
    test_records = [
        to_messages(records[index], master, index, by_index[index], test_types[index])
        for index in split_meta["test"]["indices"]
    ]
    write_datasets(train_records, test_records)
    print(f"Qwen 데이터 생성 완료: 학습 {len(train_records)}건 / 평가 {len(test_records)}건")
    print("평가 유형:", {name: subset["count"] for name, subset in split_meta["test"]["subsets"].items()})
    print("저장:", TRAIN_FILE, TEST_FILE)


if __name__ == "__main__":
    main()
