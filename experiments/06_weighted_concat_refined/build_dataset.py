"""06 실험 디렉터리에서 실행해 저장된 분할을 Qwen 메시지로 변환합니다."""

import json
from pathlib import Path


DATASET_FILE = "../../dataset/processed/dataset.json"
WORK_TYPES_FILE = "data/work_type.json"
CLUSTER_FILE = "data/cluster.json"
SPLIT_META_FILE = "data/split_meta.json"
TRAIN_FILE = "data/train.jsonl"
TEST_FILE = "data/test.jsonl"
SYSTEM_PROMPT = '당신은 공사 현장 사진대지 보드판 텍스트를 분석하여 표준 작업 내역을 JSON 형식으로 추출하는 AI 전문가입니다.\n\n[출력 스키마 및 타입 제약]\n응답은 다음 필드로 구성된 객체들의 JSON 배열(Array<Object>)이어야 합니다:\n- workType (string): 주어진 규칙에 정의된 표준 공종명\n- spec (string): 시공 규격 (규격이 없는 공종은 빈 문자열 "")\n- quantity (number): 시공 수량 (부동소수점)\n\n[준수 규칙]\n1. 반드시 주어진 [허용 표준 공종 및 규격 규칙]에 정의된 공종명과 규격 제약을 엄격히 준수하십시오.\n2. 부연 설명이나 마크다운 텍스트 없이 유효한 JSON 배열만 출력하십시오.'


def to_messages(record, master, source_index, cluster, split_group):
    rules = json.dumps(master[record["site"]], ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"[허용 표준 공종 및 규격 규칙]\n{rules}\n\n[보드판 텍스트]\n{record.get('x', '')}"},
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
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    records = json.loads(Path(DATASET_FILE).read_text(encoding="utf-8"))
    master = json.loads(Path(WORK_TYPES_FILE).read_text(encoding="utf-8"))
    cluster_data = json.loads(Path(CLUSTER_FILE).read_text(encoding="utf-8"))
    split_meta = json.loads(Path(SPLIT_META_FILE).read_text(encoding="utf-8"))
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
