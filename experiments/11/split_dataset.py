import json
import random
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"
SEEDS = range(42, 47)


def record_keys(records):
    keys = {(record["site"], record["image_id"]) for record in records}
    if len(keys) != len(records):
        raise ValueError("중복된 (site, image_id)가 있습니다.")
    return keys


def main():
    with (DATA_DIR / "dataset.json").open(encoding="utf-8") as source:
        records = json.load(source)
    source_keys = record_keys(records)
    train_size = len(records) * 4 // 5
    if train_size == 0:
        raise ValueError("train과 test로 분할하려면 레코드가 2건 이상 필요합니다.")

    output_paths = {
        (seed, split): DATA_DIR / f"{seed}_{split}.json"
        for seed in SEEDS
        for split in ("train", "test")
    }
    for path in output_paths.values():
        if path.exists():
            raise FileExistsError(f"분할 파일이 이미 있습니다: {path}")

    for seed in SEEDS:
        shuffled = records.copy()
        random.Random(seed).shuffle(shuffled)
        train, test = shuffled[:train_size], shuffled[train_size:]
        train_keys, test_keys = record_keys(train), record_keys(test)
        if train_keys & test_keys or train_keys | test_keys != source_keys:
            raise ValueError(f"시드 {seed}의 분할에 중복 또는 누락이 있습니다.")

        for split, subset in (("train", train), ("test", test)):
            with output_paths[seed, split].open("x", encoding="utf-8") as output:
                json.dump(subset, output, ensure_ascii=False, indent=2)
                output.write("\n")
        print(f"{seed}: train {len(train)}, test {len(test)}")


if __name__ == "__main__":
    main()
