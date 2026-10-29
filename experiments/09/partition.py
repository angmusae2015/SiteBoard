"""프로젝트 루트에서 실행해 클러스터 기반 분할 메타데이터를 생성합니다."""

import hashlib
import json
from pathlib import Path
import random


DATASET_FILE = "dataset/processed/dataset.json"
CLUSTER_FILE = "experiments/09/data/cluster.json"
SPLIT_META_FILE = "experiments/09/data/split_meta.json"
SEED = 42
TEST_RATIO = 0.18
HOLDOUT_CLUSTER_IDS = [20, 22, 35, 36, 42, 65]      # 이전 실험과의 비교를 위해 해당 홀드 아웃으로 고정함


def partition_clusters(clusters, holdout_ids, seed, test_ratio):
    rng = random.Random(seed)
    result = {"train": [], "seen": [], "unseen_complex": [], "unseen_rare": []}

    for cluster in clusters:
        indices = list(cluster["sample_indices"])
        rng.shuffle(indices)

        if cluster["cluster_id"] in holdout_ids:
            result["unseen_complex"].extend(indices)
        elif len(indices) == 1:
            result["unseen_rare"].extend(indices)
        else:
            count = min(len(indices) - 1, max(1, round(len(indices) * test_ratio)))
            result["seen"].extend(indices[:count])
            result["train"].extend(indices[count:])

    return {key: sorted(value) for key, value in result.items()}


def build_split_meta(records, clusters, groups, seed, holdout_ids):
    total_samples = len(records)
    sample_to_cluster = {
        index: cluster["cluster_id"]
        for cluster in clusters
        for index in cluster["sample_indices"]
    }
    train_indices = groups["train"]
    test_indices = sorted(groups["seen"] + groups["unseen_complex"] + groups["unseen_rare"])

    def percentage(count, total):
        return round(count / total * 100, 2) if total else 0.0

    def cluster_count(indices):
        return len({sample_to_cluster[index] for index in indices})

    train_multilabel = sum(len(records[index]["y"]) >= 2 for index in train_indices)
    test_multilabel = sum(len(records[index]["y"]) >= 2 for index in test_indices)
    return {
        "total_samples": total_samples,   # 전체 레코드 수
        "total_clusters": len(clusters),  # 전체 클러스터 수
        "seed": seed,                     # 분할에 사용한 난수 시드
        "train": {                        # 학습셋 통계와 원본 인덱스
            "count": len(train_indices),                                           # 학습 레코드 수
            "ratio": percentage(len(train_indices), total_samples),                # 전체 중 학습셋 비율(%)
            "cluster_count": cluster_count(train_indices),                         # 학습셋에 포함된 클러스터 수
            "multilabel_count": train_multilabel,                                  # 정답 작업이 2개 이상인 학습 레코드 수
            "multilabel_ratio": percentage(train_multilabel, len(train_indices)),  # 학습셋 내 다중 작업 레코드 비율(%)
            "indices": train_indices,                                              # 학습 레코드의 원본 배열 인덱스
        },
        "test": {                         # 평가셋 통계와 유형별 분할
            "total_count": len(test_indices),                                      # 전체 평가 레코드 수
            "total_ratio": percentage(len(test_indices), total_samples),           # 전체 중 평가셋 비율(%)
            "multilabel_count": test_multilabel,                                   # 정답 작업이 2개 이상인 평가 레코드 수
            "multilabel_ratio": percentage(test_multilabel, len(test_indices)),    # 평가셋 내 다중 작업 레코드 비율(%)
            "subsets": {                                                           # 평가 유형별 통계와 원본 인덱스
                "seen_patterns": {           # 학습셋과 클러스터를 공유하는 평가 유형
                    "count": len(groups["seen"]),                              # 해당 유형의 평가 레코드 수
                    "cluster_count": cluster_count(groups["seen"]),            # 해당 유형에 포함된 클러스터 수
                    "indices": groups["seen"],                                 # 해당 유형 레코드의 원본 배열 인덱스
                },
                "unseen_complex_holdout": {  # 지정한 클러스터 전체를 평가에 배정한 유형
                    "count": len(groups["unseen_complex"]),                    # 홀드아웃 평가 레코드 수
                    "cluster_count": cluster_count(groups["unseen_complex"]),  # 홀드아웃 클러스터 수
                    "cluster_ids": sorted(holdout_ids),                        # 홀드아웃으로 지정한 클러스터 ID
                    "indices": groups["unseen_complex"],                       # 홀드아웃 레코드의 원본 배열 인덱스
                },
                "unseen_rare": {             # 홀드아웃이 아닌 단일 레코드 클러스터 유형
                    "count": len(groups["unseen_rare"]),                       # 희소 패턴 평가 레코드 수
                    "cluster_count": cluster_count(groups["unseen_rare"]),     # 희소 패턴 클러스터 수
                    "indices": groups["unseen_rare"],                          # 희소 패턴 레코드의 원본 배열 인덱스
                },
            },
            "indices": test_indices,      # 모든 평가 유형을 합친 원본 배열 인덱스
        },
    }


def project_path(path):
    return Path(__file__).resolve().parents[2] / path


def main():
    with open(project_path(DATASET_FILE), "rb") as stream:
        payload = stream.read()
    records = json.loads(payload)
    with open(project_path(CLUSTER_FILE), encoding="utf-8") as stream:
        cluster_data = json.load(stream)

    clusters = cluster_data["clusters"]
    groups = partition_clusters(clusters, HOLDOUT_CLUSTER_IDS, SEED, TEST_RATIO)
    result = build_split_meta(records, clusters, groups, SEED, HOLDOUT_CLUSTER_IDS)
    result["dataset_sha256"] = hashlib.sha256(payload).hexdigest()
    result["test_ratio"] = TEST_RATIO
    result["holdout_cluster_ids"] = list(HOLDOUT_CLUSTER_IDS)

    project_path(SPLIT_META_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(project_path(SPLIT_META_FILE), "x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(f"분할 완료: 학습 {result['train']['count']}건 / 평가 {result['test']['total_count']}건")
    print("평가 유형:", {key: len(value) for key, value in groups.items() if key != "train"})
    print("저장:", SPLIT_META_FILE)


if __name__ == "__main__":
    main()
