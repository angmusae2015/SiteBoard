"""프로젝트 루트에서 실행해 패턴 클러스터를 생성합니다."""

import difflib
import hashlib
import json
from pathlib import Path
import re
from collections import defaultdict
from typing import Any, Dict, List, Set


DATASET_FILE = "dataset/processed/dataset.json"
CLUSTER_FILE = "experiments/09/data/cluster.json"
SIMILARITY_THRESHOLD = 0.85


class PatternClusterer:

    """
    similarity_threshold (float) :
        클러스터링 기준으로 사용할 유사도 임계 값.


    clusters (List[Dict[str, Any]]) :
        클러스터 정보 딕셔너리의 리스트. fit()이 완료되면 클러스터의 소속 레코드 수(count) 내림차순으로 정렬됨.

        클러스터 정보 딕셔너리의 필드 :
            cluster_id (int)           : 클러스터의 ID. 정렬 후 인덱스로 결정됨.
            rep_signature (str)        : 유사도 비교 기준이 되는 대표 정규화 문자열.
            sub_signatures (List[str]) : 대표 문자열을 포함해 이 클러스터에 묶인 정규화 문자열.
            count (int)                : 소속 레코드 수. 클러스터의 정렬 기준으로 사용됨.
            ratio (float)              : 소속 레코드 수의 전체 레코드 대비 비율(%). 결과 기록용으로 사용됨.
            sample_indices (List[int]) : 소속 레코드들의 인덱스 리스트.


    sample_to_cluster (Dict[int, int]) :
        레코드의 인덱스로 소속 클러스터의 ID를 색인할 수 있는 딕셔너리.


    total_samples (int) :
        전체 레코드 수.
    """


    def __init__(self, similarity_threshold: float):
        self.similarity_threshold = similarity_threshold

        self.clusters: List[Dict[str, Any]] = []

        self.sample_to_cluster: Dict[int, int] = {}
        self.total_samples: int = 0

    @staticmethod
    def extract_signature(text: str) -> str:
        if not text:
            return '<EMPTY>'
        t = text.strip()
        t = re.sub('^(?:no\\.?\\s*)?\\d+[\\.\\-\\/:]\\s*', '', t, flags=re.IGNORECASE)
        t = re.sub('^\\d+번\\s*', '', t)
        t = re.sub('\\d{2,4}\\s*[\\*xX×]\\s*\\d{2,4}', '<DIM>', t)
        t = re.sub('[\\*xX]\\s*\\d+\\b', '*<CNT>', t)
        t = re.sub('\\d+', '<NUM>', t)
        t = re.sub('[\\.,;]+$', '', t)
        t = re.sub('\\s+', ' ', t).strip()
        return t if t else '<EMPTY>'

    @staticmethod
    def calculate_similarity(s1: str, s2: str) -> float:
        if s1 == s2:
            return 1.0
        return difflib.SequenceMatcher(None, s1, s2).ratio()

    def fit(self, dataset: List[Dict[str, Any]]) -> 'PatternClusterer':
        """데이터셋의 레코드를 클러스터링하는 메소드."""

        self.total_samples = len(dataset)
        self.clusters = []
        self.sample_to_cluster = {}
        # 정규화 문자열과 소속 레코드들의 인덱스를 매핑하는 딕셔너리.
        sig_to_records: Dict[str, List[int]] = defaultdict(list)

        # 레코드를 추출한 정규화 문자열에 따라 분류함.
        for idx, item in enumerate(dataset):
            x = item.get('x', '')
            sig = self.extract_signature(x)
            sig_to_records[sig].append(idx)

        # 레코드 빈도 내림차순으로 정규화 문자열을 정렬한 리스트.
        # 레코드가 많은 패턴부터 클러스터 대표로 선택하기 위함.
        sorted_sigs: List[str] = sorted(
            sig_to_records.keys(),
            key=lambda s: len(sig_to_records[s]),
            reverse=True
        )

        # 클러스터링된 정규화 문자열 세트. 중복 클러스터링을 방지하기 위해 사용함.
        visited: Set[str] = set()
        # 클러스터 ID 카운터
        cluster_id = 0

        for i, sig in enumerate(sorted_sigs):
            if sig in visited:
                continue
            # 클러스터링되지 않은 패턴일 경우 신규 클러스터를 생성함.

            # 클러스터 소속 레코드
            cluster_indices = list(sig_to_records[sig])
            # 클러스터 소속 정규화 문자열
            cluster_sigs = [sig]
            visited.add(sig)

            # 이후 패턴을 방문하여 유사도를 계산해 임계값을 넘을 경우 흡수함.
            for j in range(i + 1, len(sorted_sigs)):
                other_sig = sorted_sigs[j]

                if other_sig in visited:
                    continue

                if self.calculate_similarity(sig, other_sig) >= self.similarity_threshold:
                    cluster_indices.extend(sig_to_records[other_sig])
                    cluster_sigs.append(other_sig)
                    visited.add(other_sig)

            # 클러스터 소속 레코드의 수
            count = len(cluster_indices)
            # 전체 레코드 대비 클러스터 소속 레코드 수의 비율
            ratio = count / self.total_samples * 100 if self.total_samples > 0 else 0.0
            # 클러스터 정보 딕셔너리
            cluster_info = {
                'cluster_id': cluster_id, 'rep_signature': sig, 'sub_signatures': cluster_sigs,
                'count': count, 'ratio': round(ratio, 2),
                'sample_indices': cluster_indices,
            }
            self.clusters.append(cluster_info)

            # 클러스터 ID 색인 딕셔너리에 기록
            for idx in cluster_indices:
                self.sample_to_cluster[idx] = cluster_id
            cluster_id += 1

        # 클러스터 소속 레코드 수 기준 내림차순으로 클러스터 정보 딕셔너리를 정렬하고,
        # 정렬된 인덱스로 새 ID를 부여함
        self.clusters.sort(key=lambda c: c['count'], reverse=True)
        for new_id, cl in enumerate(self.clusters):
            cl['cluster_id'] = new_id
            for idx in cl['sample_indices']:
                self.sample_to_cluster[idx] = new_id

        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_samples': self.total_samples, 'total_clusters': len(self.clusters),
            'similarity_threshold': self.similarity_threshold,
            'clusters': self.clusters, 'sample_to_cluster': self.sample_to_cluster,
        }


def project_path(path):
    return Path(__file__).resolve().parents[2] / path


def main():
    with open(project_path(DATASET_FILE), "rb") as stream:
        payload = stream.read()

    records = json.loads(payload)
    clusterer = PatternClusterer(SIMILARITY_THRESHOLD).fit(records)

    result = {
        **clusterer.to_dict(),
        "dataset_sha256": hashlib.sha256(payload).hexdigest(),
    }

    project_path(CLUSTER_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(project_path(CLUSTER_FILE), "x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(f"클러스터 생성 완료: {clusterer.total_samples}건 / {len(clusterer.clusters)}개 클러스터")
    print("저장:", CLUSTER_FILE)


if __name__ == "__main__":
    main()
