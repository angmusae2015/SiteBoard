"""09번 패턴 클러스터를 탐색하는 로컬 웹 서버."""

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import math
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlparse
from unicodedata import normalize


ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("siteboard_cluster09", ROOT / "experiments/09/cluster.py")
cluster_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cluster_module)
similarity = lru_cache(maxsize=None)(cluster_module.PatternClusterer.calculate_similarity)


class CachedClusterer(cluster_module.PatternClusterer):
    calculate_similarity = staticmethod(similarity)


def letters(text):
    return "".join(char for char in normalize("NFC", text.lower()) if char.isalpha())


def ngrams(text, n):
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


class Explorer:
    def __init__(self):
        payload = (ROOT / "dataset/processed/dataset.json").read_bytes()
        self.records = json.loads(payload)
        self.search_texts = tuple(letters(record.get("x") or "") for record in self.records)
        self.baseline = self.read_json("experiments/09/data/cluster.json")
        self.split = self.read_json("experiments/09/data/split_meta.json")
        master = self.read_json("experiments/09/data/work_type.json")
        digest = hashlib.sha256(payload).hexdigest()
        if any(item.get("dataset_sha256") != digest for item in (self.baseline, self.split)):
            raise ValueError("정제본과 09번 기록의 원천 해시가 일치하지 않습니다.")
        identities = [(r["site"], r["image_id"]) for r in self.records]
        if len(set(identities)) != len(identities):
            raise ValueError("현장과 사진 ID가 중복됩니다.")
        for record in self.records:
            rules = master[record["site"]]
            for work in record["y"]:
                allowed = rules[work["workType"]]
                value = work["spec"]
                if allowed == [] and value != "":
                    raise ValueError("규격 없음 규칙에 맞지 않습니다.")
                if allowed and value not in {str(v) for v in allowed}:
                    raise ValueError("공종 마스터에 없는 규격입니다.")
        self.groups = {}
        for index in self.split["train"]["indices"]:
            self.groups[index] = "train"
        for name, subset in self.split["test"]["subsets"].items():
            for index in subset["indices"]:
                if index in self.groups:
                    raise ValueError("학습 또는 평가 인덱스가 중복됩니다.")
                self.groups[index] = name
        if set(self.groups) != set(range(len(self.records))):
            raise ValueError("분할에 누락되거나 잘못된 인덱스가 있습니다.")
        for filename, expected in (("train", "train"), ("test", None)):
            found = set()
            path = ROOT / f"experiments/09/data/{filename}.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                meta = row["metadata"]
                index = meta["source_index"]
                record = self.records[index]
                if index in found or (meta["site"], meta["image_id"]) != identities[index]:
                    raise ValueError("메시지 식별자 또는 인덱스가 일치하지 않습니다.")
                if meta["split_group"] != self.groups[index] or (expected and self.groups[index] != expected):
                    raise ValueError("메시지의 분할 유형이 일치하지 않습니다.")
                messages = row["messages"]
                user = next(m["content"] for m in messages if m["role"] == "user")
                answer = json.loads(next(m["content"] for m in messages if m["role"] == "assistant"))
                if user.split("[보드판 텍스트]\n", 1)[1] != record.get("x", "") or answer != record["y"]:
                    raise ValueError("원문 또는 정답이 공통 정제본과 일치하지 않습니다.")
                found.add(index)
            expected_indices = {i for i, group in self.groups.items() if (group == "train") == (filename == "train")}
            if found != expected_indices:
                raise ValueError("메시지 분할에 누락이 있습니다.")
        self.signatures = [CachedClusterer.extract_signature(r.get("x", "")) for r in self.records]
        self.frequencies = Counter(self.signatures)
        self.lock = threading.Lock()
        self.cache = {}
        result = CachedClusterer(0.85).fit(self.records).to_dict()
        if result["clusters"] != self.baseline["clusters"]:
            raise ValueError("0.85 재계산 결과가 보존 기록과 일치하지 않습니다.")
        self.get_result(0.85)

    @staticmethod
    def read_json(path):
        return json.loads((ROOT / path).read_text(encoding="utf-8"))

    def get_result(self, threshold):
        with self.lock:
            if threshold in self.cache:
                return self.cache[threshold]
            clusters = CachedClusterer(threshold).fit(self.records).clusters
            for cluster in clusters:
                rep = cluster["rep_signature"]
                cluster["patterns"] = [
                    {"text": sig, "count": self.frequencies[sig], "similarity": similarity(rep, sig)}
                    for sig in cluster["sub_signatures"]
                ]
                cluster["min_similarity"] = min(p["similarity"] for p in cluster["patterns"])
                cluster["baseline_overlap"] = dict(Counter(
                    self.baseline["sample_to_cluster"][str(i)] for i in cluster["sample_indices"]
                ))
                cluster["split_counts"] = dict(Counter(self.groups[i] for i in cluster["sample_indices"]))
            result = {"threshold": threshold, "clusters": clusters, "total": len(self.records)}
            if len(self.cache) >= 64:
                self.cache.pop(next(key for key in self.cache if key != 0.85))
            self.cache[threshold] = result
            return result

    def warm_cache(self):
        self.get_result(1.0)
        print("유사도 캐시 준비 완료", flush=True)

    def bootstrap(self):
        return {
            "records": [dict(r, source_index=i, signature=self.signatures[i], split=self.groups[i])
                        for i, r in enumerate(self.records)],
            "baseline": self.get_result(0.85),
            "sites": sorted({r["site"] for r in self.records}),
            "work_types": sorted({w["workType"] for r in self.records for w in r["y"]}),
            "unique_patterns": len(self.frequencies),
        }

    @lru_cache(maxsize=5)
    def ngram_index(self, n):
        vectors = [ngrams(text, n) for text in self.search_texts]
        frequencies = Counter(gram for vector in vectors for gram in vector)
        idf = {gram: math.log((1 + len(vectors)) / (1 + count)) + 1
               for gram, count in frequencies.items()}
        return vectors, idf

    def search_ngrams(self, text, n):
        if not 1 <= n <= 5:
            raise ValueError("N은 1~5 사이여야 합니다.")
        query = ngrams(letters(text), n)
        if not query:
            raise ValueError(f"숫자와 기호를 제외하고 문자 {n}자 이상을 입력하세요.")

        vectors, idf = self.ngram_index(n)
        unseen_idf = math.log(1 + len(vectors)) + 1
        weighted_query = {gram: count * idf.get(gram, unseen_idf) for gram, count in query.items()}
        query_norm = math.hypot(*query.values())
        weighted_query_norm = math.hypot(*weighted_query.values())
        results = {"jaccard": [], "cosine": [], "tfidf": []}
        for index, vector in enumerate(vectors):
            common = query.keys() & vector.keys()
            if not common:
                continue
            weighted_vector = {gram: count * idf[gram] for gram, count in vector.items()}
            scores = {
                "jaccard": len(common) / (len(query) + len(vector) - len(common)),
                "cosine": sum(query[gram] * vector[gram] for gram in common)
                          / (query_norm * math.hypot(*vector.values())),
                "tfidf": math.fsum(weighted_query[gram] * weighted_vector[gram] for gram in common)
                         / (weighted_query_norm * math.hypot(*weighted_vector.values())),
            }
            record = self.records[index]
            for method, score in scores.items():
                results[method].append({
                    "score": min(1.0, score),
                    "x": record.get("x") or "",
                    "site": record["site"],
                    "image_id": record["image_id"],
                    "y": record["y"],
                })
        for matches in results.values():
            matches.sort(key=lambda match: match["score"], reverse=True)
        return results


def make_handler(explorer):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/bootstrap":
                    self.send_json(explorer.bootstrap())
                elif parsed.path == "/api/clusters":
                    raw = parse_qs(parsed.query).get("threshold", ["0.85"])[0]
                    threshold = float(raw)
                    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
                        raise ValueError("유사도는 0 이상 1 이하로 입력하세요.")
                    self.send_json(explorer.get_result(round(threshold, 3)))
                elif parsed.path == "/api/ngram":
                    params = parse_qs(parsed.query)
                    self.send_json(explorer.search_ngrams(
                        params.get("q", [""])[0], int(params.get("n", ["2"])[0])
                    ))
                else:
                    files = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                             "/ngram": ("ngram.html", "text/html"), "/ngram.js": ("ngram.js", "text/javascript"),
                             "/style.css": ("style.css", "text/css")}
                    if parsed.path not in files:
                        self.send_error(404)
                        return
                    filename, content_type = files[parsed.path]
                    self.send_payload((ASSETS / filename).read_bytes(), content_type)
            except (ValueError, KeyError) as error:
                self.send_json({"error": str(error)}, 400)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def send_json(self, data, status=200):
            self.send_payload(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                              "application/json", status)

        def send_payload(self, payload, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    explorer = Explorer()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(explorer))
    print(f"http://127.0.0.1:{args.port}", flush=True)
    threading.Thread(target=explorer.warm_cache, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
