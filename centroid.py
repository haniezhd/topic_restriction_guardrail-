# centroids.py 

import os
import json
import numpy as np
from ml.embedder import embed

DATA_DIR = "app/data"


def load_texts(path):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "text" in obj:
                    texts.append(obj["text"])
                    continue
            except:
                pass

            if '"' in line:
                texts.append(line.split('"')[1])

    return texts


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-9)


def build_centroid(embs):
    return normalize(np.mean(embs, axis=0))


def embed_all(texts):
    return np.array([embed(t) for t in texts])


def main():

    safe = load_texts(os.path.join(DATA_DIR, "test_safe.jsonl"))
    hard = load_texts(os.path.join(DATA_DIR, "hard_mixed.jsonl"))
    unsafe = load_texts(os.path.join(DATA_DIR, "unwanted_dataset.jsonl"))

    safe = safe + hard

    safe_emb = embed_all(safe)
    unsafe_emb = embed_all(unsafe)

    safe_centroid = build_centroid(safe_emb)
    unsafe_centroid = build_centroid(unsafe_emb)

    np.save(os.path.join(DATA_DIR, "safe_centroids.npy"), safe_centroid)
    np.save(os.path.join(DATA_DIR, "unsafe_centroids.npy"), unsafe_centroid)

    print("DONE")


if __name__ == "__main__":
    main()