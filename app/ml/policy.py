# ml/policy.py

import numpy as np
from ml.embedder import embed
from ml.scoring import compute_scores

from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

SAFE_CENTROIDS = np.load(BASE_DIR / "data" / "safe_centroids.npy")
UNSAFE_CENTROIDS = np.load(BASE_DIR / "data" / "unsafe_centroids.npy")

from config import RISK_THRESHOLD


def predict(text: str):

    emb = embed(text)

    safe_sim, unsafe_sim, margin, risk = compute_scores(
        emb,
        SAFE_CENTROIDS,
        UNSAFE_CENTROIDS
    )

    blocked = risk > RISK_THRESHOLD

    return {
        "blocked": bool(blocked)
    }


def explain(text: str):

    emb = embed(text)

    safe_sim, unsafe_sim, margin, risk = compute_scores(
        emb,
        SAFE_CENTROIDS,
        UNSAFE_CENTROIDS
    )

    return {
        "safe_sim": safe_sim,
        "unsafe_sim": unsafe_sim,
        "margin": margin,
        "risk": risk
    }


def classify_with_score(text: str):

    emb = embed(text)

    safe_sim, unsafe_sim, margin, risk = compute_scores(
        emb,
        SAFE_CENTROIDS,
        UNSAFE_CENTROIDS
    )

    label = "unsafe" if unsafe_sim > safe_sim else "safe"

    return {
        "text": text,
        "label": label,
        "score": max(safe_sim, unsafe_sim)
    }