import numpy as np
import os
from ml.embedder import embed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

safe_centroids = np.load(os.path.join(BASE_DIR, "data", "safe_centroids.npy"))
unsafe_centroids = np.load(os.path.join(BASE_DIR, "data", "unsafe_centroids.npy"))

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def lse_pool(scores, alpha=10.0):
    scores = np.array(scores)
    return (1.0 / alpha) * np.log(np.sum(np.exp(alpha * scores)) + 1e-9)


def max_sim_lse(x, centroids):
    sims = [cosine(x, c) for c in centroids]
    return lse_pool(sims)


def compute_scores(embedding, safe_centroids, unsafe_centroids):
    safe_sim = max_sim_lse(embedding, safe_centroids)
    unsafe_sim = max_sim_lse(embedding, unsafe_centroids)

    margin = unsafe_sim - safe_sim
    risk = margin

    return safe_sim, unsafe_sim, margin, risk