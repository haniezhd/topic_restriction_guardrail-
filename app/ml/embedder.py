# ml/embedder.py


from sentence_transformers import SentenceTransformer

MODEL_PATH = r"E:\MODELS\multilingual-e5-base"

model = SentenceTransformer(MODEL_PATH)

def embed(text: str):
    return model.encode(
        "query: " + text,
        normalize_embeddings=True
    )