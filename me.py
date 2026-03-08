import json
import time
import numpy as np
import pandas as pd
from huggingface_hub import InferenceClient

from model_prediction import model_prediction


# ---------------------------
# 1) Embeddings + cosine match
# ---------------------------

HF_TOKEN = ""
EMBED_MODEL = "intfloat/multilingual-e5-large"

client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

def embed(texts: list[str]) -> np.ndarray:
    """
    Retourne une matrice (n, d) de vecteurs normalisés.
    Avec e5: utiliser "query:" et "passage:".
    """
    vecs = []
    for t in texts:
        v = client.feature_extraction(t, model=EMBED_MODEL)
        v = np.array(v, dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-12)  # normalisation
        vecs.append(v)
    return np.vstack(vecs)

def coverage_cosine(expected_keywords: list[str],
                    candidate_keywords: list[str],
                    threshold: float = 0.80) -> float:
    """
    Pour chaque expected_keyword, on prend le meilleur cosine vs candidats.
    On compte couvert si score >= threshold.
    Retourne un % de couverture.
    """
    if not expected_keywords:
        return 0.0
    if not candidate_keywords:
        return 0.0

    exp_vecs = embed([f"query: {k}" for k in expected_keywords])           # (E, d)
    cand_vecs = embed([f"passage: {k}" for k in candidate_keywords])       # (C, d)

    sims = exp_vecs @ cand_vecs.T                                         # (E, C)
    best_scores = sims.max(axis=1)                                        # (E,)
    covered = int((best_scores >= threshold).sum())

    return covered / len(expected_keywords) * 100.0


# Optionnel: au lieu de matcher sur des mots, on peut matcher sur n-grams (souvent meilleur)
def make_ngrams(tokens: list[str], n_min=1, n_max=3) -> list[str]:
    out = []
    L = len(tokens)
    for n in range(n_min, n_max + 1):
        for i in range(L - n + 1):
            out.append(" ".join(tokens[i:i+n]))
    return out


# ---------------------------
# 2) Benchmark loop
# ---------------------------

with open("data/golden-set.json", "r") as file:
    golden_set = json.load(file)["golden_set"]

model_evaluation = []

for item in golden_set:
    question = item["question"]
    expected_answer = item["expected_answer_summary"]
    expected_keywords = item.get("expected_keywords", [])

    try:
        start_time = time.time()
        final_answer, model = model_prediction(question)
        end_time = time.time()

        model_answer = final_answer
        model_init_keywords = model.get("standard_keyword", []) or []

        # ✅ fix split
        tokens = model_answer.split()

        # Choisis l’un des deux:
        # 1) simple (tokens)
        # model_answer_candidates = tokens
        # 2) meilleur (n-grams)
        model_answer_candidates = make_ngrams(tokens, 1, 3)

        keyword_match_pct = coverage_cosine(
            expected_keywords=expected_keywords,
            candidate_keywords=model_answer_candidates,
            threshold=0.80
        )

        init_keyword_match_pct = coverage_cosine(
            expected_keywords=expected_keywords,
            candidate_keywords=model_init_keywords,
            threshold=0.80
        )

        question_eval = {
            "question": question,
            "expected_answer": expected_answer,
            "model_answer_time_min": (end_time - start_time) / 60,
            "model_answer": model_answer,
            "keywords_match_pct_cosine": keyword_match_pct,
            "context_keywords_match_pct_cosine": init_keyword_match_pct,
        }

    except Exception as e:
        question_eval = {
            "question": question,
            "expected_answer": expected_answer,
            "model_answer_time_min": None,
            "model_answer": None,
            "keywords_match_pct_cosine": None,
            "context_keywords_match_pct_cosine": None,
            "error": repr(e),
        }

    print(question_eval)
    model_evaluation.append(question_eval)

df = pd.DataFrame(model_evaluation)
df.to_parquet("data/model_benchmark.parquet", index=False)
