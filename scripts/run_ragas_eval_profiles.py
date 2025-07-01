"""
run_ragas_eval_profiles
=======================

End-to-end evaluation of the personalised *RecommendationService*.

Workflow
--------
1. Replay synthetic user histories from ``tests/profile_eval.jsonl``:
   every prior query is sent to RAG and logged into the recommender.
2. Submit the **current** user question, again logging sources & query.
3. Ask the recommender for *k=3* unseen suggestions.
4. Collect latency for both RAG and recommender layers.
5. Compute Precision@k against the expected document set.
6. Save a three-panel dashboard to
   ``dashboards/recommender_eval_dashboard.png`` showing:
      • average latency per component  
      • Precision@k distribution  
      • call volume (RAG vs recommender).

Running
-------
>>> python scripts/run_ragas_eval_profiles.py

Output
------
The script prints mean Precision@k plus latency statistics and writes
the evaluation figure to the *dashboards/* folder.

Environment
-----------
Requires the same ``OPENAI_API_KEY`` and runtime dependencies as the
main RAG service.
"""


from app.deps import get_rag, get_rec
import pandas as pd
import json
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np

total_rag_latency = 0
total_rec_latency = 0
total_rag_queries = 0
total_rec_calls = 0

examples = []
with open("tests/profile_eval.jsonl", encoding="utf-8") as f:
    for line in f:
        examples.append(json.loads(line))

rag = get_rag()
rec = get_rec()

dataset_rows = []

for idx, example in enumerate(examples):
    uid = f"user_{idx}"

    # Historial previo
    for prev_query in example["history"]:
        start_time = time.time()
        answer, sources = rag.ask(prev_query, uid=uid)
        elapsed = time.time() - start_time

        total_rag_latency += elapsed
        total_rag_queries += 1

        rec.log_sources(uid, sources)
        rec.log_query(uid, prev_query)

    # User input (también cuenta para latencia)
    start_time = time.time()
    answer, sources = rag.ask(example["user_input"], uid=uid)
    elapsed = time.time() - start_time

    total_rag_latency += elapsed
    total_rag_queries += 1

    rec.log_sources(uid, sources)
    rec.log_query(uid, example["user_input"])

    # Recommender timing
    start_time = time.time()
    recommendations = rec.recommend(uid, k=3)
    elapsed = time.time() - start_time

    total_rec_latency += elapsed
    total_rec_calls += 1

    retrieved_contexts = [r["title"].lower() for r in recommendations]

    dataset_rows.append({
        "user_input": example["user_input"],
        "reference": example["reference"],
        "retrieved_contexts": retrieved_contexts,
        "expected_sources": example["expected_sources"],
    })


def precision_at_k(row):
    """
    Precision@k for one synthetic user example.

    Parameters
    ----------
    row
        A ``pandas.Series`` with fields
        ``retrieved_contexts`` and ``expected_sources``.

    Returns
    -------
    float
        The fraction of recommended titles that match the expected set.
    """
    hits = sum(1 for doc in row["retrieved_contexts"] if doc in row["expected_sources"])
    return hits / len(row["retrieved_contexts"]) if row["retrieved_contexts"] else 0

df = pd.DataFrame(dataset_rows)
df["precision_at_k"] = df.apply(precision_at_k, axis=1)

print("\n=== Recommender Title-level Precision@k ===")
print(df["precision_at_k"].mean())

print("\n=== Latency and Call Metrics ===")
print(f"Total RAG queries: {total_rag_queries}")
print(f"Total RAG time (s): {total_rag_latency:.2f}")
print(f"Average RAG latency per query (s): {total_rag_latency / total_rag_queries:.3f}")

print(f"Total recommender calls: {total_rec_calls}")
print(f"Total Recommender time (s): {total_rec_latency:.2f}")
print(f"Average recommender latency (s): {total_rec_latency / total_rec_calls:.3f}")

# ---- Dashboard gráfico ----
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# 1. Latencia promedio
labels = ['RAG avg latency (s)', 'Recommender avg latency (s)']
latencies = [
    total_rag_latency / total_rag_queries if total_rag_queries else 0,
    total_rec_latency / total_rec_calls if total_rec_calls else 0
]
axs[0].bar(labels, latencies, color=['blue', 'orange'])
axs[0].set_title('Average Latency')
axs[0].set_ylabel('Seconds')

# 2. Distribución Precision@k
axs[1].hist(df["precision_at_k"], bins=np.linspace(0, 1, 5), edgecolor="black")
axs[1].set_title('Precision@k Distribution')
axs[1].set_xlabel('Precision@k')
axs[1].set_ylabel('Number of examples')

# 3. Número de llamadas
counts = [total_rag_queries, total_rec_calls]
axs[2].bar(['RAG queries', 'Rec calls'], counts, color=['green', 'purple'])
axs[2].set_title('Call Volume')

plt.tight_layout()

output_dir = Path("dashboards")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "recommender_eval_dashboard.png"
plt.savefig(output_path, dpi=150)
print(f"✅ Dashboard guardado en: {output_path}")
plt.show()