from ragas import evaluate, EvaluationDataset
from ragas.metrics import ResponseRelevancy, Faithfulness, ContextPrecision
from app.deps import get_rag
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

dataset = EvaluationDataset.from_jsonl("tests/qa_eval.jsonl")

rag = get_rag()
chain = rag._get_chain("eval")

total_latency = 0
total_queries = 0
latencies = []
inputs = []

# Procesar dataset
for s in dataset:
    start_time = time.time()
    
    out = chain.invoke({
        "question": s.user_input,
        "style": "profesional",
        "emoji": "🙂",
    })
    
    elapsed = time.time() - start_time
    total_latency += elapsed
    total_queries += 1

    # Guardar latencia individual e input
    latencies.append(elapsed)
    inputs.append(s.user_input[:50])  # Recorta el input para que el gráfico no sea ilegible

    s.response = out["answer"]
    s.retrieved_contexts = [d.page_content for d in out["source_documents"]]

# Evaluación RAGAS
result = evaluate(
    dataset,
    metrics=[ResponseRelevancy(), Faithfulness(), ContextPrecision()],
)

df = result.to_pandas()

# Mostrar métricas numéricas
print("\n=== RAGAS scores ===")
for metric in ['response_relevancy', 'faithfulness', 'context_precision']:
    if metric in df.columns:
        score = df[metric].mean()
        print(f"{metric:22s}: {score:.3f}")
    else:
        print(f"{metric:22s}: Metric not found in the results.")

# Mostrar latencia
print("\n=== Latency Metrics ===")
print(f"Total queries: {total_queries}")
print(f"Total time (s): {total_latency:.2f}")
print(f"Average latency per query (s): {total_latency / total_queries:.3f}")

# ---- Dashboard visual ----
fig, axs = plt.subplots(1, 3, figsize=(18, 4))

# 1. Latencia promedio
axs[0].bar(['Avg Latency per Query'], [total_latency / total_queries], color='blue')
axs[0].set_title('Average Latency per Query')
axs[0].set_ylabel('Seconds')

# 2. Distribución de métricas RAGAS
metrics = ['faithfulness', 'context_precision']
data = [df[m].mean() if m in df.columns else 0 for m in metrics]

axs[1].bar(metrics, data, color=['green', 'orange', 'purple'])
axs[1].set_ylim(0, 1)
axs[1].set_title('Average RAGAS Scores')
axs[1].set_ylabel('Score (0-1)')

# 3. Latencia por input
axs[2].bar(range(len(latencies)), latencies, color='red')
axs[2].set_title('Latency per Input')
axs[2].set_xlabel('Query Index')
axs[2].set_ylabel('Latency (s)')
axs[2].set_xticks(range(len(inputs)))
axs[2].set_xticklabels(inputs, rotation=90, fontsize=7)

plt.tight_layout()
output_dir = Path("dashboards")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "ragas_eval_dashboard.png"
plt.savefig(output_path, dpi=150)
print(f"✅ Dashboard guardado en: {output_path}")

plt.show()