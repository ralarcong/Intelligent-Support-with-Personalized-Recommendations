# scripts/run_ragas_eval.py  – stable & concise
from ragas import evaluate, EvaluationDataset
from ragas.metrics import ResponseRelevancy, Faithfulness, ContextPrecision
from app.deps import get_rag
import pandas as pd

dataset = EvaluationDataset.from_jsonl("tests/qa_eval.jsonl")

chain = get_rag()._get_chain("eval")
for s in dataset:
    out = chain.invoke({"question": s.user_input})
    s.response            = out["answer"]
    s.retrieved_contexts  = [d.page_content for d in out["source_documents"]]

result = evaluate(
    dataset,
    metrics=[ResponseRelevancy(), Faithfulness(), ContextPrecision()],
)

# Convert the EvaluationResult to a pandas DataFrame
df = result.to_pandas()

# Display the scores
print("\n=== RAGAS scores ===")
for metric in ['response_relevancy', 'faithfulness', 'context_precision']:
    if metric in df.columns:
        score = df[metric].mean()
        print(f"{metric:22s}: {score:.3f}")
    else:
        print(f"{metric:22s}: Metric not found in the results.")






