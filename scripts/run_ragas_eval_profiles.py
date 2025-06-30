import json
from ragas import evaluate, EvaluationDataset
from ragas.metrics import ResponseRelevancy, Faithfulness, ContextPrecision
from app.deps import get_rag

# Load synthetic user profiles and query histories
with open("tests/synthetic_user_profiles.json", "r", encoding="utf-8") as f:
    user_profiles = json.load(f)

with open("tests/synthetic_query_histories.json", "r", encoding="utf-8") as f:
    query_histories = json.load(f)

# Create a mapping from user_id to user profile
user_profile_map = {profile["user_id"]: profile for profile in user_profiles}

# Initialize the RAG chain
rag_chain = get_rag()._get_chain("eval")  # Ensure this returns a chain compatible with Ragas

# Prepare evaluation samples
samples = []

for user_history in query_histories:
    user_id = user_history["user_id"]
    profile = user_profile_map.get(user_id, {})
    for query in user_history["query_history"]:
        # Invoke the RAG chain with the user's query
        result = rag_chain.invoke({"question": query})
        response = result.get("answer", "")
        contexts = [doc.page_content for doc in result.get("source_documents", [])]

        # Create an evaluation sample
        sample = {
            "user_input": query,
            "response": response,
            "retrieved_contexts": contexts,
            "reference": "",  # Provide the reference answer if available
            "metadata": {
                "user_id": user_id,
                "role": profile.get("role", ""),
                "expertise": profile.get("expertise", ""),
                "goals": profile.get("goals", [])
            }
        }
        samples.append(sample)

# Convert samples to EvaluationDataset
dataset = EvaluationDataset.from_list(samples)

# Evaluate the dataset
result = evaluate(
    dataset,
    metrics=[ResponseRelevancy(), Faithfulness(), ContextPrecision()],
)

# Print evaluation results
print("\n=== RAGAS Evaluation Results ===")
for metric, score in result.items():
    print(f"{metric:22s}: {score:.3f}")
