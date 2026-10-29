import os
import sys
import warnings
from pathlib import Path

# RAGAS ki DeprecationWarnings sirf noise hain, script nahi rokti
warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq

# 1. Paths / Config
DB_DIR = ROOT / "data" / "rag" / "chroma_db"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION = "loanlens"

# llama-3.3-70b-versatile Groq ne 16 Aug 2026 ko band kar diya.
# Model ab env var se bhi badal sakte ho:  $env:GROQ_MODEL="qwen/qwen3.6-27b"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# 2. Setup Models
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY set nahi hai!")

print(f"Using Groq model: {GROQ_MODEL}")

# Generation ke liye
generator_llm = ChatGroq(model=GROQ_MODEL, api_key=api_key, temperature=0)

# RAGAS judge ke liye (same model)
evaluator_llm = LangchainLLMWrapper(
    ChatGroq(model=GROQ_MODEL, api_key=api_key, temperature=0)
)

embeddings = FastEmbedEmbeddings(model_name=EMBED_MODEL)
evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

# 3. ChromaDB Load
print("Loading ChromaDB...")
db = Chroma(
    persist_directory=str(DB_DIR),
    collection_name=COLLECTION,
    embedding_function=embeddings,
)
if db._collection.count() == 0:
    raise RuntimeError(f"ChromaDB khaali hai: {DB_DIR} (collection '{COLLECTION}')")

# 4. Test Data
questions = [
    "What is the maximum loan-to-value ratio for a home loan?",
    "What is EBLR and how does it work?",
]
ground_truths = [
    "Under SBI rules, LTV is 90% for loans up to Rs. 30 lakh, 80% for Rs. 30-75 lakh, and 75% above Rs. 75 lakh.",
    "EBLR is a floating interest rate benchmarked to the RBI REPO rate. When the REPO changes, the home loan interest rate changes automatically.",
]

answers = []
contexts = []

print("Generating answers from your RAG system...")
for q in questions:
    # Retrieval
    docs = db.similarity_search(q, k=3)
    context_list = [doc.page_content for doc in docs]
    contexts.append(context_list)

    # Generation - context ko list ke bajaye saaf text ki tarah bhejo
    context_text = "\n\n---\n\n".join(context_list)
    prompt = (
        "Answer the question based only on the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {q}\nAnswer:"
    )
    response = generator_llm.invoke(prompt)
    answers.append(response.content.strip())
    print(f"  done: {q}")

# 5. Evaluate
print("Running Ragas Evaluation...")
dataset = Dataset.from_dict(
    {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
)

# Groq free tier pe rate limit (429) se bachne ke liye ek-ek call, retries ke saath
run_config = RunConfig(max_workers=1, timeout=180, max_retries=10, max_wait=60)

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
    run_config=run_config,
)

print("\n================ RESULTS ================")
print(result)

# Har question ka detail score CSV mein
out_csv = ROOT / "data" / "rag" / "ragas_results.csv"
df = result.to_pandas()
df.to_csv(out_csv, index=False)
print(f"\nPer-question scores saved: {out_csv}")
