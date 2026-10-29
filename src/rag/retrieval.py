from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_community.embeddings import FastEmbedEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
# Apne LLM ka import yahan karein (jo aap app.py mein use kar rahe hain)
# from langchain_community.llms import Ollama
# llm = Ollama(model="phi")

# 1. Models setup karein (Ragas ko batayein kaunsa LLM aur Embedding use karna hai)
evaluator_llm = LangchainLLMWrapper(
    your_llm_instance
)  # Jo bhi LLM aap use kar rahe hain
evaluator_embeddings = LangchainEmbeddingsWrapper(
    FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
)

# 2. Test Questions aur Ground Truths (Reference answers)
questions = [
    "What is the maximum loan-to-value ratio for a home loan?",
    "What is EBLR and how does it work?",
    "What documents are needed for a loan?",
]

ground_truths = [
    "Under SBI rules, LTV is 90% for loans up to Rs. 30 lakh, 80% for Rs. 30-75 lakh, and 75% above Rs. 75 lakh.",  # Aapke PDF ke hisaab se
    "EBLR is a floating interest rate benchmarked to the RBI REPO rate. When the REPO changes, the home loan interest rate changes automatically.",
    "Applicants need a copy of the sanctioned plan, an affidavit-cum-undertaking, and architect certification for construction loans.",
]

# 3. Aapke RAG system se answers aur contexts nikaalein
answers = []
contexts = []

for q in questions:
    # Yahan aap apna retrieval logic call karein (jo aapne pehle test kiya tha)
    # retrieved_docs = your_retriever.invoke(q)
    # context_list = [doc.page_content for doc in retrieved_docs]

    # Yahan apna generation logic call karein
    # answer = your_llm_chain.invoke({"question": q, "context": context_list})

    # Abhi ke liye dummy data daal rahe hain
    contexts.append(["dummy context 1", "dummy context 2"])
    answers.append("dummy answer")

# 4. Dataset banayein
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
}
dataset = Dataset.from_dict(data)

# 5. Evaluate karein
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)

print(result)
df = result.to_pandas()
print(df.head())
