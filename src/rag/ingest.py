"""Builds the policy vector database from the RBI, SBI and FAQ PDFs.

Run from anywhere:  python -m src.rag.ingest
"""

import shutil
from pathlib import Path

import pymupdf
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[2]
PDFS = [
    ROOT / "data/rag/documents/rbi.pdf",
    ROOT / "data/rag/documents/sbi.pdf",
    ROOT / "data/rag/documents/faq.pdf",
]
DB_DIR = ROOT / "data/rag/chroma_db"

# These three must match app.py exactly.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION = "loanlens"
COLLECTION_METADATA = {"hnsw:space": "cosine"}


def to_markdown(rows):
    rows = [[str(c or "").replace("\n", " ").strip() for c in r] for r in rows]
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    lines.insert(1, "|" + "---|" * len(rows[0]))
    return "\n".join(lines)


text_docs, table_docs = [], []
for path in PDFS:
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    # Check if the current file is the FAQ
    is_faq = "faq" in path.name.lower()

    for page_no, page in enumerate(pymupdf.open(path), start=1):
        tables = page.find_tables().tables
        rects = [pymupdf.Rect(t.bbox) for t in tables]

        text = "\n".join(
            b[4]
            for b in page.get_text("blocks", sort=True)
            if b[6] == 0
            and not any(
                r.contains(pymupdf.Point((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
                for r in rects
            )
        ).strip()

        if text:
            text_docs.append(
                Document(
                    text,
                    metadata={
                        "source": path.name,
                        "page": page_no,
                        "type": "text",
                        "is_faq": is_faq,
                    },
                )
            )

        for t in tables:
            rows = t.extract()
            if len(rows) > 1:
                table_docs.append(
                    Document(
                        to_markdown(rows),
                        metadata={
                            "source": path.name,
                            "page": page_no,
                            "type": "table",
                            "is_faq": False,
                        },
                    )
                )

# Separate chunking logic: Small chunks for FAQ, larger for PDFs
pdf_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
faq_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=0, separators=["\n\n", "\n", ". ", " "]
)

text_chunks = []
for doc in text_docs:
    if doc.metadata.get("is_faq"):
        text_chunks.extend(faq_splitter.split_documents([doc]))
    else:
        text_chunks.extend(pdf_splitter.split_documents([doc]))

# Clean up junk dots
text_chunks = [c for c in text_chunks if "......" not in c.page_content]

# Combine text and table chunks
chunks = [c for c in text_chunks + table_docs if c.page_content.strip()]

print(
    "Pages:",
    len(text_docs),
    "| Text chunks:",
    len(text_chunks),
    "| Table chunks:",
    len(table_docs),
    "| Total:",
    len(chunks),
)
if not chunks:
    raise SystemExit("No text found in the PDFs (are they scanned images?).")

if DB_DIR.exists():
    print("Deleting old database:", DB_DIR)
    shutil.rmtree(DB_DIR)

embeddings = FastEmbedEmbeddings(model_name=EMBED_MODEL)
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(DB_DIR),
    collection_name=COLLECTION,
    collection_metadata=COLLECTION_METADATA,
)
print(f"Chunks stored: {db._collection.count()} in {DB_DIR}")

# Test search
for q in [
    "What is the maximum loan-to-value ratio for a home loan?",
    "What documents are needed for a loan?",
    "What is EBLR and how does it work?",
]:
    print("\nTest question:", q)
    for doc, score in db.similarity_search_with_relevance_scores(q, k=3):
        preview = doc.page_content[:70].replace("\n", " ")
        print(
            f"  score {score:.2f} | {doc.metadata['source']} p.{doc.metadata['page']} | {preview}..."
        )
print("\nVector database created successfully!")
