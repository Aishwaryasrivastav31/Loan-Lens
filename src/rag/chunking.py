"""Builds the policy vector database from the RBI, SBI and FAQ PDFs.

Run from anywhere:  python src/rag/ingest.py

- Paths are found from this file's location, so the folder you run it from doesn't matter.
- The old database is deleted first, so chunks made with an older embedding
  model can never mix with the new ones.
- Cosine distance, so relevance scores are between 0 and 1.
- At the end it runs a test search and prints the scores, to check MIN_RELEVANCE in app.py.
"""

import shutil
from pathlib import Path

import pymupdf
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[2]  # src/rag/ingest.py -> project folder
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
                    metadata={"source": path.name, "page": page_no, "type": "text"},
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
                        },
                    )
                )

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
text_chunks = splitter.split_documents(text_docs)
text_chunks = [c for c in text_chunks if "......" not in c.page_content]

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

# Test search: these scores tell you whether MIN_RELEVANCE in app.py is right.
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
