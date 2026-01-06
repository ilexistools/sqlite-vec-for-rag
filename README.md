# SQLite-Vec for RAG

A lightweight **local-first semantic search and RAG backend** built with **SQLite + sqlite-vec**
and **Sentence Transformers**, exposing a simple **FastAPI** interface.

---

## Features

- Semantic search with embeddings
- SQLite-based vector storage using `sqlite-vec (vec0)`
- No external vector database
- Portable and local-first
- FastAPI REST interface
- Utilities for TSV, JSON, JSONL and PDF ingestion

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

---

## Running the API

```bash
python main.py
```

Access:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

---

## API Endpoints

### Health Check
GET /

### Add Documents
POST /add

### Search
POST /search

---

## Core Usage (Python)

```python
from embeddings.sqlite_vec_search import SQLiteVecSearch

idx = SQLiteVecSearch("db.sqlite3")

idx.add([
    {"text": "Python is great for data science", "metadata": {"lang": "en"}}
])

results = idx.search("data science language", top_k=3)
```

---

## Utilities

Use `util.py` to load files:

- Text
- TSV
- JSON / JSONL
- PDF

Example:

```python
from util import load_tsv_to_json_list
docs = load_tsv_to_json_list("data.tsv")
```

---

## License

MIT
