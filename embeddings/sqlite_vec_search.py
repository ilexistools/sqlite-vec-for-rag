from __future__ import annotations

"""
SQLite-backed semantic index using sqlite-vec (vec0 virtual table) + Sentence-Transformers.

What you get
------------
- No Haystack dependency
- Vector search happens inside SQLite via sqlite-vec (vec0)
- Keeps your "MemorySearch-like" API: add() + search()
- Stores documents in a normal table and embeddings in a vec0 virtual table.

sqlite-vec notes
----------------
- Load extension in Python with sqlite_vec.load(conn) after enabling load_extension.
- KNN query pattern:
    SELECT id, distance FROM vec_table
    WHERE embedding MATCH :query AND k = :k;

- You can set cosine distance metric:
    CREATE VIRTUAL TABLE vec_docs USING vec0(
      doc_id INTEGER PRIMARY KEY,
      embedding float[768] distance_metric=cosine
    );
"""

from typing import List, Dict, Any, Optional, Iterable, Tuple
from pathlib import Path
import os
import json
import sqlite3
import numpy as np

import sqlite_vec  # pip install sqlite-vec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv

# Carrega o .env (se existir)
load_dotenv(find_dotenv())

_DEFAULT_MODEL = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-mpnet-base-v2")


# ---------------------------- cache utilities -----------------------------

def _models_dir(explicit: Optional[str] = None) -> Path:
    """Return ./models next to this file (or the provided explicit path)."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path(__file__).resolve().parent / "models"


def _setup_caches(models_dir: Optional[str] = None) -> Path:
    """Centralize caches under chosen directory (default: ./models)."""
    root = _models_dir(models_dir)
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(root))
    os.environ.setdefault("HF_HOME", str(root))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(root))
    return root


def download_model(model_name: str = _DEFAULT_MODEL, models_dir: Optional[str] = None) -> str:
    """Download (or confirm) model into ./models and return the repo_id."""
    target_root = _setup_caches(models_dir)
    SentenceTransformer(model_name, cache_folder=str(target_root))
    return model_name


# ------------------------------- core class --------------------------------

class SQLiteVecSearch:
    """
    SQLite-backed index for RAG using sqlite-vec.

    API:
      - add(items: List[{"text": str, "metadata": dict}]) -> None
      - search(query: str, top_k: int = 5) -> List[{"rank","similarity","distance","doc"}]

    similarity:
      - With cosine distance metric: similarity = 1 - distance
    """

    def __init__(
        self,
        db_path: str | os.PathLike = "./rag_index.sqlite3",
        *,
        model_name: str = _DEFAULT_MODEL,
        normalize: bool = True,
        device: Optional[str] = None,
        models_dir: Optional[str] = None,
        download_if_missing: bool = True,
        pragmas: Optional[Dict[str, Any]] = None,
        embedding_batch_size: int = 64,
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._normalize = normalize
        self._embedding_batch_size = int(max(1, embedding_batch_size))

        # Cache unification
        local_root = _setup_caches(models_dir)
        if download_if_missing:
            SentenceTransformer(model_name, cache_folder=str(local_root))

        # Embedder (no Haystack)
        # NOTE: SentenceTransformer accepts device like "cpu", "cuda", "cuda:0"
        self._model = SentenceTransformer(
            model_name,
            cache_folder=str(local_root),
            device=device,
        )

        # Infer embedding dimension once
        self._dim = int(self._model.get_sentence_embedding_dimension())

        # Initialize DB
        self._conn = sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False)  # autocommit
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._set_pragmas(pragmas)

        # Load sqlite-vec extension
        self._load_sqlite_vec()

        # Create schema (documents + vec0 table)
        self._create_schema()

    # --------------------------- SQLite helpers ---------------------------

    def _load_sqlite_vec(self) -> None:
        # sqlite-vec Python docs recommend enable_load_extension + sqlite_vec.load(conn) :contentReference[oaicite:2]{index=2}
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        # Optional sanity check
        _ = self._conn.execute("SELECT vec_version()").fetchone()

    def _set_pragmas(self, pragmas: Optional[Dict[str, Any]]) -> None:
        defaults = {
            "journal_mode": "WAL",
            "synchronous": 1,     # NORMAL
            "temp_store": 2,      # MEMORY
            "cache_size": -65536, # 64MB page cache (negative => KB)
            # you can also consider: "mmap_size": 268435456,  # 256MB
        }
        if pragmas:
            defaults.update(pragmas)
        cur = self._conn.cursor()
        for k, v in defaults.items():
            cur.execute(f"PRAGMA {k} = {json.dumps(v)};")
        cur.close()

    def _create_schema(self) -> None:
        cur = self._conn.cursor()

        # Documents table (your original)
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_documents_id ON documents(id);
            """
        )

        # Persist model dim in meta (so you can detect mismatches)
        cur.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            ("embedding_dim", str(self._dim)),
        )

        # vec0 virtual table for embeddings (cosine distance)
        # KNN docs show:
        #   create virtual table vec_documents using vec0(
        #     document_id integer primary key,
        #     contents_embedding float[768] distance_metric=cosine
        #   ); :contentReference[oaicite:3]{index=3}
        cur.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_documents USING vec0(
                doc_id INTEGER PRIMARY KEY,
                embedding float[{self._dim}] distance_metric=cosine
            );
            """
        )

        cur.close()

    # --------------------------- embedding utils --------------------------

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        # SentenceTransformer does batching internally; still keep your batch_size control
        all_embs: List[np.ndarray] = []
        bs = self._embedding_batch_size

        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            embs = self._model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=False,  # we'll do it ourselves if requested
                show_progress_bar=False,
            )
            embs = np.asarray(embs, dtype=np.float32)
            all_embs.append(embs)

        arr = np.vstack(all_embs).astype(np.float32, copy=False)

        if self._normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            arr = arr / norms

        return arr

    # ------------------------------- public API ---------------------------

    def add(self, items: List[Dict[str, Any]]) -> None:
        """
        Add items to the index.
        items: [{"text": str, "metadata": dict}, ...]
        """
        if not items:
            return

        texts: List[str] = []
        doc_rows: List[Tuple[str, str]] = []

        for i, it in enumerate(items):
            if not isinstance(it, dict) or "text" not in it or "metadata" not in it:
                raise ValueError(f"Item {i} must be {{'text': str, 'metadata': dict}}.")
            txt = it["text"]
            meta = it["metadata"]
            if not isinstance(txt, str):
                raise ValueError(f"Item {i}: 'text' must be a string.")
            if not isinstance(meta, dict):
                raise ValueError(f"Item {i}: 'metadata' must be a dict.")
            texts.append(txt)
            doc_rows.append((txt, json.dumps(meta, ensure_ascii=False)))

        cur = self._conn.cursor()

        # Insert documents
        cur.executemany(
            "INSERT INTO documents(text, metadata_json) VALUES (?, ?)",
            doc_rows,
        )

        # Fetch the ids of the last N inserted rows (single-writer assumption)
        n = len(doc_rows)
        id_rows = cur.execute(
            "SELECT id FROM documents ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()
        id_rows.reverse()
        doc_ids = [int(r[0]) for r in id_rows]

        # Embed texts
        embs = self._embed_texts(texts)  # shape (n, dim)
        if embs.shape[1] != self._dim:
            raise ValueError(f"Embedding dim mismatch: expected {self._dim}, got {embs.shape[1]}")

        # Insert into vec0 table
        # sqlite-vec Python docs: NumPy float32 arrays can be passed as parameters (buffer protocol). :contentReference[oaicite:4]{index=4}
        vec_rows = [(doc_ids[i], embs[i].astype(np.float32, copy=False)) for i in range(n)]
        cur.executemany(
            "INSERT OR REPLACE INTO vec_documents(doc_id, embedding) VALUES (?, ?)",
            vec_rows,
        )

        cur.close()

    def count(self) -> int:
        cur = self._conn.cursor()
        (n,) = cur.execute("SELECT COUNT(*) FROM documents").fetchone()
        cur.close()
        return int(n)

    def _fetch_docs(self, ids: Iterable[int]) -> Dict[int, Dict[str, Any]]:
        id_list = list(ids)
        if not id_list:
            return {}
        placeholders = ",".join(["?"] * len(id_list))
        cur = self._conn.cursor()
        rows = cur.execute(
            f"SELECT id, text, metadata_json FROM documents WHERE id IN ({placeholders})",
            id_list,
        ).fetchall()
        cur.close()

        out: Dict[int, Dict[str, Any]] = {}
        for (doc_id, text, meta_json) in rows:
            out[int(doc_id)] = {"text": text, "metadata": json.loads(meta_json)}
        return out

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic similarity search using sqlite-vec KNN.

        Returns:
          [{"rank": int, "similarity": float, "distance": float,
            "doc": {"text": str, "metadata": dict}}, ...]
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        k = int(max(1, top_k))

        q_emb = self._embed_texts([query])[0].astype(np.float32, copy=False)

        # KNN query via vec0 + MATCH + k=...
        # Docs show:
        #   select document_id, distance
        #   from vec_documents
        #   where contents_embedding match :query and k = 10; :contentReference[oaicite:5]{index=5}
        cur = self._conn.cursor()
        matches = cur.execute(
            """
            SELECT doc_id, distance
            FROM vec_documents
            WHERE embedding MATCH ?
              AND k = ?;
            """,
            (q_emb, k),
        ).fetchall()
        cur.close()

        if not matches:
            return []

        ids = [int(r[0]) for r in matches]
        dist_map = {int(doc_id): float(distance) for (doc_id, distance) in matches}

        docs = self._fetch_docs(ids)

        # Keep results ordered as returned by vec0 (closest first)
        results: List[Dict[str, Any]] = []
        for rank, doc_id in enumerate(ids, start=1):
            distance = dist_map.get(doc_id, 1.0)
            similarity = 1.0 - distance  # cosine distance -> similarity proxy
            results.append(
                {
                    "rank": rank,
                    "similarity": float(similarity),
                    "distance": float(distance),
                    "doc": docs.get(doc_id, {"text": "", "metadata": {}}),
                }
            )
        return results

    # --------------------------- maintenance ops --------------------------

    def vacuum(self) -> None:
        self._conn.execute("VACUUM")

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# ------------------------------- quick demo -------------------------------

if __name__ == "__main__":
    print("Downloading/confirming local model...")
    _ = download_model(_DEFAULT_MODEL)

    data = [
        {"text": "Python é muito usada em ciência de dados.", "metadata": {"id": 1, "category": "programming", "language": "Python"}},
        {"text": "Rust foca em segurança de memória e performance.", "metadata": {"id": 2, "category": "programming", "language": "Rust"}},
        {"text": "Java é comum em backends corporativos.", "metadata": {"id": 3, "category": "programming", "language": "Java"}},
    ]

    idx = SQLiteVecSearch("./demo.sqlite3", download_if_missing=False)  # model already cached
    idx.add(data)

    print(f"Total docs: {idx.count()}")
    print("\nResultados da busca:")
    for r in idx.search("linguagens de programação para backend com performance", top_k=3):
        print(f"Rank {r['rank']}: similarity={r['similarity']:.4f} distance={r['distance']:.4f}")
        print(f"  Text: {r['doc']['text']}")
        print(f"  Metadata: {r['doc']['metadata']}")
        print("")
