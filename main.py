from fastapi import FastAPI
from pydantic import BaseModel
from embeddings.sqlite_vec_search import SQLiteVecSearch

app = FastAPI(title="SQLite-Vec RAG")

idx = SQLiteVecSearch("db.sqlite3")

class DocIn(BaseModel):
    text: str
    metadata: dict

class SearchIn(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
def works():
    return {"status": "ok"}

@app.post("/add")
def add_docs(docs: list[DocIn]):
    idx.add([d.model_dump() for d in docs])
    return {"status": "ok", "count": idx.count()}

@app.post("/search")
def search(q: SearchIn):
    return idx.search(q.query, q.top_k)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
