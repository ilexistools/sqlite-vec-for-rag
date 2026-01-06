from embeddings.sqlite_vec_search import SQLiteVecSearch
import os 

def create_index():
    idx = SQLiteVecSearch(
        db_path="data/db.sqlite3",
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    idx.add([
        {
            "text": "Python is widely used in data science.",
            "metadata": {"lang": "en", "topic": "python"}
        },
        {
            "text": "Rust focuses on memory safety and performance.",
            "metadata": {"lang": "en", "topic": "rust"}
        },
        {
            "text": "Java is common in enterprise backends.",
            "metadata": {"lang": "en", "topic": "java"}
        }
    ])
    print("Documents in the index:", idx.count())

def test_search():
    idx = SQLiteVecSearch(
        db_path="data/db.sqlite3",
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    results = idx.search("Which programming language is fast and safe?", top_k=2)
    print("Search results:", results)

if __name__ == "__main__":
    if not os.path.exists("data/db.sqlite3"):
        create_index()
    test_search()
