import sqlite3
import numpy as np
from openai import OpenAI


DATABASE_FILE = "philosophy_rag.db"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"

FOUNDRY_URL = "http://127.0.0.1:49892/v1"

TOP_K = 5
SIMILARITY_THRESHOLD = 0.40


client = OpenAI(
    base_url=FOUNDRY_URL,
    api_key="local"
)


def get_query_embedding(query):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )

    return np.array(
        response.data[0].embedding,
        dtype=np.float32
    )


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def retrieve(query):

    query_embedding = get_query_embedding(query)

    conn = sqlite3.connect(DATABASE_FILE)

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            source,
            file_type,
            page,
            chunk_id,
            text,
            embedding
        FROM chunks
    """)

    rows = cursor.fetchall()

    conn.close()

    results = []

    for row in rows:

        (
            db_id,
            source,
            file_type,
            page,
            chunk_id,
            text,
            embedding_blob
        ) = row

        embedding = np.frombuffer(
            embedding_blob,
            dtype=np.float32
        )

        similarity = cosine_similarity(
            query_embedding,
            embedding
        )

        results.append({
            "id": db_id,
            "source": source,
            "file_type": file_type,
            "page": page,
            "chunk_id": chunk_id,
            "text": text,
            "similarity": float(similarity)
        })

    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    # Threshold uygulanıyor
    results = [
        result
        for result in results
        if result["similarity"] >= SIMILARITY_THRESHOLD
    ]

    return results[:TOP_K]


def main():

    print("=" * 50)
    print("RETRIEVAL TEST")
    print("=" * 50)

    query = input("\nSorunuzu yazın: ")

    print("\nAranıyor...")

    results = retrieve(query)

    print("\n" + "=" * 50)
    print("EN ALAKALI CHUNK'LAR")
    print("=" * 50)

    if not results:
        print("\nBu soru için yeterli bilgi bulunamadı.")
        return

    for i, result in enumerate(results, start=1):

        print(f"\n--- RESULT {i} ---")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"\n{result['text'][:800]}...")


if __name__ == "__main__":
    main()