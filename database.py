import json
import sqlite3
import numpy as np
from pathlib import Path


EMBEDDINGS_FILE = Path("embedded_chunks.jsonl")
DATABASE_FILE = Path("philosophy_rag.db")


def main():

    conn = sqlite3.connect(DATABASE_FILE)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            file_type TEXT NOT NULL,
            page INTEGER,
            chunk_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    """)

    # Baştan temiz database oluşturuyoruz
    cursor.execute("DELETE FROM chunks")

    total = 0

    with open(
        EMBEDDINGS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            chunk = json.loads(line)

            embedding = np.array(
                chunk["embedding"],
                dtype=np.float32
            )

            cursor.execute("""
                INSERT INTO chunks
                (source, file_type, page, chunk_id, text, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                chunk["source"],
                chunk["file_type"],
                chunk["page"],
                chunk["chunk_id"],
                chunk["text"],
                embedding.tobytes()
            ))

            total += 1

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM chunks")
    count = cursor.fetchone()[0]

    conn.close()

    print("==============================")
    print("SQLITE TAMAMLANDI")
    print("==============================")
    print(f"Eklenen chunk: {total}")
    print(f"Database kayıt sayısı: {count}")
    print(f"Database: {DATABASE_FILE}")


if __name__ == "__main__":
    main()