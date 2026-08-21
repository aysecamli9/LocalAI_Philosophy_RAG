from pathlib import Path
import json
from openai import OpenAI


CHUNKS_DIR = Path("chunks")
OUTPUT_FILE = Path("embedded_chunks.jsonl")

EMBEDDING_MODEL = "qwen3-embedding-0.6b"

client = OpenAI(
    base_url="http://127.0.0.1:59861/v1",
    api_key="local"
)


def get_embedding(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding


def main():

    files = list(CHUNKS_DIR.glob("*.jsonl"))

    total = 0

    # Eski çıktı varsa temiz başlıyoruz
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output:

        for file_path in files:

            print(f"\nİşleniyor: {file_path.name}")

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    chunk = json.loads(line)

                    embedding = get_embedding(
                        chunk["text"]
                    )

                    record = {
                        "source": chunk["source"],
                        "file_type": chunk["file_type"],
                        "page": chunk["page"],
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "embedding": embedding
                    }

                    output.write(
                        json.dumps(
                            record,
                            ensure_ascii=False
                        ) + "\n"
                    )

                    total += 1

                    if total % 10 == 0:
                        print(
                            f"  {total} chunk embedding oluşturuldu"
                        )

    print("\n==============================")
    print("EMBEDDING TAMAMLANDI")
    print("==============================")
    print(f"Toplam chunk: {total}")
    print(f"Kaydedildi: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()