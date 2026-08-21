from pathlib import Path
import json


INPUT_DIR = Path("extracted")
OUTPUT_DIR = Path("chunks")

CHUNK_SIZE = 400
OVERLAP = 80


def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = min(start + chunk_size, len(words))

        chunk_text = " ".join(words[start:end])

        if chunk_text.strip():
            chunks.append(chunk_text)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    files = list(INPUT_DIR.glob("*.jsonl"))

    total_chunks = 0

    print(f"Toplam kaynak: {len(files)}")

    for file_path in files:

        print(f"\nİşleniyor: {file_path.name}")

        chunks = []

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                record = json.loads(line)

                text = record["text"]

                page_chunks = create_chunks(text)

                for chunk_id, chunk_text in enumerate(page_chunks):

                    chunk = {
                        "source": record["source"],
                        "file_type": record["file_type"],
                        "page": record["page"],
                        "chunk_id": chunk_id,
                        "text": chunk_text
                    }

                    chunks.append(chunk)

        output_file = OUTPUT_DIR / file_path.name

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            for chunk in chunks:
                f.write(
                    json.dumps(
                        chunk,
                        ensure_ascii=False
                    ) + "\n"
                )

        print(f"  Chunk sayısı: {len(chunks)}")
        print(f"  Kaydedildi: {output_file}")

        total_chunks += len(chunks)

    print("\n==============================")
    print("CHUNKING TAMAMLANDI")
    print("==============================")
    print(f"Toplam chunk: {total_chunks}")


if __name__ == "__main__":
    main()