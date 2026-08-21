from pathlib import Path
import json

from pypdf import PdfReader
from docx import Document


DOCUMENTS_DIR = Path("documents")
OUTPUT_DIR = Path("extracted")


def extract_pdf(file_path):
    reader = PdfReader(file_path)
    records = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            records.append({
                "source": file_path.name,
                "file_type": "pdf",
                "page": page_number,
                "text": text.strip()
            })

    return records


def extract_docx(file_path):
    document = Document(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    full_text = "\n".join(paragraphs)

    if not full_text.strip():
        return []

    return [{
        "source": file_path.name,
        "file_type": "docx",
        "page": None,
        "text": full_text
    }]


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    files = list(DOCUMENTS_DIR.glob("*.pdf"))
    files += list(DOCUMENTS_DIR.glob("*.docx"))

    print(f"Toplam doküman: {len(files)}")

    total_pages_or_docs = 0

    for file_path in files:

        print(f"\nİşleniyor: {file_path.name}")

        if file_path.suffix.lower() == ".pdf":
            records = extract_pdf(file_path)
        else:
            records = extract_docx(file_path)

        output_file = OUTPUT_DIR / f"{file_path.stem}.jsonl"

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            for record in records:
                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )

        print(f"  Çıkarılan kayıt: {len(records)}")
        print(f"  Kaydedildi: {output_file}")

        total_pages_or_docs += len(records)

    print("\n==============================")
    print("INGESTION TAMAMLANDI")
    print("==============================")
    print(f"Toplam kayıt: {total_pages_or_docs}")


if __name__ == "__main__":
    main()