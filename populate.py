import os

from markitdown import MarkItDown

import service.database as db
from service.embedding.chunk import extract_chunks
from service.embedding.embedder import embed_chunks

ACCEPTED_EXTENSIONS = ["txt", "md", "log", "pdf"]
DATA_DIR = "data/workset"
MAX_WORDS = 400


def extract_from_plain_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        return content


def extract_from_pdf(path: str) -> str:
    mid = MarkItDown()
    md = mid.convert(source=path).markdown
    return md


def main():
    for file in os.listdir(DATA_DIR):
        ext = file.split(".")[-1]
        path = DATA_DIR + "/" + file
        if ext not in ACCEPTED_EXTENSIONS:
            continue

        print(f"Trying to extract from {file} ext {ext}")

        content = (
            extract_from_pdf(path) if ext == "pdf" else extract_from_plain_file(path)
        )

        content_chunks = extract_chunks(content, max_words=MAX_WORDS)

        print(f"Total chunks: {len(content_chunks)}")
        embeddings = embed_chunks(content_chunks)

        db.batch_insert_embeddings(embeddings)
        print(f"Inserted {len(embeddings)} embeddings for {file}")


if __name__ == "__main__":
    main()
