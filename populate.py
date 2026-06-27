import os

import service.database as db
from service.embedding.chunk import extract_chunks
from service.embedding.embedder import embed_chunks

ACCEPTED_EXTENSIONS = [".txt", ".md", ".log"]
DATA_DIR = "data/workset"


def main():
    for file in os.listdir(DATA_DIR):
        ext = os.path.splitext(file)[1]
        if ext not in ACCEPTED_EXTENSIONS:
            continue

        with open(DATA_DIR + "/" + file, "r", encoding="utf-8") as f:
            print("Embedding: " + file)
            content = f.read()
            content_chunks = extract_chunks(content, delimiter="\n\n")

            print(f"Total chunks: {len(content_chunks)}")
            embeddings = embed_chunks(content_chunks)

            db.batch_insert_embeddings(embeddings)
            print(f"Inserted {len(embeddings)} embeddings for {file}")


if __name__ == "__main__":
    main()
