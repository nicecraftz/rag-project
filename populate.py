import service.database as db
import service.embedding as embedding


def main():
    with open("data/rag_test.txt", "r") as f:
        content = f.read()
        content_embedding = embedding.calculate_embeddings(content)

    db.batch_insert_embeddings(content_embedding)
    print(f"INFO: Populated db successfully with {len(content_embedding)} entries")


if __name__ == "__main__":
    main()
