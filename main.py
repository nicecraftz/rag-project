import service.embedding as embedding


def main():
    data = embedding.calculate_embeddings("data/rag_test.txt")
    print(data)

if __name__ == "__main__":
    main()