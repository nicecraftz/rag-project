import ollama

from service.env_provider import CHUNK_MAX_WORDS, EMBEDDING_MODEL


def calculate_embeddings(content: str) -> list[dict[str, list[float]]]:
    chunks = extract_chunks(content)
    response = ollama.embed(model=EMBEDDING_MODEL, input=chunks)

    vector_data = []
    for i, vector in enumerate(response.embeddings):
        vector_data.append({"content": chunks[i], "embedding": vector})

    return vector_data


def embed_query(content: str) -> list[float]:
    response = ollama.embed(model=EMBEDDING_MODEL, input=content)
    return list(response.embeddings[0])


def extract_chunks(content: str) -> list[str]:
    words = content.split()
    chunks = []

    for i in range(0, len(words), CHUNK_MAX_WORDS):
        chunks.append(" ".join(words[i : i + CHUNK_MAX_WORDS]))

    return chunks
