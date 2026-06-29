from typing import Any

import ollama

from service.embedding.chunk import extract_chunks
from service.env_provider import EMBEDDING_MODEL

MAX_CHUNK_QUANTITY = 200


def embed_content(content: str, delimiter: str = "\n", max_words: int = -1):
    chunks = extract_chunks(content, delimiter, max_words)
    return embed_chunks(chunks)


def embed_query(content: str) -> list[float]:
    response = ollama.embed(model=EMBEDDING_MODEL, input=content)
    return list(response.embeddings[0])


def embed_chunks(
    chunks: list[str],
) -> list[dict[str, Any]]:
    total = len(chunks)
    completed = 0
    vector_data = []

    for i in range(0, total, MAX_CHUNK_QUANTITY):
        batch = chunks[i : i + MAX_CHUNK_QUANTITY]
        response = ollama.embed(model=EMBEDDING_MODEL, input=batch)
        completed += len(batch)
        print(
            f"Embedding progress: {completed}/{total} ({completed / total * 100:.1f}%)"
        )
        for j, vector in enumerate(response.embeddings):
            vector_data.append({"content": batch[j], "embedding": vector})

    return vector_data
