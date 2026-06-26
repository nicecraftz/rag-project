import ollama
from service.env_provider import EMBEDDING_MODEL, CHUNK_MAX_WORDS

def calculate_embeddings(file_path: str):
    chunks = extract_chunks(file_path=file_path)
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=chunks
    )

    vector_data = []
    for i, vector in enumerate(response.embeddings):
        vector_data.append({
            "content": chunks[i],
            "embedding": vector
        })
    
    return vector_data
    

def extract_chunks(file_path: str) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    words = content.split()
    chunks = []

    for i in range(0, len(words), CHUNK_MAX_WORDS):
        chunks.append(" ".join(words[i:i + CHUNK_MAX_WORDS]))

    return chunks

