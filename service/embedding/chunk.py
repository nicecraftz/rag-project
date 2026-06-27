def extract_chunks(
    content: str, delimiter: str = "\n", max_words: int = -1
) -> list[str]:
    words = content.split()
    chunks = []

    if max_words == -1:
        splitted_content = content.split(delimiter)
        chunks = [c for c in splitted_content if c.strip()]
    else:
        for i in range(0, len(words), max_words):
            chunks.append(" ".join(words[i : i + max_words]))

    return chunks
