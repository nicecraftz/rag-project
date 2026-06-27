def log(text: str):
    with open("log.txt", "a+", encoding="utf-8") as f:
        f.write(text + "\n\n")
