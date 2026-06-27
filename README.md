# Progetto RAG

Questo e' il mio primo progetto Retrieval-Augmented Generation (RAG).

Sono appassionato dell'utilizzo dell'IA per rendere le informazioni più accessibili. Questo progetto implementa una piattaforma di embedding leggera, progettata per recuperare il contesto più pertinente e ancorare le risposte dell'LLM ai dati reali, garantendo informazioni accurate e contestualizzate.

## Architettura e Tech Stack

- **Motore LLM:** [Ollama](https://ollama.com/) in esecuzione locale.
- **Modello di Arricchimento e Generazione:** `qwen3:8b`
- **Modello di Embedding:** `qwen3-embedding:0.6b`
- **Database Vettoriale:** PostgreSQL con l'estensione `pgvector`.

## Come Funziona

La pipeline di recupero e generazione segue questi passaggi fondamentali:

1. **Inserimento della Query:** L'utente inserisce una domanda direttamente nel terminale.
2. **Arricchimento della Query:** Il modello `qwen3:8b` analizza ed espande la query iniziale dell'utente per migliorarne la corrispondenza semantica.
3. **Embedding Vettoriale:** La query arricchita viene elaborata dal modello `qwen3-embedding:0.6b` e convertita in una rappresentazione vettoriale.
4. **Ricerca per Similarità:** Gli embedding sono salvati in un database PostgreSQL. Il sistema utilizza l'operatore `<=>` di `pgvector` per calcolare la distanza del coseno direttamente all'interno del DBMS, recuperando in modo efficiente i 5 chunk più rilevanti (con un limite massimo definito da `CHUNK_MAX_WORDS` dentro il .env).
5. **Generazione Contestuale:** La query arricchita e i chunk recuperati vengono passati nuovamente all'LLM. Il modello genera una risposta finale basandosi esclusivamente e rigorosamente sul contesto fornito, prevenendo così le allucinazioni.

## Guida all'Uso

Questo progetto utilizza [uv](https://github.com/astral-sh/uv) per una gestione rapida dei pacchetti Python.

### Prerequisiti

- Python installato sul tuo computer.
- `uv` installato (`pip install uv` oppure tramite il gestore di pacchetti del tuo sistema operativo).
- Ollama in esecuzione locale con i modelli necessari già scaricati:
  ```bash
  ollama pull qwen3:8b
  ollama pull qwen3-embedding:0.6b
  ```
- PostgreSQL installato e in esecuzione con l'estensione `pgvector` abilitata.

### Configurazione

Crea un file `.env` nella directory principale del progetto e configura le tue variabili d'ambiente:

```env
OLLAMA_EMBEDDING_MODEL="qwen3-embedding:0.6b"
OLLAMA_ENRICHMENT_MODEL="qwen3:8b"
CHUNK_MAX_WORDS=300

DB_USER="postgres"
DB_PASSWORD="your_password"
DB_HOST="localhost"
DB_PORT=5432
DB_NAME="rag_project"
```

### Installazione ed Esecuzione
1. per l'esecuzione di pgvector vvia il compose direttamente da terminale
    ```bash
    docker compose up -d
    ```

2. **Installa le dipendenze:**
    ```bash
    uv sync
    ```

3. **Popola il database vettoriale:**
    Posiziona il documento che desideri elaborare all'interno della cartella `data/`. Lo script leggerà e creerà gli embedding per il singolo file presente in questa directory.
    ```bash
    uv run populate.py
    ```

4. **Avvia l'applicazione:**
    Lancia l'interfaccia da terminale per iniziare a interrogare i tuoi dati:
    ```bash
    uv run main.py
    ```
