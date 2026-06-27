import service.database as db
import service.embedding as embedding
import service.enrichment as enrichment
from service.logging import log


def main():
    query = input("Inserisci la tua query: ")
    log("Query utente (RAW): " + query)

    enriched_query = enrichment.enrich_query(query)
    log("Query Enriched (LLM): " + enriched_query)

    query_embedding = embedding.embed_query(enriched_query)
    db_found_chunks = db.query_near_chunks(query_embedding)
    log("Chunk Trovati (RAG): " + "\n\n".join(db_found_chunks))

    enriched_response = enrichment.enrich_response(enriched_query, db_found_chunks)
    log("Risposta Migliorata: " + enriched_response)
    print(enriched_response)


if __name__ == "__main__":
    main()
