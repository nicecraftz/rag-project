import service.database as db
import service.embedding.embedder as embedder
import service.enrichment as enrichment


def main():
    query = input("Inserisci la tua query: ")
    enriched_query = enrichment.enrich_query(query)
    query_embedding = embedder.embed_query(enriched_query)
    db_found_chunks = db.query_near_chunks(query_embedding)

    enriched_response = enrichment.enrich_response(enriched_query, db_found_chunks)
    print(enriched_response)


if __name__ == "__main__":
    main()
