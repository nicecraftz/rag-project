import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.connection import Connection

from service.env_provider import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER


def get_connection() -> Connection:
    conn = psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    register_vector(conn)
    return conn


def create_missing_tables():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding VECTOR(1024) NOT NULL
                );
            """)


def batch_insert_embeddings(vector_data: list[dict[str, list[float]]]):
    tuple_values = [
        (item["content"], Vector(item["embedding"])) for item in vector_data
    ]
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO embeddings (text, embedding) VALUES (%s, %s)",
                tuple_values,
            )


def insert_embedding(chunk: str, embedding: list[float]):
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO embeddings (text, embedding) VALUES (%s, %s)",
                (chunk, Vector(embedding)),
            )


def query_near_chunks(embedding: list[float]) -> list[str]:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT text FROM embeddings ORDER BY embedding <=> %s LIMIT 5",
                (Vector(embedding),),
            )
            result = cursor.fetchall()
            return [row[0] for row in result]


create_missing_tables()
