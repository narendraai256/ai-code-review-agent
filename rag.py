
import requests
import psycopg
import numpy as np
import os

from pgvector.psycopg import register_vector


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"

EMBEDDING_MODEL = "nomic-embed-text"

TOP_K = 3


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():
    connection = psycopg.connect(
    host="localhost",
    port=5432,
    dbname="code_review",
    user="postgres",
    password=os.environ["DB_PASSWORD"]
)

    register_vector(connection)

    return connection


# ============================================================
# CREATE QUERY EMBEDDING
# ============================================================

def create_embedding(text):
    if not text or not text.strip():
        print("Embedding input is empty.")
        return None

    try:
        response = requests.post(
            OLLAMA_EMBED_URL,
            json={
                "model": EMBEDDING_MODEL,
                "input": text
            },
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        if "embeddings" not in data or not data["embeddings"]:
            print("Embedding response did not contain a vector.")
            return None

        embedding = np.array(
            data["embeddings"][0],
            dtype=np.float32
        )

        if embedding.shape != (768,):
            print("Unexpected embedding dimensions:", embedding.shape)
            return None

        return embedding

    except requests.RequestException as error:
        print("Embedding request failed:", error)
        return None

    except (ValueError, KeyError, IndexError, TypeError) as error:
        print("Invalid embedding response:", error)
        return None


# ============================================================
# RETRIEVE RELEVANT KNOWLEDGE FROM POSTGRESQL
# ============================================================

def retrieve_context(query):
    if not query or not query.strip():
        print("Retrieval query is empty.")
        return None

    # Convert developer code into a query vector.
    query_vector = create_embedding(query)

    if query_vector is None:
        print("Could not create query embedding.")
        return None

    try:
        # Open a PostgreSQL connection with pgvector support.
        with get_db_connection() as connection:

            with connection.cursor() as cursor:

                # Find the three stored vectors closest to our query.
                cursor.execute(
                    """
                    SELECT
                        chunk_text,
                        source_file,
                        chunk_id,
                        1 - (embedding <=> %s) AS similarity
                    FROM knowledge_embeddings
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (
                        query_vector,
                        query_vector,
                        TOP_K
                    )
                )

                results = cursor.fetchall()

        if not results:
            print("No knowledge embeddings found in PostgreSQL.")
            return None

        # Combine the retrieved guideline text into one context string.
        context = "\n\n".join(
            chunk_text
            for chunk_text, source_file, chunk_id, similarity in results
        )

        return context

    except psycopg.Error as error:
        print("PostgreSQL vector search failed:", error)
        return None
