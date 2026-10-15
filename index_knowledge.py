import requests
import psycopg
import numpy as np
import os

from pgvector.psycopg import register_vector


# ============================================================
# CONFIGURATION
# ============================================================

# Knowledge document that will be stored in the vector database.
KNOWLEDGE_FILE = "knowledge/coding_guidelines.txt"

# Metadata name stored with each knowledge chunk.
SOURCE_FILE = "coding_guidelines.txt"

# Ollama embedding API.
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"

# Local embedding model.
EMBEDDING_MODEL = "nomic-embed-text"


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

    # Enables pgvector support for this Psycopg connection.
    register_vector(connection)

    return connection


# ============================================================
# LOAD KNOWLEDGE
# ============================================================

def load_knowledge():
    try:
        with open(
            KNOWLEDGE_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            knowledge = file.read()

        if not knowledge.strip():
            print("Knowledge file is empty.")
            return None

        return knowledge

    except FileNotFoundError:
        print(
            "Knowledge file not found:",
            KNOWLEDGE_FILE
        )
        return None

    except OSError as error:
        print(
            "Could not read knowledge file:",
            error
        )
        return None


# ============================================================
# CREATE EMBEDDING
# ============================================================

def create_embedding(text):
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

        if (
            "embeddings" not in data
            or not data["embeddings"]
        ):
            print(
                "Embedding response did not contain a vector."
            )
            return None

        embedding = np.array(
            data["embeddings"][0],
            dtype=np.float32
        )

        # Our PostgreSQL column is VECTOR(768).
        if len(embedding) != 768:
            print(
                "Unexpected embedding dimension:",
                len(embedding)
            )
            return None

        return embedding

    except requests.RequestException as error:
        print(
            "Embedding request failed:",
            error
        )
        return None

    except (
        ValueError,
        KeyError,
        IndexError,
        TypeError
    ) as error:
        print(
            "Invalid embedding response:",
            error
        )
        return None


# ============================================================
# DELETE OLD KNOWLEDGE VECTORS
# ============================================================

def delete_existing_embeddings(
    connection,
    source_file
):
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            DELETE FROM knowledge_embeddings
            WHERE source_file = %s
            """,
            (source_file,)
        )

    finally:
        cursor.close()


# ============================================================
# SAVE ONE KNOWLEDGE VECTOR
# ============================================================

def save_embedding(
    connection,
    chunk_text,
    source_file,
    chunk_id,
    embedding
):
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO knowledge_embeddings (
                chunk_text,
                source_file,
                chunk_id,
                embedding
            )
            VALUES (%s, %s, %s, %s)
            """,
            (
                chunk_text,
                source_file,
                chunk_id,
                embedding
            )
        )

    finally:
        cursor.close()


# ============================================================
# INDEX KNOWLEDGE
# ============================================================

def index_knowledge():
    # Step 1:
    # Read coding_guidelines.txt.
    knowledge = load_knowledge()

    if knowledge is None:
        return


    # Step 2:
    # Split the document into meaningful chunks.
    chunks = [
        chunk.strip()
        for chunk in knowledge.split("\n\n")
        if chunk.strip()
    ]

    if not chunks:
        print(
            "No knowledge chunks were created."
        )
        return

    print(
        f"Found {len(chunks)} knowledge chunks."
    )


    connection = None

    try:
        # Step 3:
        # Connect to PostgreSQL + pgvector.
        connection = get_db_connection()


        # Step 4:
        # Remove vectors previously created from this file.
        delete_existing_embeddings(
            connection,
            SOURCE_FILE
        )


        # Step 5:
        # Create and store an embedding for every chunk.
        for chunk_id, chunk in enumerate(
            chunks,
            start=1
        ):
            print(
                f"Creating embedding for chunk {chunk_id}..."
            )

            embedding = create_embedding(
                chunk
            )

            if embedding is None:
                raise RuntimeError(
                    f"Could not create embedding "
                    f"for chunk {chunk_id}."
                )


            save_embedding(
                connection=connection,
                chunk_text=chunk,
                source_file=SOURCE_FILE,
                chunk_id=chunk_id,
                embedding=embedding
            )


            print(
                f"Stored chunk {chunk_id}."
            )


        # Step 6:
        # Save all database changes together.
        connection.commit()

        print()
        print(
            "Knowledge indexing completed successfully."
        )

        print(
            f"Stored {len(chunks)} vectors "
            "in PostgreSQL."
        )


    except (
        psycopg.Error,
        RuntimeError
    ) as error:

        # If something fails halfway through,
        # undo the database changes from this run.
        if connection is not None:
            connection.rollback()

        print()
        print(
            "Knowledge indexing failed:",
            error
        )


    finally:
        # Always close the database connection.
        if connection is not None:
            connection.close()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    index_knowledge()
