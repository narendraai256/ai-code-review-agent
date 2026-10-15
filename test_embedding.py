import requests
import numpy as np


def create_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": text
        }
    )

    data = response.json()

    return data["embeddings"][0]


def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)

    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity


vector_a = create_embedding(
    "Never hardcode passwords in source code."
)

vector_b = create_embedding(
    "Use meaningful variable names."
)

query_vector = create_embedding(
    'password = "admin123"'
)


similarity_a = cosine_similarity(query_vector, vector_a)

similarity_b = cosine_similarity(query_vector, vector_b)


print("Password guideline similarity:", similarity_a)

print("Variable naming guideline similarity:", similarity_b)
results = [
    ("Never hardcode passwords in source code.", similarity_a),
    ("Use meaningful variable names.", similarity_b)
]

best_match = max(results, key=lambda item: item[1])

print("Best matching knowledge:")
print(best_match[0])

print("Similarity score:")
print(best_match[1])
