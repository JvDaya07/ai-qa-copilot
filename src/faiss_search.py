
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# Find the project root from this Python file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Save the index and metadata locally.
STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
INDEX_PATH = STORE_DIR / "qa.index"
METADATA_PATH = STORE_DIR / "metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Our initial QA knowledge base.
DOCUMENTS = [
    "Users must enter both email and password to log in.",
    "A successful login requires a registered email and correct password.",
    "Invalid login credentials should display an appropriate error message.",
    "The account must be temporarily locked after five consecutive failed login attempts.",
    "Passwords must never be displayed in plain text.",
    "The password field should mask characters while the user types.",
    "Users should be able to log out of their accounts.",
    "A user should receive an error when the email field is empty.",
    "A user should receive an error when the password field is empty.",
    "The application should validate email address format.",
    "A locked account should prevent further login attempts during the lock period.",
    "Testers should verify login functionality using valid credentials.",
    "Testers should verify login functionality using invalid credentials.",
    "The application should protect sensitive user information.",
    "The login page should display a clear validation message for missing fields.",
    "A shopping application should calculate the total price of items in a cart.",
    "Users should be able to reset a forgotten password.",
    "The application should record important security events."
]


def load_model():
    """Load the sentence embedding model."""
    return SentenceTransformer(MODEL_NAME)


def build_and_save_index(model):
    """Encode documents and persist the FAISS index and metadata."""
    STORE_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = model.encode(
        DOCUMENTS,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Each vector has one row; its column count is the vector dimension.
    dimension = embeddings.shape[1]

    # Inner product of normalized vectors equals cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    metadata = [
        {"id": i, "text": text}
        for i, text in enumerate(DOCUMENTS)
    ]

    faiss.write_index(index, str(INDEX_PATH))
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print(f"Saved FAISS index: {INDEX_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Indexed documents: {index.ntotal}")

    return index, metadata


def load_saved_index():
    """Load an existing index and its corresponding metadata."""
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Saved index or metadata is missing. Build the index first."
        )

    index = faiss.read_index(str(INDEX_PATH))
    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    if index.ntotal != len(metadata):
        raise ValueError(
            "Index and metadata do not match. Rebuild the index."
        )

    return index, metadata


def search(query, model, index, metadata, top_k=3):
    """Retrieve the most similar documents from FAISS."""
    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_vector,
        min(top_k, index.ntotal)
    )

    print(f"\nQuery: {query}")
    print("-" * 60)
    threshold = 0.35
    found_match = False

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < threshold:
            continue

        found_match = True
        document = metadata[idx]

        print(f"Similarity: {float(score):.3f}")
        print(f"Document ID: {document['id']}")
        print(f"Document: {document['text']}\n")

    if not found_match:
        print("No sufficiently relevant requirement was found.")


def main():
    model = load_model()

    # Build the index on the first run.
    if INDEX_PATH.exists() and METADATA_PATH.exists():
        index, metadata = load_saved_index()
        print("Loaded existing index from disk.")
    else:
        index, metadata = build_and_save_index(model)

    print("\nAI QA Copilot — FAISS Search")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nEnter your QA question: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Please enter a question.")
            continue

        search(query, model, index, metadata)


if __name__ == "__main__":
    main()
