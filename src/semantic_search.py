
from sentence_transformers import SentenceTransformer, util

# 1. Sample QA knowledge base
documents = [
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

# 2. Load a pretrained sentence embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 3. Convert each document into a numerical embedding
document_embeddings = model.encode(
    documents,
    convert_to_tensor=True
)


def search(query: str, top_k: int = 3) -> None:
    """Find the most semantically similar documents."""

    # Convert the user's query into an embedding
    query_embedding = model.encode(
        query,
        convert_to_tensor=True
    )

    # Compare the query embedding with every document embedding
    scores = util.cos_sim(query_embedding, document_embeddings)[0]

    # Return the indices of the highest-scoring documents
    results = scores.topk(k=min(top_k, len(documents)))

    print(f"\nQuery: {query}")
    print("-" * 60)

    for score, index in zip(results.values, results.indices):
        print(f"Similarity: {score.item():.3f}")
        print(f"Document: {documents[index.item()]}")
        print()



if __name__ == "__main__":
    print("AI QA Copilot — Semantic Search")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nEnter your QA question: ").strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

        if not query:
            print("Please enter a question.")
            continue

        search(query, top_k=3)

