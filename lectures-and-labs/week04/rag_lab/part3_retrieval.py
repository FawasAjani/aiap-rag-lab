"""
Part 3: Retrieval -- RAG Lab

In this part, you will:
1. Implement semantic search over the chunks you stored in part 2
2. Show each hit with its score and the file it came from
3. Filter hits by a score threshold
4. Fit the winning chunks into a context window

Run it as:   python part3_retrieval.py

Estimated time: 35 minutes
"""

import chromadb
from sentence_transformers import SentenceTransformer

COLLECTION = "cs_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def semantic_search(query, collection, model, top_k=3):
    """
    Return the top_k chunks nearest in meaning to the query.
    """
    query_embedding = model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        (text, metadata["source"], 1 - distance)
        for text, metadata, distance in zip(
            documents, metadatas, distances
        )
    ]


def filter_by_relevance(results, min_similarity=0.2):
    """
    Keep only the hits whose similarity clears a threshold.
    """
    return [
        (text, source, similarity)
        for text, source, similarity in results
        if similarity >= min_similarity
    ]


def manage_context_window(results, max_tokens=1500):
    """
    Join chunk texts into one context string that fits a token budget.
    """
    pieces = []
    total_chars = 0
    max_chars = max_tokens * 4

    for text, source, similarity in results:
        piece = f"[source: {source}]\n{text}"
        piece_chars = len(piece)

        if total_chars + piece_chars <= max_chars:
            pieces.append(piece)
            total_chars += piece_chars
        else:
            break

    return "\n\n---\n\n".join(pieces)


def display_results(query, results):
    """Print hits the way the README shows them: score, source, preview."""
    print(f'Query: "{query}"')
    if not results:
        print("  no results -- check your semantic_search() function")
        return
    for text, source, similarity in results:
        preview = " ".join(text.split()[:6])
        print(f'  {similarity:.2f}  {source:<34} "{preview}..."')


def main():
    """Run retrieval against the index from part 2."""
    print("=" * 70)
    print("Part 3: Retrieval")
    print("=" * 70)
    print()

    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        collection = client.get_collection(name=COLLECTION)
    except Exception:
        print("Collection not found. Run part2_embeddings.py first.")
        return
    print(f"Connected to collection with {collection.count()} chunks")
    print()

    queries = [
        "what is a variable",
        "how can my code remember a number for later",
        "how do I bake sourdough",
    ]

    for query in queries:
        results = semantic_search(query, collection, model, top_k=3)
        display_results(query, results)
        if results:
            kept = filter_by_relevance(results, min_similarity=0.2)
            if kept is not None:
                print(f"  kept after threshold 0.20: {len(kept)} of {len(results)}")
            context = manage_context_window(results, max_tokens=500)
            if context:
                print(f"  context window: ~{len(context) // 4} tokens")
        print()

    print("=" * 70)
    print("Part 3 complete. Next: python part4_generation.py")
    print("=" * 70)


if __name__ == "__main__":
    main()