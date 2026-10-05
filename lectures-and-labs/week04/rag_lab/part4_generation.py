"""
Part 4: Generation -- RAG Lab

In this part, you will:
1. Connect to a hosted model through an OpenAI-compatible API
2. Build a grounded prompt from retrieved chunks
3. Complete the pipeline: retrieve, augment, generate
4. Return the answer together with the documents it came from

Run it as:   python part4_generation.py

Estimated time: 30 minutes
"""

import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

COLLECTION = "cs_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-3.5-flash-lite"

GROUNDING_RULES = (
    "Answer the question using ONLY the context below. "
    "If the context does not contain the answer, reply exactly: "
    "I don't know - the provided context does not cover this. "
    "Finish your answer with the document you used, in square brackets, "
    "like [source: introduction_to_programming.txt]."
)


def llm_settings():
    """Return (api_key, base_url, model) from .env, with the defaults above."""
    load_dotenv()
    return (
        os.getenv("LLM_API_KEY"),
        os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        os.getenv("LLM_MODEL", DEFAULT_MODEL),
    )


def initialize_llm():
    """
    Initialize the client for the hosted model.

    Returns:
        OpenAI client object, or None when no key is set
    """
    api_key, base_url, model = llm_settings()

    if not api_key:
        return None

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    return client


def build_rag_prompt(query, context_chunks):
    """
    Build a grounded prompt from the retrieved chunks.
    """
    context_parts = []

    for text, source in context_chunks:
        context_parts.append(
            f"[source: {source}]\n{text}"
        )

    context = "\n\n".join(context_parts)

    prompt = (
        f"{GROUNDING_RULES}\n\n"
        f"CONTEXT:\n"
        f"{context}\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER:"
    )

    return prompt


def call_llm(client, prompt, max_tokens=500):
    """
    Call the hosted model with the given prompt.
    """
    _, _, model = llm_settings()

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


def rag_query(question, collection, embedding_model, llm_client, top_k=3):
    """
    Complete RAG pipeline: retrieve, augment, generate.
    """
    # Step 1: Retrieve
    query_embedding = embedding_model.encode(question)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    texts = results["documents"][0]
    sources = [
        metadata["source"]
        for metadata in results["metadatas"][0]
    ]

    # Step 2: Augment
    context_chunks = list(zip(texts, sources))
    prompt = build_rag_prompt(question, context_chunks)

    # Step 3: Generate
    answer = call_llm(llm_client, prompt) if llm_client else None

    # Remove duplicate sources while preserving order
    unique_sources = []
    for source in sources:
        if source not in unique_sources:
            unique_sources.append(source)

    # Step 4: Return
    return {
        "answer": answer,
        "sources": unique_sources,
        "context_used": texts,
    }


def test_rag_system():
    """Ask the questions DIY 5 names, and show the answer with its source."""
    print("=" * 70)
    print("Part 4: Generation")
    print("=" * 70)
    print()

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path="./chroma_db")

    try:
        collection = client.get_collection(name=COLLECTION)
        print(f"Found {collection.count()} chunks in the index")
    except Exception:
        print("Index not found. Run part2_embeddings.py first.")
        return

    llm_client = initialize_llm()

    if llm_client is None:
        print("No LLM_API_KEY in .env - retrieval will run, generation is skipped.")
        print("Copy .env.example to .env and add a free key (the README says where).")

    print()

    questions = [
        "What is a variable?",
        "How do linked lists work?",
        "How do I bake sourdough?",
    ]

    for question in questions:
        try:
            result = rag_query(
                question,
                collection,
                embedding_model,
                llm_client,
                top_k=3
            )
        except Exception as e:
            print(f"Error processing question: {e}")
            print("Check your implementation and try again")
            print()
            continue

        if not result:
            print(f"Q: {question}")
            print("   rag_query() returned nothing - check your implementation")
            print()
            continue

        print(f"Q: {question}")
        print(
            f"A: {result.get('answer') or '(no key set - retrieval only)'}"
        )

        sources = result.get("sources") or []
        print(
            f"   retrieved from: "
            f"{', '.join(sources) if sources else 'nothing'}"
        )
        print()

    print("=" * 70)
    print("Part 4 complete. Next: python part5_experiments.py")
    print("=" * 70)


def interactive_mode():
    """Ask your own questions."""
    print("\n" + "=" * 70)
    print("Interactive mode - type a question, or 'quit'")
    print("=" * 70)

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name=COLLECTION)
    llm_client = initialize_llm()

    while True:
        question = input("\nQ: ").strip()

        if question.lower() in ("quit", "exit", "q"):
            break

        if not question:
            continue

        try:
            result = rag_query(
                question,
                collection,
                embedding_model,
                llm_client
            )

            if result:
                print(f"A: {result['answer']}")
                print(
                    f"   retrieved from: "
                    f"{', '.join(result.get('sources') or [])}"
                )

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    test_rag_system()

    # Uncomment for interactive mode:
    # interactive_mode()