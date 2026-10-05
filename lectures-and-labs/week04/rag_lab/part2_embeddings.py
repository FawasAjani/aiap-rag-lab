"""
Part 2: Document Processing & Embeddings -- RAG Lab

In this part, you will:
1. Load documents from the data directory
2. Split each one into overlapping chunks of WORDS
3. Generate vector embeddings
4. Store the chunks, their embeddings and the file each came from in ChromaDB

Run it as:   python part2_embeddings.py

DIY 6 rebuilds the index at other chunk sizes (each run replaces the index):
             python part2_embeddings.py --chunk-words 50
             python part2_embeddings.py --chunk-words 800

Estimated time: 30 minutes
"""

import argparse
import os

import chromadb
from sentence_transformers import SentenceTransformer

DEFAULT_CHUNK_WORDS = 200
DEFAULT_OVERLAP_WORDS = 40
COLLECTION = "cs_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_documents(data_dir="data"):
    """
    Load all .txt files from the data directory.

    Args:
        data_dir: Path to directory containing text files

    Returns:
        List of (filename, content) tuples, sorted by filename
    """
    documents = []

    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            documents.append((filename, content))

    return documents


def chunk_text(
    text,
    chunk_words=DEFAULT_CHUNK_WORDS,
    overlap_words=DEFAULT_OVERLAP_WORDS
):
    """
    Split text into overlapping chunks of roughly chunk_words WORDS.

    Args:
        text: The full document text
        chunk_words: Words per chunk (default 200)
        overlap_words: Words shared between neighbouring chunks (default 40)

    Returns:
        List of text chunks (strings)
    """
    chunks = []

    words = text.split()
    step = chunk_words - overlap_words
    start = 0

    while start < len(words):
        chunk = words[start:start + chunk_words]
        chunks.append(" ".join(chunk))

        if start + chunk_words >= len(words):
            break

        start += step

    return chunks


def generate_embeddings(chunks, model_name=EMBEDDING_MODEL):
    """
    Generate vector embeddings for text chunks.

    Args:
        chunks: List of text chunks
        model_name: Name of the sentence-transformer model

    Returns:
        numpy array of embedding vectors, one row per chunk
    """
    print(f"Loading embedding model: {model_name}...")

    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)

    return embeddings


def store_in_chromadb(chunks, embeddings, sources, collection_name=COLLECTION):
    """
    Store chunks, their embeddings and the file each came from in ChromaDB.

    Args:
        chunks: List of text chunks
        embeddings: Embedding vectors, one per chunk
        sources: List of filenames, one per chunk, in the same order
        collection_name: Name for the ChromaDB collection

    Returns:
        ChromaDB collection object
    """
    client = chromadb.PersistentClient(path="./chroma_db")

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        configuration={"hnsw": {"space": "cosine"}},
    )

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[{"source": s} for s in sources],
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    return collection


def main():
    """Run the complete Part 2 pipeline."""
    parser = argparse.ArgumentParser(
        description="Build the chunk index for the RAG lab."
    )

    parser.add_argument(
        "--chunk-words",
        type=int,
        default=DEFAULT_CHUNK_WORDS,
        help=f"words per chunk (default {DEFAULT_CHUNK_WORDS})"
    )

    parser.add_argument(
        "--overlap-words",
        type=int,
        default=None,
        help=f"words shared between neighbours (default {DEFAULT_OVERLAP_WORDS}, "
             f"or a fifth of --chunk-words when that is smaller)"
    )

    args = parser.parse_args()

    overlap = (
        args.overlap_words
        if args.overlap_words is not None
        else min(DEFAULT_OVERLAP_WORDS, args.chunk_words // 5)
    )

    print("=" * 70)
    print("Part 2: Document Processing & Embeddings")
    print("=" * 70)
    print()

    # Step 1: Load documents
    documents = load_documents("data")

    if not documents:
        print("No documents loaded. Check your load_documents() function.")
        return

    print(f"Loaded {len(documents)} documents")

    for filename, _ in documents:
        print(f"   - {filename}")

    print()

    # Step 2: Chunk documents, remembering which file each chunk came from
    all_chunks = []
    sources = []

    for filename, content in documents:
        chunks = chunk_text(
            content,
            chunk_words=args.chunk_words,
            overlap_words=overlap
        )

        all_chunks.extend(chunks)
        sources.extend([filename] * len(chunks))

    if not all_chunks:
        print("No chunks created. Check your chunk_text() function.")
        return

    lengths = [len(c.split()) for c in all_chunks]

    print(
        f"Produced {len(all_chunks)} chunks "
        f"({args.chunk_words} words, {overlap} overlap)"
    )

    print(f"  shortest: {min(lengths)} words")
    print(f"  longest:  {max(lengths)} words")

    if len(all_chunks) > 1:
        opening = " ".join(all_chunks[1].split()[:5])

        print(
            f'Overlap check: chunk 1 begins "{opening}..." -- '
            f"those words also sit inside chunk 0: "
            f"{opening in all_chunks[0]}"
        )

    print()

    # Step 3: Generate embeddings
    embeddings = generate_embeddings(all_chunks)

    if embeddings is None:
        print(
            "No embeddings generated. "
            "Check your generate_embeddings() function."
        )
        return

    print(f"Embedded {len(embeddings)} chunks")
    print(f"Vector dimensionality: {len(embeddings[0])}")

    query_vector = SentenceTransformer(
        EMBEDDING_MODEL
    ).encode("what is a variable")

    print(
        f"Query vector (first 5): "
        f"{[round(float(x), 3) for x in query_vector[:5]]}"
    )

    print(
        f"Dimensions match: "
        f"{len(query_vector) == len(embeddings[0])}"
    )

    print()

    # Step 4: Store in ChromaDB
    collection = store_in_chromadb(
        all_chunks,
        embeddings,
        sources
    )

    if collection is None:
        print(
            "Failed to create collection. "
            "Check your store_in_chromadb() function."
        )
        return

    print(
        f"Stored {collection.count()} chunks in ./chroma_db "
        f"(collection '{COLLECTION}')"
    )

    print()

    print("=" * 70)
    print("Part 2 complete. Next: python part3_retrieval.py")
    print("=" * 70)


if __name__ == "__main__":
    main()