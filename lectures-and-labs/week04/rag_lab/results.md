# RAG Lab - results

**Name:** _[Your name]_

**Date:** _[Date]_

---

## Part 1: chunking and embeddings (DIY 1 and 2)

- Documents loaded: **5**

- Chunks produced at 200 words: **14 (shortest 72 words, longest 200 words)**

- Vector dimensionality: **384**

- Did the query vector have the same dimensionality as the chunks? **Yes**

_What did the overlap check show?_

The overlap check showed that chunk 1 contains words that also appear in chunk 0, confirming that the 40-word overlap is working correctly.

---

## Part 2: retrieval (DIY 3 and 4)

**"what is a variable"** - top hit, score and source: **0.42, `introduction_to_programming.txt`** - "Introduction to Programming Programming is the..."

**"how can my code remember a number for later"** - top hit, score and source: **0.31, `introduction_to_programming.txt`** - "Introduction to Programming Programming is the..."

_Same meaning, different words: did both queries find the same chunk?_

Yes. Both queries found the same top chunk from `introduction_to_programming.txt`, although the scores were different (0.42 vs 0.31).

**"how do I bake sourdough"** - what came back, and with what scores:

- `algorithms_overview.txt` - **0.09**
- `web_development_intro.txt` - **0.09**
- `algorithms_overview.txt` - **0.05**

None passed the **0.20** relevance threshold.

_One sentence on what a retrieval system does when nothing is relevant:_

When nothing is relevant, the similarity scores remain below the relevance threshold and no chunks are kept for the context.

---

## Part 3: grounding (DIY 5)

**Q:** What is a variable?

**A:** A variable is a named storage location that holds a value that can be changed during a program's execution. [source: introduction_to_programming.txt]

**Source it cited:** `introduction_to_programming.txt`

**Q:** How do I bake sourdough?

**A:** I don't know - the provided context does not cover this. (Yes, it declined.)

---

## Part 4: chunk size (DIY 6)

| Chunk size | Right chunk found? | Noise | Notes |
|------------|--------------------|-------|-------|
| 50 words | No for sourdough | Higher | 66 chunks; smaller chunks give fine-grained retrieval but can fragment information |
| 200 words | No for sourdough | Moderate | 14 chunks; good balance between relevant information and context |
| 800 words | Yes for programming query | Higher | Larger chunks provide more surrounding context but can include irrelevant information |

_One sentence on what goes wrong at each extreme:_

Very small chunks can fragment useful information, while very large chunks can include too much irrelevant context.

---

## Part 5: long context versus retrieval (DIY 7)

```text
Whole corpus size: approximately 1,600 tokens

Question asked: How do I bake sourdough?

  RAG answer: I don't know - the provided context does not cover this.

  Whole-corpus answer: I don't know - the provided context does not cover this.

Which was better? No difference for this question.

At what corpus size would this flip?
As the corpus grows, retrieval would become more useful because sending the entire corpus would use more context and include more irrelevant information.