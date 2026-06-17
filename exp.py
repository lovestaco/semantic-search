"""
Keyword search vs. semantic search, side by side.

Demonstrates two claims from blog.md:
  1. "chocolate milk" and "milk chocolate" share the same words but mean
     different things -- keyword search can't tell them apart, embeddings can.
  2. "comfy shoes for standing all day" should match a doc about nursing
     clogs / anti-fatigue insoles even though they share almost no words.
"""

import re

from sentence_transformers import SentenceTransformer, util

CORPUS = [
    ("milk_chocolate", "Milk chocolate is a sweet snack bar made from cocoa solids and milk, often eaten in squares."),
    ("chocolate_milk", "Chocolate milk is a cold drink made by stirring chocolate syrup or powder into a glass of milk."),
    ("comfy_shoes", "Nursing clogs and anti-fatigue insoles are built for people who stand on their feet all day at work."),
    ("quantum", "Quantum mechanics describes how tiny particles like electrons behave in strange, probabilistic ways."),
    ("puppy", "Puppies are baby dogs that need lots of training, exercise, and patience."),
    ("football_uk", "In London, football means the sport played with a round ball and no hands, also called soccer."),
    ("football_us", "In Dallas, football means the American game with touchdowns, helmets, and forward passes."),
]

QUERIES = [
    "chocolate milk",
    "milk chocolate",
    "comfy shoes for standing all day",
]

STOPWORDS = {
    "a", "an", "the", "is", "are", "and", "or", "of", "to", "in", "on", "at",
    "for", "with", "who", "their", "by", "into", "also", "called", "means",
}


def tokenize(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS}


def keyword_score(query, doc_text):
    q, d = tokenize(query), tokenize(doc_text)
    return len(q & d)


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_ids = [doc_id for doc_id, _ in CORPUS]
    doc_texts = [text for _, text in CORPUS]
    doc_embeddings = model.encode(doc_texts, normalize_embeddings=True)

    for query in QUERIES:
        print(f'\nQuery: "{query}"')
        print("-" * 60)

        query_embedding = model.encode(query, normalize_embeddings=True)
        semantic_scores = util.cos_sim(query_embedding, doc_embeddings)[0].tolist()

        rows = []
        for doc_id, text, sem_score in zip(doc_ids, doc_texts, semantic_scores):
            kw_score = keyword_score(query, text)
            rows.append((doc_id, kw_score, sem_score))

        print(f"{'doc':<16}{'keyword overlap':<18}{'semantic similarity':<20}")
        for doc_id, kw_score, sem_score in sorted(rows, key=lambda r: -r[2]):
            print(f"{doc_id:<16}{kw_score:<18}{sem_score:<20.3f}")

    print("\nDirect pairwise similarity (no corpus, just the two phrases):")
    a, b = model.encode(["chocolate milk", "milk chocolate"], normalize_embeddings=True)
    print(f'  "chocolate milk" <-> "milk chocolate": {util.cos_sim(a, b).item():.3f}')

    dog, puppy, quantum = model.encode(["dog", "puppy", "quantum mechanics"], normalize_embeddings=True)
    print(f'  "dog" <-> "puppy": {util.cos_sim(dog, puppy).item():.3f}')
    print(f'  "dog" <-> "quantum mechanics": {util.cos_sim(dog, quantum).item():.3f}')


if __name__ == "__main__":
    main()
