
import re
import random
from collections import Counter


# ==========================================
# MILESTONE 1: TEXT CORPUS AND TOKENIZATION
# ==========================================

DEFAULT_CORPUS = [
    "Once upon a time, a little girl lived in a beautiful village. She loved exploring the forest.",
    "The brave young boy walked through the dark forest. He discovered a magical castle.",
    "A small bird sat on a tree and sang a beautiful song. The flowers danced in the gentle wind.",
    "The old wizard lived near the mountains. He helped everyone in the kingdom.",
    "One sunny morning, a curious rabbit went on an adventure through the green fields."
]


def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def prepare_corpus(stories):
    return [
        tokens
        for story in stories
        if (tokens := tokenize(story))
    ]


def milestone_1():
    return prepare_corpus(DEFAULT_CORPUS)


# ==========================================
# MILESTONE 2: N-GRAM COUNTING AND MLE
# ==========================================

def build_ngram_model(corpus, n=2):
    if n < 1:
        raise ValueError("n must be at least 1")

    counts = Counter()
    context_counts = Counter()

    for story in corpus:
        for i in range(len(story) - n + 1):
            ngram = tuple(story[i:i + n])
            context = ngram[:-1]

            counts[ngram] += 1
            context_counts[context] += 1

    return {
        "counts": counts,
        "context_counts": context_counts,
        "n": n
    }


def mle_probability(counts, context_counts, ngram):
    ngram = tuple(ngram)
    context = ngram[:-1]

    if context_counts[context] == 0:
        return 0.0

    return counts[ngram] / context_counts[context]


def milestone_2():
    corpus = milestone_1()
    return build_ngram_model(corpus, n=2)


# ==========================================
# MILESTONE 3: ADD-K SMOOTHING
# ==========================================

def add_k_probability(
    counts,
    context_counts,
    ngram,
    vocabulary_size,
    k=1
):
    if vocabulary_size <= 0:
        return 0.0

    if k < 0:
        raise ValueError("k cannot be negative")

    ngram = tuple(ngram)
    context = ngram[:-1]

    numerator = counts[ngram] + k
    denominator = context_counts[context] + (k * vocabulary_size)

    if denominator == 0:
        return 0.0

    return numerator / denominator


def milestone_3():
    model = milestone_2()

    vocabulary = {
        word
        for story in milestone_1()
        for word in story
    }

    return {
        "counts": model["counts"],
        "context_counts": model["context_counts"],
        "vocabulary_size": len(vocabulary)
    }


def suggest_next_words(model, seed, limit=5, k=1):
    if limit < 1:
        raise ValueError("limit must be at least 1")

    tokens = tokenize(seed)
    n = model["n"]

    if not tokens:
        return []

    context = tuple(tokens[-(n - 1):]) if n > 1 else ()

    counts = model["counts"]
    context_counts = model["context_counts"]

    vocabulary = sorted({
        word
        for ngram in counts
        for word in ngram
    })

    suggestions = []

    for word in vocabulary:
        ngram = context + (word,)

        probability = add_k_probability(
            counts,
            context_counts,
            ngram,
            len(vocabulary),
            k
        )

        suggestions.append((word, probability))

    suggestions.sort(key=lambda item: (-item[1], item[0]))

    return suggestions[:limit]



# ==========================================
# MILESTONE 4: TEXT GENERATION BY SAMPLING
# ==========================================

def generate_text(model, seed, max_words=20, k=1):
    if max_words < 0:
        raise ValueError("max_words cannot be negative")

    tokens = tokenize(seed)
    generated = tokens.copy()

    counts = model["counts"]
    context_counts = model["context_counts"]
    n = model["n"]

    vocabulary = sorted({
        word
        for ngram in counts
        for word in ngram
    })

    if not vocabulary:
        return " ".join(generated)

    for _ in range(max_words):

        if n == 1:
            context = ()
        else:
            context = tuple(generated[-(n - 1):])

        candidates = []
        weights = []

        for word in vocabulary:
            ngram = context + (word,)

            probability = add_k_probability(
                counts,
                context_counts,
                ngram,
                len(vocabulary),
                k
            )

            candidates.append(word)
            weights.append(probability)

        if sum(weights) == 0:
            break

        next_word = random.choices(
            candidates,
            weights=weights,
            k=1
        )[0]

        generated.append(next_word)

    return " ".join(generated)


def milestone_4():
    model = milestone_2()

    return generate_text(
        model,
        seed="once upon a time",
        max_words=15
    )


# ==========================================
# MILESTONE 5: COMMAND-LINE INTERFACE
# ==========================================

def milestone_5():
    return "Command-line interface is implemented in app/cli.py"
    