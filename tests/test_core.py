
from app.core import (
    DEFAULT_CORPUS,
    tokenize,
    prepare_corpus,
    milestone_1,
    build_ngram_model,
    mle_probability,
    add_k_probability,
    suggest_next_words,
    generate_text   
)


# MILESTONE 1 TESTS

def test_tokenize():
    result = tokenize("The Little Girl was Happy!")
    assert result == ["the", "little", "girl", "was", "happy"]


def test_tokenize_removes_punctuation():
    result = tokenize("Hello, world! How are you?")
    assert result == ["hello", "world", "how", "are", "you"]


def test_prepare_corpus():
    stories = [
        "The little bird flew away.",
        "A beautiful flower grew."
    ]

    result = prepare_corpus(stories)

    assert len(result) == 2
    assert result[0] == ["the", "little", "bird", "flew", "away"]
    assert result[1] == ["a", "beautiful", "flower", "grew"]


def test_milestone_1():
    result = milestone_1()

    assert len(result) >= 3
    assert all(len(story) > 0 for story in result)


# MILESTONE 2 TESTS

def test_ngram_counting():
    corpus = [
        ["the", "cat", "sat"],
        ["the", "cat", "ate"]
    ]

    model = build_ngram_model(corpus, n=2)

    assert model["counts"][("the", "cat")] == 2
    assert model["counts"][("cat", "sat")] == 1
    assert model["counts"][("cat", "ate")] == 1


def test_mle_probability():
    corpus = [
        ["the", "cat", "sat"],
        ["the", "cat", "ate"]
    ]

    model = build_ngram_model(corpus, n=2)

    probability = mle_probability(
        model["counts"],
        model["context_counts"],
        ("cat", "sat")
    )

    assert probability == 0.5


# MILESTONE 3 TESTS

def test_add_k_smoothing():
    corpus = [
        ["the", "cat", "sat"],
        ["the", "cat", "ate"]
    ]

    model = build_ngram_model(corpus, n=2)

    probability = add_k_probability(
        model["counts"],
        model["context_counts"],
        ("cat", "ran"),
        vocabulary_size=5,
        k=1
    )

    assert probability > 0


def test_add_k_seen_ngram():
    corpus = [
        ["the", "cat", "sat"],
        ["the", "cat", "ate"]
    ]

    model = build_ngram_model(corpus, n=2)

    probability = add_k_probability(
        model["counts"],
        model["context_counts"],
        ("cat", "sat"),
        vocabulary_size=5,
        k=1
    )

    assert probability == 2 / 7


# MILESTONE 4 TEST

def test_generate_text():
    corpus = [
        ["the", "cat", "sat", "down"],
        ["the", "cat", "ate", "food"],
        ["the", "dog", "sat", "outside"]
    ]

    model = build_ngram_model(corpus, n=2)

    result = generate_text(
        model,
        seed="the cat",
        max_words=5
    )

    assert isinstance(result, str)
    assert len(result.split()) > 2


def test_suggest_next_words():
    from app.core import suggest_next_words

    corpus = [
        ["the", "cat", "sat"],
        ["the", "cat", "ate"],
        ["the", "dog", "ran"]
    ]

    model = build_ngram_model(corpus, n=2)

    suggestions = suggest_next_words(
        model,
        "the cat",
        limit=5
    )

    assert len(suggestions) > 0
    assert all(isinstance(word, str) for word, probability in suggestions)
    assert all(probability > 0 for word, probability in suggestions)

