# Creative Writing Prompter

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/296e82g45q/badge.svg)](https://ask.thili.ai/b/296e82g45q)

A Python-based command-line application that helps writers overcome writer's block using context-aware next-word suggestions and text generation.

## Project Overview

Creative Writing Prompter uses an n-gram language model trained on a collection of short stories. It analyzes word sequences and predicts possible next words based on the supplied context.

## Features

- Text preprocessing and tokenization
- Short-story corpus preparation
- Unigram and bigram counting
- Maximum Likelihood Estimation (MLE)
- Add-k smoothing for unseen word combinations
- Context-aware next-word suggestions
- Text generation using weighted random sampling
- Interactive command-line interface

## Technologies Used

- Python 3.12
- Regular Expressions
- N-gram Language Models
- Probability and Statistical Language Modelling
- Pytest
- Git and GitHub

## Project Structure

```text
creative-writing-prompter/
│
├── app/
│   ├── __init__.py
│   ├── cli.py
│   ├── core.py
│   ├── llm.py
│   └── storage.py
│
├── tests/
│   ├── test_core.py
│   └── test_smoke.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── pytest.ini
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd creative-writing-prompter
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

Check whether the application is working:

```bash
python -m app.cli hello
```

## Usage

### Next-word suggestions

```bash
python -m app.cli suggest "once upon a time"
```

### Generate text

```bash
python -m app.cli generate "once upon a time" --words 15
```

### Customize the number of suggestions

```bash
python -m app.cli suggest "the little girl" --limit 3
```

## Testing

Run the complete test suite:

```bash
python -m pytest -v
```

**Current test result: 13 tests passed.**

## Project Milestones

- [x] Prepare a text corpus of short stories.
- [x] Implement n-gram counting and maximum-likelihood probabilities.
- [x] Add add-k smoothing.
- [x] Develop text generation by sampling.
- [x] Create a command-line interface for next-word suggestions.

## How It Works

1. Short stories are converted into lowercase tokens.
2. The n-gram model calculates word-sequence frequencies.
3. MLE calculates conditional probabilities.
4. Add-k smoothing handles unseen word combinations.
5. Weighted random sampling generates text and next-word suggestions.

## Limitations

The current implementation uses a small built-in story corpus and a bigram model. Generated text may occasionally contain grammatically incorrect or unrelated word sequences.

## Future Improvements

- Expand the training corpus.
- Implement trigram and higher-order language models.
- Improve sentence coherence.
- Add a graphical user interface.
- Support user-provided text datasets.

## Author

Developed as a practical NLP project using Python and statistical language modelling.