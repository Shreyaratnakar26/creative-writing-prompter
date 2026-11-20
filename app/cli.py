
"""Command-line interface for Creative Writing Prompter."""

import argparse

from app.core import (
    milestone_1,
    build_ngram_model,
    suggest_next_words,
    generate_text
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="creative-writing-prompter",
        description="Get context-aware next-word suggestions for creative writing."
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True
    )

    sub.add_parser(
        "hello",
        help="Check that everything is working"
    )

    suggest_parser = sub.add_parser(
        "suggest",
        help="Suggest the next words for your writing"
    )

    suggest_parser.add_argument(
        "text",
        nargs="+",
        help="Enter your sentence or phrase"
    )

    suggest_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of suggestions (default: 5)"
    )

    generate_parser = sub.add_parser(
        "generate",
        help="Generate text from your starting phrase"
    )

    generate_parser.add_argument(
        "text",
        nargs="+",
        help="Enter your starting phrase"
    )

    generate_parser.add_argument(
        "--words",
        type=int,
        default=15,
        help="Number of words to generate"
    )

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Creative Writing Prompter is ready.")
        return

    corpus = milestone_1()
    model = build_ngram_model(corpus, n=2)

    if args.command == "suggest":
        if args.limit < 1:
            parser.error("--limit must be at least 1")

        seed = " ".join(args.text)

        suggestions = suggest_next_words(
            model,
            seed,
            limit=args.limit
        )

        print("\nNext-word suggestions:")
        for word, probability in suggestions:
            print(f"{word}: {probability:.4f}")

    elif args.command == "generate":
        if args.words < 0:
            parser.error("--words cannot be negative")

        seed = " ".join(args.text)

        result = generate_text(
            model,
            seed,
            max_words=args.words
        )

        print("\nGenerated text:")
        print(result)


if __name__ == "__main__":
    main()
