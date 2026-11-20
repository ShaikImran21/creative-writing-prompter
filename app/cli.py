"""Command-line entry point: `python -m app.cli --help`."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from app.corpora import list_genres
from app.core import NGramModel, corpus_summary, evaluate_model
from app.storage import Store


def _load_model(args: argparse.Namespace) -> NGramModel:
    if getattr(args, "corpus", None):
        return NGramModel.from_file(Path(args.corpus), n=args.n, k=args.k)
    return NGramModel.from_genre(genre=args.genre, n=args.n, k=args.k)


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    genre_choices = [*list_genres(), "all"]
    parser.add_argument(
        "--genre",
        choices=genre_choices,
        default="all",
        help="Built-in Media & Sports genre corpus to sample from (default: all).",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to a custom .txt corpus file (overrides --genre).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="N-gram order (default: 3).",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=0.001,
        help="Add-k smoothing parameter > 0 (default: 0.001).",
    )
    parser.add_argument(
        "--strategy",
        choices=("temperature", "greedy", "sample"),
        default="temperature",
        help="Sampling strategy (default: temperature).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature when using --strategy temperature (default: 0.7).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible output.",
    )


def _cmd_suggest(args: argparse.Namespace) -> int:
    model = _load_model(args)
    prompt = " ".join(args.prompt).strip()
    suggestions = model.suggest_next_words(
        prompt=prompt,
        top_k=args.top_k,
        strategy=args.strategy,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(f"Prompt: {prompt or '(start of story)'}")
    print(f"Next-word suggestions ({args.strategy}, genre={args.genre if not args.corpus else args.corpus}):")
    for idx, (word, prob) in enumerate(suggestions, start=1):
        print(f"  {idx}. {word:<18} (p={prob:.4f})")

    if args.save:
        store = Store()
        history = store.get("suggestions_history", [])
        history.append({"prompt": prompt, "suggestions": [w for w, _ in suggestions]})
        store.put("suggestions_history", history)
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    model = _load_model(args)
    prompt = " ".join(args.prompt).strip()
    output = model.generate_text(
        prompt=prompt,
        max_words=args.length,
        strategy=args.strategy,
        temperature=args.temperature,
        mode=args.mode,
        seed=args.seed,
    )
    print(output)
    return 0



def _cmd_evaluate(args: argparse.Namespace) -> int:
    rows = evaluate_model(genre=args.genre)
    print(f"Held-out 80/20 evaluation (genre={args.genre}, test_tokens={rows[0]['test_tokens']}):")
    print(f"{'n':>3} | {'k':>6} | {'Perplexity':>10} | {'Top-1 Acc':>9} | {'Top-5 Acc':>9}")
    print("-" * 50)
    for r in rows:
        print(
            f"{r['n']:>3} | {r['k']:>6} | {r['perplexity']:>10.2f} | "
            f"{r['top1_acc']*100:>8.2f}% | {r['top5_acc']*100:>8.2f}%"
        )
    return 0


def _cmd_genres(_args: argparse.Namespace) -> int:
    stats = corpus_summary()
    print("Available built-in genre corpora:")
    for name, info in stats.items():
        print(
            f"  - {name:<12} {info['sentences']:>3} sentences | "
            f"{info['words']:>4} words | {info['vocabulary']:>4} unique words"
        )
    print("  - all          combines all built-in corpora")
    return 0


def _cmd_interactive(args: argparse.Namespace) -> int:
    model = _load_model(args)
    story_words: list[str] = args.prompt.split() if args.prompt else []
    print("Creative Writing Prompter — Interactive Mode")
    print("Type a number to pick a suggestion, type your own word(s), or enter :quit to finish.\n")

    while True:
        current = " ".join(story_words)
        suggestions = model.suggest_next_words(
            prompt=current,
            top_k=args.top_k,
            strategy=args.strategy,
            temperature=args.temperature,
            seed=args.seed,
        )
        print(f"\nStory so far: {current or '(empty)'}")
        for idx, (word, prob) in enumerate(suggestions, start=1):
            print(f"  [{idx}] {word} ({prob:.4f})")

        try:
            user_input = input("> ").strip()
        except EOFError:
            break

        if not user_input or user_input.lower() in (":quit", ":q", "quit", "exit"):
            break
        if user_input.isdigit():
            choice = int(user_input)
            if 1 <= choice <= len(suggestions):
                story_words.append(suggestions[choice - 1][0])
                continue
        story_words.extend(user_input.split())

    print("\nFinal draft:")
    print(" ".join(story_words))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="creative-writing-prompter",
        description=(
            "Aspiring writers use this CLI tool to get context-aware next-word suggestions, "
            "helping them develop stories and overcome writer's block by sampling from a genre corpus."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("hello", help="check that everything is wired up")
    sub.add_parser("genres", help="list available built-in Media & Sports genre corpora")
    eval_parser = sub.add_parser("evaluate", help="benchmark unigram/bigram/trigram models and smoothing values")
    eval_parser.add_argument("--genre", choices=[*list_genres(), "all"], default="all", help="Genre corpus to evaluate (default: all).")

    suggest_parser = sub.add_parser("suggest", help="get next-word suggestions for a prompt")
    suggest_parser.add_argument("prompt", nargs="*", help="starting text or story fragment")
    suggest_parser.add_argument("--top-k", type=int, default=5, help="number of suggestions (default: 5)")
    suggest_parser.add_argument("--save", action="store_true", help="save suggestions to local JSON store")
    _add_model_args(suggest_parser)

    gen_parser = sub.add_parser("generate", help="generate words, a sentence, or a paragraph from a prompt")
    gen_parser.add_argument("prompt", nargs="*", help="starting text or story fragment")
    gen_parser.add_argument("--length", type=int, default=20, help="maximum number of new words (default: 20)")
    gen_parser.add_argument(
        "--mode",
        choices=("words", "sentence", "paragraph", "story"),
        default="story",
        help="generation mode (default: story)",
    )
    _add_model_args(gen_parser)

    interactive_parser = sub.add_parser("interactive", help="write a story step-by-step with live suggestions")
    interactive_parser.add_argument("--prompt", type=str, default="", help="optional starting prompt")
    interactive_parser.add_argument("--top-k", type=int, default=5, help="number of suggestions per turn")
    _add_model_args(interactive_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Creative Writing Prompter is ready. Run `python -m app.cli --help` to see commands.")
        return 0
    if args.command == "genres":
        return _cmd_genres(args)
    if args.command == "evaluate":
        return _cmd_evaluate(args)
    if args.command == "suggest":
        return _cmd_suggest(args)
    if args.command == "generate":
        return _cmd_generate(args)
    if args.command == "interactive":
        return _cmd_interactive(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
