"""Core n-gram language model and creative writing prompter engine.

Implements all five milestones:
1. Corpus preparation (cleaning & tokenization)
2. N-gram counting & maximum-likelihood probabilities
3. Add-k smoothing for unseen n-grams
4. Probabilistic text sampling & generation (greedy, weighted, temperature)
5. CLI integration helpers
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
import random
import re
from typing import Sequence

from app.corpora import list_genres, load_genre_text

START_TOKEN = "<s>"
END_TOKEN = "</s>"
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_WORD_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?")


# --- Milestone 1: Prepare a text corpus of short stories.
def clean_text(text: str, lowercase: bool = True) -> str:
    """Normalize typography, whitespace, and casing in raw story text."""
    normalized = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2014", " ")
        .replace("\u2013", " ")
    )
    if lowercase:
        normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    """Clean text and return a flat list of word tokens."""
    cleaned = clean_text(text, lowercase=lowercase)
    return _WORD_TOKEN_RE.findall(cleaned)


def tokenize_sentences(text: str, lowercase: bool = True) -> list[list[str]]:
    """Split text into cleaned sentences, each represented as a list of word tokens."""
    cleaned = clean_text(text, lowercase=lowercase)
    sentences: list[list[str]] = []
    for chunk in _SENTENCE_SPLIT_RE.split(cleaned):
        words = _WORD_TOKEN_RE.findall(chunk)
        if words:
            sentences.append(words)
    return sentences


def prepare_corpus(
    source: str | Path | Sequence[str] | None = None,
    genre: str = "all",
    lowercase: bool = True,
) -> list[str]:
    """Load, clean, and tokenize a corpus from a file path, raw text, list of strings, or genre."""
    if source is None:
        raw_text = load_genre_text(genre)
    elif isinstance(source, Path):
        raw_text = source.read_text(encoding="utf-8")
    elif isinstance(source, str):
        possible_path = Path(source)
        if "\n" not in source and len(source) < 260 and possible_path.is_file():
            raw_text = possible_path.read_text(encoding="utf-8")
        else:
            raw_text = source
    else:
        raw_text = "\n".join(str(item) for item in source)

    return tokenize(raw_text, lowercase=lowercase)


def milestone_1(
    source: str | Path | Sequence[str] | None = None,
    genre: str = "all",
) -> list[str]:
    """Milestone 1: Prepare and tokenize a text corpus of short stories."""
    return prepare_corpus(source=source, genre=genre)


# --- Milestone 2: Implement n-gram counting and maximum-likelihood probabilities.
def count_ngrams(
    tokens: Sequence[str],
    n: int = 2,
) -> tuple[dict[tuple[str, ...], Counter[str]], Counter[tuple[str, ...]]]:
    """Count n-grams as (context -> Counter(next_word)) and context totals.

    For n=1 (unigrams), context is the empty tuple ().
    For n>=2, context is the preceding (n - 1) tokens.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    ngram_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    context_counts: Counter[tuple[str, ...]] = Counter()

    if len(tokens) < n:
        return dict(ngram_counts), context_counts

    for idx in range(len(tokens) - n + 1):
        context = tuple(tokens[idx : idx + n - 1])
        next_word = tokens[idx + n - 1]
        ngram_counts[context][next_word] += 1
        context_counts[context] += 1

    return dict(ngram_counts), context_counts


def mle_probabilities(
    tokens: Sequence[str],
    n: int = 2,
) -> dict[tuple[str, ...], dict[str, float]]:
    """Compute maximum-likelihood probabilities P(w | context) = c(context, w) / c(context)."""
    ngram_counts, context_counts = count_ngrams(tokens, n=n)
    probs: dict[tuple[str, ...], dict[str, float]] = {}
    for context, next_counts in ngram_counts.items():
        total = context_counts[context]
        probs[context] = {word: count / total for word, count in next_counts.items()}
    return probs


def milestone_2(
    tokens: Sequence[str],
    n: int = 2,
) -> dict[tuple[str, ...], dict[str, float]]:
    """Milestone 2: Compute n-gram counts and maximum-likelihood probabilities."""
    return mle_probabilities(tokens, n=n)


# --- Milestone 3: Add add-k smoothing to handle unseen n-grams.
def smoothed_probability(
    context: Sequence[str],
    word: str,
    ngram_counts: dict[tuple[str, ...], Counter[str]],
    context_counts: Counter[tuple[str, ...]],
    vocab: set[str] | Sequence[str],
    k: float = 1.0,
) -> float:
    """Compute add-k smoothed probability P_k(word | context).

    P_k(w | ctx) = (count(ctx, w) + k) / (count(ctx) + k * V)
    If `word` is outside `vocab`, V is expanded by 1 so the unseen word still receives
    a strictly positive smoothed probability.
    """
    if k <= 0:
        raise ValueError("Smoothing parameter k must be > 0")
    vocab_set = set(vocab)
    if not vocab_set:
        raise ValueError("Vocabulary must not be empty")

    ctx = tuple(context)
    vocab_size = len(vocab_set) if word in vocab_set else len(vocab_set) + 1
    word_count = ngram_counts.get(ctx, Counter()).get(word, 0)
    ctx_total = context_counts.get(ctx, 0)
    return (word_count + k) / (ctx_total + k * vocab_size)


def smoothed_distribution(
    context: Sequence[str],
    ngram_counts: dict[tuple[str, ...], Counter[str]],
    context_counts: Counter[tuple[str, ...]],
    vocab: set[str] | Sequence[str],
    k: float = 1.0,
) -> dict[str, float]:
    """Return the full add-k smoothed probability distribution over `vocab` (sums to 1.0)."""
    if k <= 0:
        raise ValueError("Smoothing parameter k must be > 0")
    vocab_set = sorted(set(vocab))
    if not vocab_set:
        raise ValueError("Vocabulary must not be empty")

    ctx = tuple(context)
    vocab_size = len(vocab_set)
    ctx_counts = ngram_counts.get(ctx, Counter())
    denom = context_counts.get(ctx, 0) + k * vocab_size
    return {word: (ctx_counts.get(word, 0) + k) / denom for word in vocab_set}


def milestone_3(
    tokens: Sequence[str],
    context: Sequence[str],
    word: str | None = None,
    n: int = 2,
    k: float = 1.0,
) -> float | dict[str, float]:
    """Milestone 3: Compute add-k smoothed probability for a word or full distribution."""
    ngram_counts, context_counts = count_ngrams(tokens, n=n)
    vocab = set(tokens)
    ctx = tuple(context[-(n - 1) :]) if n > 1 else ()
    if word is not None:
        return smoothed_probability(ctx, word, ngram_counts, context_counts, vocab, k=k)
    return smoothed_distribution(ctx, ngram_counts, context_counts, vocab, k=k)


# --- Milestone 4: Develop a function to generate text by sampling from the model.
def _apply_temperature(distribution: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    """Rescale a probability distribution with temperature > 0."""
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")
    if not distribution:
        raise ValueError("Distribution must not be empty")

    if math.isclose(temperature, 1.0):
        total = sum(distribution.values())
        return {w: p / total for w, p in distribution.items()}

    # Use log-space scaling to prevent underflow for low temperatures
    log_weights = {
        word: math.log(max(prob, 1e-15)) / temperature
        for word, prob in distribution.items()
    }
    max_log = max(log_weights.values())
    exp_weights = {word: math.exp(lw - max_log) for word, lw in log_weights.items()}
    total = sum(exp_weights.values())
    return {word: w / total for word, w in exp_weights.items()}


def sample_next_word(
    distribution: dict[str, float],
    strategy: str = "temperature",
    temperature: float = 1.0,
    rng: random.Random | None = None,
) -> str:
    """Select the next word from a probability distribution using greedy or weighted sampling."""
    if not distribution:
        raise ValueError("Distribution must not be empty")

    strategy_norm = strategy.strip().lower()
    if strategy_norm == "greedy":
        return max(distribution.items(), key=lambda item: (item[1], item[0]))[0]

    if strategy_norm in ("sample", "weighted"):
        adjusted = _apply_temperature(distribution, temperature=1.0)
    elif strategy_norm == "temperature":
        adjusted = _apply_temperature(distribution, temperature=temperature)
    else:
        raise ValueError(f"Unknown sampling strategy '{strategy}'. Use 'greedy', 'sample', or 'temperature'.")

    generator = rng or random.Random()
    words = list(adjusted.keys())
    weights = list(adjusted.values())
    return generator.choices(words, weights=weights, k=1)[0]


class NGramModel:
    """Multi-order n-gram language model with add-k smoothing and backoff."""

    def __init__(self, n: int = 3, k: float = 0.001):
        if n < 1:
            raise ValueError("n must be >= 1")
        if k <= 0:
            raise ValueError("Smoothing parameter k must be > 0")
        self.n = n
        self.k = k
        self.vocab: set[str] = set()
        self.orders_ngram_counts: dict[int, dict[tuple[str, ...], Counter[str]]] = {}
        self.orders_context_counts: dict[int, Counter[tuple[str, ...]]] = {}
        self.sentence_starters: Counter[str] = Counter()

    def fit(self, tokens: Sequence[str], sentences: Sequence[Sequence[str]] | None = None) -> "NGramModel":
        """Train n-gram counts for orders 1..n from tokens (and optional sentence lists)."""
        token_list = list(tokens)
        if not token_list:
            raise ValueError("Cannot train NGramModel on an empty token list")

        self.vocab = set(token_list)
        self.sentence_starters = Counter()

        if sentences:
            for sent in sentences:
                if sent:
                    self.sentence_starters[sent[0]] += 1
            for order in range(1, self.n + 1):
                agg_ngrams: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
                agg_contexts: Counter[tuple[str, ...]] = Counter()
                for sent in sentences:
                    padded = [START_TOKEN] * (order - 1) + list(sent) + [END_TOKEN]
                    ng_counts, ctx_counts = count_ngrams(padded, n=order)
                    for ctx, next_map in ng_counts.items():
                        agg_ngrams[ctx].update(next_map)
                    agg_contexts.update(ctx_counts)
                self.orders_ngram_counts[order] = dict(agg_ngrams)
                self.orders_context_counts[order] = agg_contexts
        else:
            self.sentence_starters[token_list[0]] += 1
            for order in range(1, self.n + 1):
                ng_counts, ctx_counts = count_ngrams(token_list, n=order)
                self.orders_ngram_counts[order] = ng_counts
                self.orders_context_counts[order] = ctx_counts

        return self

    @classmethod
    def from_text(cls, text: str, n: int = 3, k: float = 0.001) -> "NGramModel":
        """Build and fit an NGramModel directly from raw text."""
        sentences = tokenize_sentences(text)
        tokens = [tok for sent in sentences for tok in sent]
        model = cls(n=n, k=k)
        return model.fit(tokens, sentences=sentences)

    @classmethod
    def from_genre(cls, genre: str = "all", n: int = 3, k: float = 0.001) -> "NGramModel":
        """Build and fit an NGramModel from one of the built-in Media & Sports genres."""
        raw_text = load_genre_text(genre)
        return cls.from_text(raw_text, n=n, k=k)

    @classmethod
    def from_file(cls, path: str | Path, n: int = 3, k: float = 0.001) -> "NGramModel":
        """Build and fit an NGramModel from a local text file."""
        raw_text = Path(path).read_text(encoding="utf-8")
        return cls.from_text(raw_text, n=n, k=k)

    def _effective_order_and_context(
        self,
        context_tokens: Sequence[str],
        use_backoff: bool = True,
    ) -> tuple[int, tuple[str, ...]]:
        """Pick the highest order (up to self.n) that has observed the context (or fall back)."""
        if not use_backoff:
            ctx = tuple(context_tokens[-(self.n - 1) :]) if self.n > 1 else ()
            return self.n, ctx

        max_order = min(self.n, len(context_tokens) + 1)
        for order in range(max_order, 1, -1):
            ctx = tuple(context_tokens[-(order - 1) :])
            if self.orders_context_counts.get(order, Counter()).get(ctx, 0) > 0:
                return order, ctx
        return 1, ()

    def next_word_distribution(
        self,
        context_tokens: Sequence[str],
        use_backoff: bool = True,
        include_end_token: bool = False,
    ) -> dict[str, float]:
        """Return the add-k smoothed next-word probability distribution for a context."""
        if not self.vocab:
            raise RuntimeError("Model has not been fitted yet")

        order, ctx = self._effective_order_and_context(context_tokens, use_backoff=use_backoff)
        ng_counts = self.orders_ngram_counts[order]
        ctx_counts = self.orders_context_counts[order]

        candidate_vocab = set(self.vocab)
        if include_end_token:
            candidate_vocab.add(END_TOKEN)

        return smoothed_distribution(ctx, ng_counts, ctx_counts, candidate_vocab, k=self.k)

    def suggest_next_words(
        self,
        prompt: str,
        top_k: int = 5,
        strategy: str = "temperature",
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> list[tuple[str, float]]:
        """Return up to `top_k` context-aware next-word suggestions with probabilities."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        prompt_tokens = tokenize(prompt)
        dist = self.next_word_distribution(prompt_tokens, use_backoff=True, include_end_token=False)
        limit = min(top_k, len(dist))

        strategy_norm = strategy.strip().lower()
        if strategy_norm == "greedy":
            ranked = sorted(dist.items(), key=lambda item: (-item[1], item[0]))
            return ranked[:limit]

        rng = random.Random(seed)
        remaining = dict(dist)
        chosen: list[tuple[str, float]] = []
        for _ in range(limit):
            word = sample_next_word(remaining, strategy=strategy_norm, temperature=temperature, rng=rng)
            chosen.append((word, dist[word]))
            remaining.pop(word, None)
            if not remaining:
                break
        return chosen

    def _pick_sentence_starter(self, strategy: str, temperature: float, rng: random.Random) -> str:
        if self.sentence_starters:
            total = sum(self.sentence_starters.values())
            starter_dist = {w: c / total for w, c in self.sentence_starters.items()}
        else:
            starter_dist = self.next_word_distribution([], use_backoff=True)
        return sample_next_word(starter_dist, strategy=strategy, temperature=temperature, rng=rng)

    def generate_text(
        self,
        prompt: str = "",
        max_words: int = 20,
        strategy: str = "temperature",
        temperature: float = 0.7,
        mode: str = "words",
        seed: int | None = None,
    ) -> str:
        """Generate text in 'words', 'sentence', 'paragraph', or 'story' mode."""
        if max_words < 1:
            raise ValueError("max_words must be >= 1")

        mode_norm = mode.strip().lower()
        if mode_norm not in ("words", "sentence", "paragraph", "story"):
            raise ValueError("mode must be one of: 'words', 'sentence', 'paragraph', 'story'")

        rng = random.Random(seed)
        prompt_tokens = tokenize(prompt)
        generated: list[str] = list(prompt_tokens)

        if mode_norm == "words":
            for _ in range(max_words):
                dist = self.next_word_distribution(generated, use_backoff=True, include_end_token=False)
                next_word = sample_next_word(dist, strategy=strategy, temperature=temperature, rng=rng)
                generated.append(next_word)
            return " ".join(generated)

        if mode_norm == "sentence":
            num_paragraphs, sents_per_para, words_budget = 1, 1, max_words
        elif mode_norm == "paragraph":
            num_paragraphs, sents_per_para, words_budget = 1, 3, max(max_words, 75)
        else:  # story
            num_paragraphs, sents_per_para, words_budget = 3, 3, max(max_words, 200)

        paragraphs_out: list[str] = []
        words_generated = 0
        used_prompt = False

        for _ in range(num_paragraphs):
            sentences_out: list[str] = []
            for _ in range(sents_per_para):
                if not used_prompt and prompt_tokens:
                    current_tokens = list(prompt_tokens)
                    used_prompt = True
                else:
                    first_word = self._pick_sentence_starter(strategy, temperature, rng)
                    current_tokens = [first_word]
                    words_generated += 1

                min_sentence_words = max(len(current_tokens) + 5, 8)
                while words_generated < words_budget:
                    allow_end = len(current_tokens) >= min_sentence_words
                    dist = self.next_word_distribution(
                        current_tokens,
                        use_backoff=True,
                        include_end_token=allow_end,
                    )
                    next_word = sample_next_word(dist, strategy=strategy, temperature=temperature, rng=rng)
                    if next_word == END_TOKEN:
                        break
                    current_tokens.append(next_word)
                    words_generated += 1
                    if len(current_tokens) >= 24:
                        break

                if current_tokens:
                    sentence_str = " ".join(current_tokens)
                    sentence_str = sentence_str[0].upper() + sentence_str[1:] + "."
                    sentences_out.append(sentence_str)

                if words_generated >= words_budget:
                    break

            if sentences_out:
                paragraphs_out.append(" ".join(sentences_out))
            if words_generated >= words_budget:
                break

        return "\n\n".join(paragraphs_out)


def milestone_4(
    tokens: Sequence[str],
    prompt: str = "",
    length: int = 15,
    n: int = 2,
    k: float = 0.5,
    strategy: str = "temperature",
    temperature: float = 1.0,
    seed: int | None = None,
) -> str:
    """Milestone 4: Generate text by sampling from the smoothed n-gram model."""
    model = NGramModel(n=n, k=k).fit(tokens)
    return model.generate_text(
        prompt=prompt,
        max_words=length,
        strategy=strategy,
        temperature=temperature,
        mode="words",
        seed=seed,
    )


# --- Milestone 5: Create a CLI interface for users to input text and get suggestions.
def milestone_5(argv: list[str] | None = None) -> int:
    """Milestone 5: Run the CLI interface for user input and next-word suggestions."""
    from app.cli import main

    return main(argv)


def corpus_summary() -> dict[str, dict[str, int]]:
    """Return word and vocabulary statistics for every built-in genre corpus."""
    summary: dict[str, dict[str, int]] = {}
    for genre in list_genres():
        text = load_genre_text(genre)
        tokens = tokenize(text)
        sentences = tokenize_sentences(text)
        summary[genre] = {
            "sentences": len(sentences),
            "words": len(tokens),
            "vocabulary": len(set(tokens)),
        }
    return summary


def evaluate_model(
    genre: str = "all",
    orders: Sequence[int] = (1, 2, 3),
    k_values: Sequence[float] = (0.001, 0.01, 0.1, 1.0),
    train_ratio: float = 0.8,
) -> list[dict[str, float | int]]:
    """Evaluate n-gram orders and add-k values on a deterministic train/test sentence split.

    Returns a list of dicts with order `n`, smoothing `k`, held-out perplexity,
    top-1 accuracy, and top-5 accuracy.
    """
    raw_text = load_genre_text(genre)
    sentences = tokenize_sentences(raw_text)
    if len(sentences) < 2:
        raise ValueError("Need at least 2 sentences to run train/test evaluation")

    split_idx = max(1, min(len(sentences) - 1, int(len(sentences) * train_ratio)))
    train_sents = sentences[:split_idx]
    test_sents = sentences[split_idx:]
    train_tokens = [tok for sent in train_sents for tok in sent]

    results: list[dict[str, float | int]] = []
    for n in orders:
        for k in k_values:
            model = NGramModel(n=n, k=k).fit(train_tokens, sentences=train_sents)
            log_prob_sum = 0.0
            total_preds = 0
            top1_hits = 0
            top5_hits = 0

            for sent in test_sents:
                for idx, target in enumerate(sent):
                    context = sent[:idx]
                    dist = model.next_word_distribution(context, use_backoff=True, include_end_token=False)
                    # Handle OOV target words cleanly using smoothed fallback
                    prob = dist.get(target, k / (len(model.vocab) * k + 1.0))
                    log_prob_sum += math.log(max(prob, 1e-12))

                    ranked = sorted(dist.items(), key=lambda item: (-item[1], item[0]))
                    top5 = [w for w, _ in ranked[:5]]
                    if top5 and top5[0] == target:
                        top1_hits += 1
                    if target in top5:
                        top5_hits += 1
                    total_preds += 1

            perplexity = math.exp(-log_prob_sum / max(total_preds, 1))
            results.append(
                {
                    "n": n,
                    "k": k,
                    "test_tokens": total_preds,
                    "perplexity": round(perplexity, 2),
                    "top1_acc": round(top1_hits / max(total_preds, 1), 4),
                    "top5_acc": round(top5_hits / max(total_preds, 1), 4),
                }
            )
    return results
