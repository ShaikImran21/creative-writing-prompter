"""Unit and integration tests for all five milestones and stretch features."""
import math
import pytest

from app.core import (
    NGramModel,
    clean_text,
    count_ngrams,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    mle_probabilities,
    sample_next_word,
    smoothed_distribution,
    smoothed_probability,
    tokenize_sentences,
)


# Milestone 1: Prepare a text corpus of short stories.
def test_milestone_1(tmp_path):
    # Built-in Media & Sports corpus loads, cleans, and tokenizes properly
    tokens = milestone_1(genre="sports")
    assert len(tokens) > 200
    assert all(tok == tok.lower() for tok in tokens)
    assert "stadium" in tokens

    # Raw text cleaning handles smart quotes, punctuation, and extra whitespace
    raw = "  The   Coach's  plan — “attack the rim!”  "
    assert clean_text(raw) == "the coach's plan \"attack the rim!\""
    assert milestone_1(raw) == ["the", "coach's", "plan", "attack", "the", "rim"]

    # File-based corpus loading
    story_file = tmp_path / "story.txt"
    story_file.write_text("The crowd roared. The striker scored!", encoding="utf-8")
    assert milestone_1(story_file) == ["the", "crowd", "roared", "the", "striker", "scored"]


# Milestone 2: Implement n-gram counting and maximum-likelihood probabilities.
def test_milestone_2():
    tokens = ["the", "crowd", "roared", "the", "crowd", "cheered", "the", "coach", "smiled"]
    probs = milestone_2(tokens, n=2)

    assert probs[("the",)]["crowd"] == pytest.approx(2 / 3)
    assert probs[("the",)]["coach"] == pytest.approx(1 / 3)
    assert probs[("crowd",)]["roared"] == pytest.approx(0.5)
    assert probs[("crowd",)]["cheered"] == pytest.approx(0.5)

    for dist in probs.values():
        assert sum(dist.values()) == pytest.approx(1.0)

    # Trigrams also sum to 1.0 per context
    tri_probs = mle_probabilities(tokens, n=3)
    assert tri_probs[("the", "crowd")]["roared"] == pytest.approx(0.5)
    assert tri_probs[("the", "crowd")]["cheered"] == pytest.approx(0.5)


# Milestone 3: Add add-k smoothing to handle unseen n-grams.
def test_milestone_3():
    tokens = ["the", "crowd", "roared", "the", "crowd", "cheered"]
    # Vocabulary = {"the", "crowd", "roared", "cheered"} -> V = 4
    dist = milestone_3(tokens, context=["the"], n=2, k=1.0)
    assert isinstance(dist, dict)
    assert sum(dist.values()) == pytest.approx(1.0)

    # Seen bigram ("the", "crowd") has count 2, context ("the",) has count 2 -> (2 + 1) / (2 + 4) = 0.5
    p_seen = milestone_3(tokens, context=["the"], word="crowd", n=2, k=1.0)
    assert p_seen == pytest.approx(3 / 6)

    # Unseen bigram ("the", "roared") still gets positive probability (0 + 1) / (2 + 4) = 1/6
    p_unseen_pair = milestone_3(tokens, context=["the"], word="roared", n=2, k=1.0)
    assert p_unseen_pair == pytest.approx(1 / 6)

    # Completely unseen context gives uniform distribution 1 / V
    unseen_ctx_dist = milestone_3(tokens, context=["stadium"], n=2, k=0.5)
    assert sum(unseen_ctx_dist.values()) == pytest.approx(1.0)
    assert unseen_ctx_dist["crowd"] == pytest.approx(1 / 4)

    # Out-of-vocabulary word still receives strictly positive probability
    ng_counts, ctx_counts = count_ngrams(tokens, n=2)
    p_oov = smoothed_probability(("the",), "underdog", ng_counts, ctx_counts, set(tokens), k=1.0)
    assert p_oov > 0.0


# Milestone 4: Develop a function to generate text by sampling from the model.
def test_milestone_4():
    tokens = milestone_1(genre="sports")
    generated_a = milestone_4(tokens, prompt="the crowd", length=10, n=3, k=0.5, seed=42)
    generated_b = milestone_4(tokens, prompt="the crowd", length=10, n=3, k=0.5, seed=42)

    # Seeded generation is deterministic and starts with the prompt
    assert generated_a == generated_b
    words = generated_a.split()
    assert words[:2] == ["the", "crowd"]
    assert len(words) == 12  # 2 prompt words + 10 generated words

    # Greedy vs temperature sampling works as expected
    dist = {"ball": 0.7, "whistle": 0.2, "coach": 0.1}
    assert sample_next_word(dist, strategy="greedy") == "ball"

    # Sentence and paragraph generation modes
    model = NGramModel.from_genre("media", n=3, k=0.5)
    sentence = model.generate_text("the newsroom", max_words=20, mode="sentence", seed=7)
    assert sentence.endswith(".")
    assert sentence[0].isupper()

    paragraph = model.generate_text("the newsroom", max_words=60, mode="paragraph", seed=7)
    assert paragraph.count(".") >= 2


# Milestone 5: Create a CLI interface for users to input text and get suggestions.
def test_milestone_5(capsys, tmp_path):
    # Suggest command with built-in genre and greedy strategy
    exit_code = milestone_5(["suggest", "the", "crowd", "--genre", "sports", "--strategy", "greedy", "--top-k", "3"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Prompt: the crowd" in out
    assert "1." in out and "3." in out

    # Generate command in sentence mode
    exit_code = milestone_5(["generate", "the", "newsroom", "--genre", "media", "--mode", "sentence", "--seed", "11"])
    assert exit_code == 0
    gen_out = capsys.readouterr().out.strip()
    assert gen_out.endswith(".")
    assert len(gen_out.split()) >= 4

    # Genres listing command
    exit_code = milestone_5(["genres"])
    assert exit_code == 0
    genres_out = capsys.readouterr().out
    assert "sports" in genres_out and "media" in genres_out and "commentary" in genres_out

    # Custom corpus file via --corpus
    custom = tmp_path / "custom.txt"
    custom.write_text("The striker kicked the ball. The striker celebrated the goal.", encoding="utf-8")
    exit_code = milestone_5(["suggest", "the", "striker", "--corpus", str(custom), "--strategy", "greedy", "--top-k", "2"])
    assert exit_code == 0
    custom_out = capsys.readouterr().out
    assert "kicked" in custom_out or "celebrated" in custom_out


def test_backoff_and_multiple_genres():
    for genre in ("sports", "media", "commentary", "all"):
        model = NGramModel.from_genre(genre, n=3, k=0.5)
        suggestions = model.suggest_next_words("unseenxyzcontext", top_k=3, strategy="greedy")
        assert len(suggestions) == 3
        assert all(prob > 0 for _, prob in suggestions)


def test_smoothed_distribution_sums_to_one():
    tokens = tokenize_sentences("The anchor read the script. The producer cued the camera.")
    flat = [w for s in tokens for w in s]
    ng_counts, ctx_counts = count_ngrams(flat, n=2)
    dist = smoothed_distribution(("the",), ng_counts, ctx_counts, set(flat), k=0.25)
    assert math.isclose(sum(dist.values()), 1.0, rel_tol=1e-9)


def test_standalone_story_generation():
    model = NGramModel.from_genre("sports", n=3, k=0.001)
    story = model.generate_text(prompt="", mode="story", seed=42)
    paragraphs = [p for p in story.split("\n\n") if p.strip()]
    assert len(paragraphs) >= 2
    assert all(p.endswith(".") for p in paragraphs)


def test_evaluate_model_and_cli(capsys):
    from app.core import evaluate_model

    rows = evaluate_model(genre="all", orders=(1, 2, 3), k_values=(0.01, 1.0))
    assert len(rows) == 6
    assert all(r["perplexity"] > 1.0 for r in rows)
    assert all(0.0 <= r["top1_acc"] <= r["top5_acc"] <= 1.0 for r in rows)

    exit_code = milestone_5(["evaluate", "--genre", "all"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Perplexity" in out and "Top-5 Acc" in out
