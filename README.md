# Creative Writing Prompter

[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/83aneepcwv/badge.svg)](https://msrit.ask.thili.ai/b/83aneepcwv)

A Python CLI tool for aspiring writers in the **Media & Sports** domain. It trains an n-gram language model with add-k smoothing and backoff on short-story corpora to deliver context-aware next-word suggestions, interactive writing prompts, and standalone story generation directly in the terminal.

## Features & Completed Milestones

- [x] **1. Corpus preparation (`app/core.py`, `app/corpora/`)** — Text normalization (smart quotes, dashes, whitespace, lowercasing), sentence splitting, and word tokenization across built-in Media & Sports genres (`sports`, `media`, `commentary`, `all`) or any custom `.txt` file.
- [x] **2. N-gram counting & MLE probabilities (`count_ngrams`, `mle_probabilities`)** — Arbitrary-order n-gram frequency counting and maximum-likelihood estimation `P(w | context)`.
- [x] **3. Add-k smoothing (`smoothed_probability`, `smoothed_distribution`)** — Handles unseen n-grams, unseen contexts, and out-of-vocabulary words with strictly positive, normalized probability distributions.
- [x] **4. Probabilistic sampling & story generation (`sample_next_word`, `NGramModel`)** — Supports `greedy`, `sample`, and `temperature` strategies with backoff across `words`, `sentence`, `paragraph`, and multi-paragraph `story` modes.
- [x] **5. CLI interface (`app/cli.py`)** — Full `argparse` CLI with `suggest`, `generate`, `interactive`, `genres`, and `evaluate` subcommands.

## Data & Corpus Provenance

All built-in corpora are original, self-contained Media & Sports short stories authored specifically for this project and bundled inside `app/corpora/` (no external dataset downloads or API keys required):

| Corpus File | Genre | Sentences | Words | Unique Vocabulary |
| :--- | :--- | ---: | ---: | ---: |
| `app/corpora/sports.txt` | Sports | 35 | 610 | 349 |
| `app/corpora/media.txt` | Media & Journalism | 27 | 500 | 342 |
| `app/corpora/commentary.txt` | Sports Broadcasting | 25 | 483 | 332 |
| **Combined (`--genre all`)** | **All Media & Sports** | **87** | **1,593** | **~750+** |

You can also supply any local plain-text file using `--corpus path/to/stories.txt`.

## Setup & Usage

Requires **Python 3.11+** (tested on Python 3.12 in GitHub Actions). Dependencies are pinned in `requirements.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```

### CLI Commands

```bash
# List built-in genre corpora and word counts
python -m app.cli genres

# Get context-aware next-word suggestions (greedy or temperature-sampled)
python -m app.cli suggest "the crowd" --genre sports --strategy greedy --top-k 5
python -m app.cli suggest "the newsroom" --genre media --strategy temperature --temperature 0.7

# Generate a standalone multi-paragraph story (or a single sentence/paragraph)
python -m app.cli generate --genre sports --mode story --seed 42
python -m app.cli generate "in the broadcast booth" --genre commentary --mode sentence --seed 42

# Interactive story-building mode
python -m app.cli interactive --genre all --prompt "the stadium"

# Run the reproducible 80/20 baseline & smoothing evaluation
python -m app.cli evaluate --genre all
```

## Sample Outputs

**Next-word suggestions (`python -m app.cli suggest "the crowd" --genre sports --strategy greedy --top-k 5`):**

```text
Prompt: the crowd
Next-word suggestions (greedy, genre=sports):
  1. held               (p=0.4261)
  2. rose               (p=0.4261)
  3. a                  (p=0.0004)
  4. above              (p=0.0004)
  5. across             (p=0.0004)
```

**Sentence continuation (`python -m app.cli generate "in the broadcast booth" --genre commentary --mode sentence --seed 42`):**

```text
In the broadcast booth smelled of rain warm electronics and black coffee.
```

**Standalone story (`python -m app.cli generate --genre sports --mode story --seed 42`):**

```text
On her wrist and shook out her shoulders. The bowler pitched short controlled the ball high over square had led to this single night under the afternoon sun at the cricket ground. Arjun swiveled cleanly and pulled the ball softly off the glass as the bowler pitched short.

Maya controlled the ball softly off the glass as the red light flashed around the bend. The defenders early morning drills bruised shins and quiet bus rides home had led to this single night under the rain. The ball softly off the glass as the final whistle hung quarter with the visitors trailing by nine.

When the pass in stride and drained the three pointer to tie the game. Then arena exploded into cheers while the team piled around him near the baseline screen. The goalkeeper stepped off her line arms wide daring her to shoot early.
```

## Baseline, Evaluation & Limitations

Running `python -m app.cli evaluate --genre all` benchmarks the **unigram baseline (`n=1`)** against **bigram (`n=2`)** and **trigram (`n=3`)** models with backoff across four add-k smoothing values on a deterministic 80/20 sentence split (354 held-out test tokens):

| Model Order (`n`) | Smoothing (`k`) | Held-Out Perplexity | Top-1 Accuracy | Top-5 Accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 (Unigram baseline) | 0.001 | 531.09 | 9.32% | 15.82% |
| 1 (Unigram baseline) | 0.01 | 362.28 | 9.32% | 15.82% |
| **1 (Unigram baseline)** | **0.1** | **342.59** | **9.32%** | **15.82%** |
| 2 (Bigram) | 0.001 | 725.66 | 7.34% | 15.54% |
| 2 (Bigram) | 0.1 | 405.94 | 7.34% | 15.54% |
| 3 (Trigram) | 0.001 | 722.90 | 7.34% | 14.97% |
| **3 (Trigram)** | **0.1** | **409.94** | **7.34%** | **14.97%** |
| 3 (Trigram) | 1.0 | 468.87 | 7.34% | 14.97% |

### Key Findings & Where It Falls Short

- **Small-corpus sparsity vs. context quality:** On a 1,593-word corpus (~1,239 training tokens), most held-out bigrams and trigrams are unseen, so the unigram baseline achieves slightly lower held-out perplexity (342.59 vs. 405.94–409.94). However, on seen domain phrases (e.g., `"the crowd"` → `held`, `rose`), bigram/trigram models produce far more relevant suggestions than unigrams.
- **Sensitivity to `k`:** Add-k smoothing cuts trigram perplexity by ~43% (from 722.90 at `k=0.001` down to 409.94 at `k=0.1`), while overly large `k` (`k=1.0`) over-flattens the distribution on small vocabularies and increases perplexity (468.87).
- **Fixed Markov window (`n - 1` words):** Because predictions depend only on the previous 1–2 tokens, multi-sentence stories cannot track long-range plot arcs, character continuity, or grammar across clause boundaries.
- **Punctuation & casing normalization:** Tokenization strips internal commas and quotation marks to keep the vocabulary compact, so generated sentences only restore sentence-initial capitalization and trailing periods.

## Optional LLM Configuration

`app/llm.py` is included for optional OpenAI-compatible chat endpoints (Ollama, Groq, or Gemini). Copy `.env.example` to `.env` if you choose to configure it; `.env` is ignored by git.
