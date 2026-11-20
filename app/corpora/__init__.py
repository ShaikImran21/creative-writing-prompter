"""Built-in genre corpora for the Creative Writing Prompter (Media & Sports domain)."""
from pathlib import Path

CORPORA_DIR = Path(__file__).parent

AVAILABLE_GENRES = ("sports", "media", "commentary")


def list_genres() -> list[str]:
    """Return the available built-in genre names."""
    return list(AVAILABLE_GENRES)


def load_genre_text(genre: str = "all") -> str:
    """Load raw text for a specific built-in genre or all genres combined."""
    normalized = genre.strip().lower()
    if normalized == "all":
        parts = [
            (CORPORA_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()
            for name in AVAILABLE_GENRES
        ]
        return "\n\n".join(parts)

    if normalized not in AVAILABLE_GENRES:
        valid = ", ".join([*AVAILABLE_GENRES, "all"])
        raise ValueError(f"Unknown genre '{genre}'. Choose from: {valid}.")

    return (CORPORA_DIR / f"{normalized}.txt").read_text(encoding="utf-8")
