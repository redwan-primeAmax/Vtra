"""
Glossary লোডার — project_root/glossary/*.txt থেকে term লোড করে।
প্রতি লাইনে একটি term। খালি লাইন এবং '#' দিয়ে শুরু হওয়া লাইন স্কিপ।
"""
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger("dubbing.glossary")

# project root = Vtra-main/ ; dubbing/ এর parent
DEFAULT_GLOSSARY_DIR = Path(__file__).resolve().parent.parent / "glossary"


def _read_terms_from_file(path: Path) -> List[str]:
    terms = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            terms.append(line)
    return terms


def load_glossary(glossary_dir: Path = DEFAULT_GLOSSARY_DIR) -> List[str]:
    """glossary/ ফোল্ডারের সব .txt থেকে term লোড (dedup, order preserved)।"""
    if not glossary_dir.exists():
        logger.warning(f"⚠️ Glossary ফোল্ডার নেই: {glossary_dir}")
        return []

    files = sorted(glossary_dir.glob("*.txt"))
    if not files:
        logger.warning(f"⚠️ কোনো .txt ফাইল নেই: {glossary_dir}")
        return []

    seen, merged = set(), []
    for file in files:
        terms = _read_terms_from_file(file)
        for term in terms:
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(term)
        logger.info(f"📖 Glossary লোড: {file.name} ({len(terms)} term)")

    logger.info(f"✅ মোট {len(merged)}টি glossary term ({len(files)} ফাইল)")
    return merged