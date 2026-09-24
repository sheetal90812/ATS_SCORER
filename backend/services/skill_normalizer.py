import re
from typing import List


SKILL_ALIASES = {
    "nlp": "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "natural language processing (nlp)": "Natural Language Processing",

    "llm": "Large Language Models",
    "llms": "Large Language Models",
    "large language model": "Large Language Models",
    "large language models": "Large Language Models",
    "large language models (llms)": "Large Language Models",

    "rag": "Retrieval-Augmented Generation",
    "retrieval augmented generation": "Retrieval-Augmented Generation",
    "retrieval-augmented generation": "Retrieval-Augmented Generation",

    "genai": "Generative AI",
    "gen ai": "Generative AI",
    "generative ai": "Generative AI",

    "ml": "Machine Learning",
    "machine learning": "Machine Learning",

    "dl": "Deep Learning",
    "deep learning": "Deep Learning",

    "cv": "Computer Vision",
    "computer vision": "Computer Vision",

    "api": "API",
    "apis": "API",
}


def _clean_skill(skill: str) -> str:
    """Clean whitespace and formatting from a skill."""
    skill = str(skill).strip()

    # Normalize repeated whitespace
    skill = re.sub(r"\s+", " ", skill)

    return skill


def normalize_skill(skill: str) -> str:
    """Convert a skill/alias into its canonical form."""

    skill = _clean_skill(skill)

    if not skill:
        return ""

    lookup_key = skill.lower()

    return SKILL_ALIASES.get(lookup_key, skill)


def normalize_skills(skills: List[str]) -> List[str]:
    """Normalize skills and remove duplicates."""

    normalized = []

    for skill in skills:
        canonical = normalize_skill(skill)

        if canonical and canonical not in normalized:
            normalized.append(canonical)

    return normalized

GENERIC_JD_TERMS = {
    "experience",
    "knowledge",
    "understanding",
    "ability",
    "responsibilities",
    "requirements",
    "candidate",
    "communication",
    "problem-solving",
    "teamwork",
    "team",
    "skills",
    "qualification",
    "qualifications",
    "job",
    "role",
    "position",
    "intern",
    "internship",
    "ai engineer intern",
    "basic understanding",
    "working knowledge",
    "ai applications",
}


def filter_jd_skills(skills: List[str]) -> List[str]:
    """Remove generic JD language that is not a concrete skill."""

    filtered = []

    for skill in skills:
        cleaned = _clean_skill(skill)

        if not cleaned:
            continue

        if cleaned.lower() in GENERIC_JD_TERMS:
            continue

        if cleaned not in filtered:
            filtered.append(cleaned)

    return filtered  