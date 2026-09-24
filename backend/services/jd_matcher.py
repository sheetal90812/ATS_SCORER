from typing import List, Dict
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer

from backend.utils.matching import fuzzy_match_keywords
from backend.services.skill_normalizer import normalize_skill
from rapidfuzz import fuzz


def calculate_semantic_similarity(
    resume_text: str, jd_text: str, embedder: SentenceTransformer
) -> float:
    resume_emb = embedder.encode(resume_text[:5000], convert_to_tensor=False)
    jd_emb     = embedder.encode(jd_text[:5000], convert_to_tensor=False)

    similarity = np.dot(resume_emb, jd_emb) / (
        np.linalg.norm(resume_emb) * np.linalg.norm(jd_emb)
    )
    return float(np.clip(similarity, 0.0, 1.0))


def identify_matched_keywords(
    resume_keywords: List[str], jd_keywords: List[str]
) -> List[str]:
    result = fuzzy_match_keywords(resume_keywords, jd_keywords, threshold=80)
    return result['matched']


def identify_missing_keywords(
    resume_keywords: List[str], jd_keywords: List[str], top_n: int = 15
) -> List[str]:

    result = fuzzy_match_keywords(resume_keywords, jd_keywords, threshold=80)
    return result['missing'][:top_n]

def analyze_skills_gap(
    resume_skills: List[str],
    resume_keywords: List[str],
    jd_skills: List[str],
) -> List[str]:

    resume_normalized = {
        normalize_skill(skill).lower()
        for skill in (resume_skills or []) + (resume_keywords or [])
        if skill
    }

    gap = []

    for jd_skill in jd_skills:
        if not jd_skill:
            continue

        jd_norm = normalize_skill(jd_skill).lower()

        # Exact normalized match
        if jd_norm in resume_normalized:
            continue

        # Fuzzy match against resume skills
        best_score = max(
            (
                fuzz.token_sort_ratio(jd_norm, resume_skill)
                for resume_skill in resume_normalized
            ),
            default=0,
        )

        if best_score < 75:
            gap.append(normalize_skill(jd_skill))

    return sorted(set(gap))[:20]


def calculate_match_percentage(
    resume_keywords: List[str],
    jd_keywords: List[str],
    semantic_similarity: float,
) -> float:
    if not jd_keywords:
        return 0.0
    matched = identify_matched_keywords(resume_keywords, jd_keywords)
    keyword_overlap = len(matched) / len(jd_keywords)
    match_pct = (keyword_overlap * 0.6 + semantic_similarity * 0.4) * 100
    return float(np.clip(match_pct, 0.0, 100.0))


def compare_resume_with_jd(
    resume_text: str,
    resume_keywords: List[str],
    resume_skills: List[str],
    jd_text: str,
    jd_keywords: List[str],
    jd_skills: List[str],
    embedder: SentenceTransformer,
    nlp: spacy.Language,
) -> Dict:
    semantic_similarity = calculate_semantic_similarity(resume_text, jd_text, embedder)
    matched_keywords    = identify_matched_keywords(resume_keywords, jd_keywords)
    missing_keywords    = identify_missing_keywords(resume_keywords, jd_keywords)
    skills_gap = analyze_skills_gap(
        resume_skills,
        resume_keywords,
        jd_skills
    )
    match_percentage    = calculate_match_percentage(
    resume_keywords, jd_keywords, semantic_similarity
    )

    return {
        'match_percentage':    match_percentage,
        'semantic_similarity': semantic_similarity,
        'matched_keywords':    matched_keywords,
        'missing_keywords':    missing_keywords,
        'skills_gap':          skills_gap,
    }



def evaluate_jd_gaps(actual_gaps, expected_gaps):
    actual = {skill.lower() for skill in actual_gaps}
    expected = {skill.lower() for skill in expected_gaps}

    matched = actual & expected

    precision = len(matched) / len(actual) if actual else 0
    recall = len(matched) / len(expected) if expected else 1

    return {
        "matched": sorted(matched),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
    }




