import re
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional

from backend.utils.file_utils import log_warning
from backend.core.config import SENTENCE_TRANSFORMER_MODEL
from backend.utils.matching import fuzzy_match_keywords


ZIP_CODE_PATTERN = r'\b\d{5}(?:-\d{4})?\b'

STREET_ADDRESS_PATTERN = (
    r'\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+'
    r'(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir|Way|Place|Pl)\b'
)


def _tier_score(n: float, tiers: list) -> float:
    for threshold, pts in tiers:
        if n >= threshold:
            return pts

    return 0.0


# Location/privacy detection
def detect_location_info(text: str, nlp: spacy.Language) -> Dict:
    locations = []

    # method01: spacy NER
    doc = nlp(text)

    for ent in doc.ents:
        if ent.label_ in ['GPE', 'LOC']:
            locations.append({
                'text': ent.text,
                'type': ent.label_.lower(),
                'start': ent.start_char
            })

    # method02: street address regex
    for match in re.finditer(STREET_ADDRESS_PATTERN, text, re.IGNORECASE):
        locations.append({
            'text': match.group(),
            'type': 'address',
            'start': match.start()
        })

    # method03: ZIP/PIN code regex
    for match in re.finditer(ZIP_CODE_PATTERN, text):
        locations.append({
            'text': match.group(),
            'type': 'zip',
            'start': match.start()
        })

    has_address = any(loc['type'] == 'address' for loc in locations)
    has_zip = any(loc['type'] == 'zip' for loc in locations)

    if has_address and has_zip:
        privacy_risk, penalty = 'high', 5.0
    elif has_address or has_zip:
        privacy_risk, penalty = 'high', 4.0
    elif len(locations) > 3:
        privacy_risk, penalty = 'medium', 3.0
    elif locations:
        privacy_risk, penalty = 'low', 2.0
    else:
        privacy_risk, penalty = 'none', 0.0

    recommendations = []

    if not locations:
        recommendations.append(" No privacy concerns detected.")

    if has_address:
        recommendations.append(
            " Remove full street addresses — ATS systems don't need this and it's a privacy risk."
        )

    if has_zip:
        recommendations.append(
            " Remove zip codes — this level of location detail is unnecessary."
        )

    if privacy_risk in ('low', 'medium') and not has_address and not has_zip:
        recommendations.append(
            " Consider reducing location mentions. 'City, State' in the contact header is sufficient."
        )

    return {
        'location_found': len(locations) > 0,
        'detected_locations': locations,
        'privacy_risk': privacy_risk,
        'recommendations': recommendations,
        'penalty_applied': penalty,
    }


def _calculate_semantic_similarity(
    skill: str,
    text: str,
    embedder: SentenceTransformer
) -> float:
    """Calculate cosine similarity between a skill and one evidence snippet."""

    if not skill or not text:
        return 0.0

    try:
        skill_vec = embedder.encode(
            skill,
            convert_to_tensor=False
        )

        text_vec = embedder.encode(
            text,
            convert_to_tensor=False
        )

        skill_norm = np.linalg.norm(skill_vec)
        text_norm = np.linalg.norm(text_vec)

        if skill_norm == 0 or text_norm == 0:
            return 0.0

        similarity = np.dot(skill_vec, text_vec) / (
            skill_norm * text_norm
        )

        return float(max(0.0, min(1.0, similarity)))

    except Exception as e:
        log_warning(
            f"Similarity error for '{skill}': {e}",
            context='ats_scorer'
        )
        return 0.0


def _skill_matches(
    skill: str,
    text: str,
    embedder: SentenceTransformer,
    threshold: float = 0.65
):
    """
    Check whether a skill is supported by one piece of evidence.

    Matching order:
    1. Exact phrase match
    2. All meaningful skill terms present
    3. Semantic similarity

    The semantic threshold is intentionally conservative so that
    unrelated skills are not marked as validated just because they
    are semantically close to a large paragraph.
    """

    if not skill or not text:
        return False, 0.0

    skill_clean = skill.strip().lower()
    text_clean = text.strip().lower()

    if not skill_clean or not text_clean:
        return False, 0.0
    
        # Prevent "Transformers" from matching "Sentence Transformers".
        if skill_clean == "transformers":
            text_without_sentence_transformers = re.sub(
                r'(?<!\w)sentence transformers(?!\w)',
                '',
                text_clean
            )

            if not re.search(
                r'(?<!\w)transformers(?!\w)',
                text_without_sentence_transformers
            ):
                return False, 0.0

    # 1. Exact phrase match
    if re.search(rf'(?<!\w){re.escape(skill_clean)}(?!\w)', text_clean):
        return True, 1.0
    
    # Alias match
    aliases = SKILL_ALIASES.get(skill_clean, [])

    for alias in aliases:
        if alias in text_clean:
            return True, 1.0

    # 2. Multi-word skill match
    skill_terms = [
        term
        for term in re.findall(r'\b[\w+#.-]+\b', skill_clean)
        if len(term) > 2
    ]

    if len(skill_terms) > 1 and len(skill_clean.split()) > 1 and all(
        re.search(rf'(?<!\w){re.escape(term)}(?!\w)', text_clean)
        for term in skill_terms
    ):
        return True, 1.0

    # Prevent partial-word false positives
    if len(skill_clean.split()) == 1:
        if not re.search(
            rf'(?<!\w){re.escape(skill_clean)}(?!\w)',
            text_clean
        ):
            return False, 0.0    

    # 3. Semantic similarity
    return False, 0.0
    

def _split_evidence_text(text: str) -> List[str]:
    """
    Split evidence into smaller meaningful snippets.

    This prevents one long project paragraph from producing a broad
    semantic match for an unrelated skill.
    """

    if not text:
        return []

    # Split on sentence boundaries and common bullet separators.
    pieces = re.split(
        r'(?<=[.!?])\s+|[\n•●▪◦]+',
        text
    )

    cleaned = []

    for piece in pieces:
        piece = re.sub(r'\s+', ' ', piece).strip()

        if piece:
            cleaned.append(piece)

    return cleaned

SKILL_ALIASES = {
    "vector db": [
        "vector db",
        "vector database",
        "vector databases",
        "vector similarity search",
    ],
}

# Skill validation
def validate_skills_with_projects(
    skills: List[str],
    projects: List[Dict],
    experience_entries: List[Dict],
    embedder: SentenceTransformer,
    threshold: float = 0.65,
) -> Dict:
    """
    Validate listed skills against actual project and experience evidence.

    A skill is validated when it has evidence in:
    - a project title,
    - a project description,
    - project technologies,
    - an experience title/company/description.

    Evidence is evaluated in smaller snippets rather than one giant
    combined paragraph to reduce semantic false positives.
    """
    

    if not skills:
        return {
            'validated_skills': [],
            'unvalidated_skills': [],
            'validation_percentage': 0.0,
            'skill_project_mapping': {},
            'validation_score': 0.0,
        }

    experience_evidence = []

    for experience in experience_entries:
        if not isinstance(experience, dict):
            continue

        experience_evidence.append({
            'title': experience.get('job_title', ''),
            'company': experience.get('company', ''),
            'description': experience.get('description', ''),
        })

    validated_skills = []
    unvalidated_skills = []
    skill_project_mapping = {}

    for skill in skills:
        matching_projects = []
        max_similarity = 0.0

        # ---------------------------------------------------------
        # PROJECT EVIDENCE
        # ---------------------------------------------------------
        for project in projects:
            if not isinstance(project, dict):
                continue

            project_title = str(
                project.get('title', '') or ''
            ).strip()

            project_description = str(
                project.get('description', '') or ''
            ).strip()

            technologies = project.get('technologies', []) or []

            if not isinstance(technologies, list):
                technologies = [str(technologies)]

            technology_text = " ".join(
                str(item).strip()
                for item in technologies
                if item
            )

            # Technology lists are especially strong evidence.
            evidence_snippets = []

            if project_title:
                evidence_snippets.append(project_title)

            if project_description:
                evidence_snippets.extend(
                    _split_evidence_text(project_description)
                )

            if technology_text:
                # Each technology is treated as its own evidence item.
                for item in technologies:
                    item_text = str(item).strip()

                    if not item_text:
                        continue

                    # Exact technology match
                    if skill.strip().lower() == item_text.lower():
                        matching_projects.append(
                            project_title or 'Untitled Project'
                        )
                        project_matched = True
                        break

                    evidence_snippets.append(item_text)

            project_matched = False

            for snippet in evidence_snippets:

                # Do not treat "Sentence Transformers"
                # as evidence for the separate skill "Transformers".
                if (
                    skill.strip().lower() == "transformers"
                    and "sentence transformers" in snippet.strip().lower()
                ):
                    continue

                matched, similarity = _skill_matches(
                    skill,
                    snippet,
                    embedder,
                    threshold
                )
                max_similarity = max(
                    max_similarity,
                    similarity
                )

                if matched:
                    project_matched = True
                    break

            if project_matched:
                project_name = (
                    project_title or 'Untitled Project'
                )

                if project_name not in matching_projects:
                    matching_projects.append(project_name)

        # ---------------------------------------------------------
        # EXPERIENCE EVIDENCE
        # ---------------------------------------------------------
        for experience in experience_evidence:
            experience_title = str(
                experience.get('title', '') or ''
            ).strip()

            experience_company = str(
                experience.get('company', '') or ''
            ).strip()

            experience_description = str(
                experience.get('description', '') or ''
            ).strip()

            experience_snippets = []

            if experience_title:
                experience_snippets.append(experience_title)

            if experience_company:
                experience_snippets.append(experience_company)

            if experience_description:
                experience_snippets.extend(
                    _split_evidence_text(
                        experience_description
                    )
                )

            if not experience_snippets:
                continue

            experience_matched = False

            for snippet in experience_snippets:
                matched, similarity = _skill_matches(
                    skill,
                    snippet,
                    embedder,
                    threshold
                )

                max_similarity = max(
                    max_similarity,
                    similarity
                )

                if matched:
                    experience_matched = True
                    break

            if experience_matched:
                if experience_title and experience_company:
                    experience_label = (
                        f"{experience_title} @ {experience_company}"
                    )
                elif experience_title:
                    experience_label = experience_title
                elif experience_company:
                    experience_label = experience_company
                else:
                    experience_label = "Experience"

                if experience_label not in matching_projects:
                    matching_projects.append(
                        experience_label
                    )

        # ---------------------------------------------------------
        # FINAL CLASSIFICATION
        # ---------------------------------------------------------
        if matching_projects:
            validated_skills.append({
                'skill': skill,
                'projects': matching_projects,
                'similarity': max_similarity
            })

            skill_project_mapping[skill] = matching_projects

        else:
            unvalidated_skills.append(skill)
            skill_project_mapping[skill] = []

    validation_percentage = (
        len(validated_skills) / len(skills)
        if skills
        else 0.0
    )

    validation_score = validation_percentage * 15.0

    return {
        'validated_skills': validated_skills,
        'unvalidated_skills': unvalidated_skills,
        'validation_percentage': validation_percentage,
        'skill_project_mapping': skill_project_mapping,
        'validation_score': validation_score,
    }


# 01: formatting score
def _calc_formatting_score(
    parsed_resume: Dict,
    text: str
) -> float:

    score = 0.0

    exp_entries = [
        e for e in parsed_resume.get('experience', [])
        if isinstance(e, dict)
    ]

    edu_entries = [
        e for e in parsed_resume.get('education', [])
        if isinstance(e, dict)
    ]

    skills = parsed_resume.get('skills', [])
    summary = parsed_resume.get('professional_summary', '')

    proj_entries = [
        p for p in parsed_resume.get('projects', [])
        if isinstance(p, dict)
    ]

    if exp_entries and any(
        e.get('job_title') or e.get('description')
        for e in exp_entries
    ):
        score += 3.0

    if edu_entries:
        score += 2.0

    if len(skills) >= 3:
        score += 2.0

    if len(summary) > 30:
        score += 1.5

    if proj_entries:
        score += 1.5

    bullet_count = sum(
        1
        for line in text.split('\n')
        if re.match(r'^\s*[•\-\*\◦]', line)
        or re.match(r'^\s*\d+\.', line)
    )

    score += _tier_score(
        bullet_count,
        [
            (15, 5.0),
            (10, 4.0),
            (5, 3.0),
            (3, 2.0),
            (1, 1.0)
        ]
    )

    filled = sum(
        1
        for has_it in [
            bool(exp_entries),
            bool(edu_entries),
            bool(skills),
            bool(summary.strip()),
            bool(proj_entries)
        ]
        if has_it
    )

    score += _tier_score(
        filled,
        [
            (4, 5.0),
            (3, 4.0),
            (2, 3.0),
            (1, 2.0)
        ]
    )

    return min(20.0, max(0.0, score))


# 02: keyword score
def _calc_keywords_score(
    resume_keywords: List[str],
    skills: List[str],
    jd_keywords: Optional[List[str]] = None,
) -> float:

    score = 0.0

    score += _tier_score(
        len(resume_keywords),
        [
            (20, 10.0),
            (15, 8.0),
            (10, 6.0),
            (5, 4.0),
            (3, 2.0)
        ]
    )

    score += _tier_score(
        len(skills),
        [
            (15, 10.0),
            (10, 8.0),
            (7, 6.0),
            (5, 4.0),
            (3, 2.0)
        ]
    )

    if jd_keywords:
        all_resume_terms = list(
            set(resume_keywords + skills)
        )

        fuzzy_result = fuzzy_match_keywords(
            all_resume_terms,
            jd_keywords,
            threshold=80
        )

        match_pct = (
            len(fuzzy_result['matched']) / len(jd_keywords)
            if jd_keywords
            else 0
        )

        score += _tier_score(
            match_pct,
            [
                (0.7, 5.0),
                (0.5, 4.0),
                (0.3, 3.0),
                (0.2, 2.0),
                (0.1, 1.0)
            ]
        )

    elif len(resume_keywords) >= 10:
        score += 3.0

    return min(25.0, max(0.0, score))


# 03: content quality score
def _calc_content_score(
    text: str,
    action_verbs: List[str],
    grammar_results: Dict,
) -> float:

    score = 0.0

    score += _tier_score(
        len(action_verbs),
        [
            (15, 10.0),
            (10, 8.0),
            (7, 6.0),
            (5, 4.0),
            (3, 2.0)
        ]
    )

    number_patterns = [
        r'\d+%',
        r'\$\d+',
        r'\d+[kKmMbB]',
        r'\d+\s*(?:users|customers|clients|projects|hours|days|months|years)',
        r'(?:increased|decreased|improved|reduced|grew|saved)\s+(?:by\s+)?\d+',
    ]

    achievement_count = sum(
        len(re.findall(p, text, re.IGNORECASE))
        for p in number_patterns
    )

    score += _tier_score(
        achievement_count,
        [
            (10, 5.0),
            (7, 4.0),
            (5, 3.0),
            (3, 2.0),
            (1, 1.0)
        ]
    )

    grammar_penalty = grammar_results.get(
        'penalty_applied',
        0.0
    )

    score += max(
        0.0,
        10.0 - grammar_penalty / 2.0
    )

    return min(25.0, max(0.0, score))


# 04: skill validation score
def _calc_skill_validation_score(
    validation_results: Dict
) -> float:

    return min(
        15.0,
        max(
            0.0,
            validation_results.get(
                'validation_score',
                0.0
            )
        )
    )


# 05: ATS compatibility score
def _calc_ats_compatibility_score(
    text: str,
    location_results: Dict,
    parsed_resume: Dict,
) -> float:

    score = 15.0

    # deduction01
    score -= location_results.get(
        'penalty_applied',
        0.0
    )

    # deduction02
    special_chars = len(
        re.findall(
            r'[│┤├┼┴┬╔╗╚╝═║╠╣╦╩╬]',
            text
        )
    )

    if special_chars > 20:
        score -= 2.0
    elif special_chars > 10:
        score -= 1.0

    exp_entries = [
        e for e in parsed_resume.get('experience', [])
        if isinstance(e, dict)
    ]

    edu_entries = [
        e for e in parsed_resume.get('education', [])
        if isinstance(e, dict)
    ]

    skills_count = len(
        parsed_resume.get('skills', [])
    )

    exp_desc_len = sum(
        len(e.get('description', ''))
        for e in exp_entries
    )

    edu_desc_len = sum(
        len(
            (e.get('degree') or '') +
            (e.get('institution') or '')
        )
        for e in edu_entries
    )

    # deduction03
    short_sections = sum([
        bool(exp_entries) and exp_desc_len < 20,
        bool(edu_entries) and edu_desc_len < 20,
        bool(parsed_resume.get('skills')) and skills_count < 2,
    ])

    if short_sections >= 2:
        score -= 2.0
    elif short_sections >= 1:
        score -= 1.0

    if exp_entries and skills_count > 5:
        score += 1.0

    return min(15.0, max(0.0, score))


# Score aggregation and final interpretation
def calculate_overall_score(
    text: str,
    parsed_resume: Dict,
    skills: List[str],
    keywords: List[str],
    action_verbs: List[str],
    skill_validation_results: Dict,
    grammar_results: Dict,
    location_results: Dict,
    jd_keywords: Optional[List[str]] = None,
    experience_months: int = 0,
) -> Dict:

    formatting_score = _calc_formatting_score(
        parsed_resume,
        text
    )

    keywords_score = _calc_keywords_score(
        keywords,
        skills,
        jd_keywords
    )

    content_score = _calc_content_score(
        text,
        action_verbs,
        grammar_results
    )

    skill_validation_score = _calc_skill_validation_score(
        skill_validation_results
    )

    ats_compatibility_score = _calc_ats_compatibility_score(
        text,
        location_results,
        parsed_resume
    )

    COMPONENT_MAX = {
        'formatting': 20.0,
        'keywords': 25.0,
        'content': 25.0,
        'skill_validation': 15.0,
        'ats_compatibility': 15.0,
    }

    formatting_pct = (
        formatting_score /
        COMPONENT_MAX['formatting']
    ) * 100.0

    keywords_pct = (
        keywords_score /
        COMPONENT_MAX['keywords']
    ) * 100.0

    content_pct = (
        content_score /
        COMPONENT_MAX['content']
    ) * 100.0

    skill_validation_pct = (
        skill_validation_score /
        COMPONENT_MAX['skill_validation']
    ) * 100.0

    ats_compatibility_pct = (
        ats_compatibility_score /
        COMPONENT_MAX['ats_compatibility']
    ) * 100.0

    skills_keywords_pct = (
        keywords_pct * 0.6
    ) + (
        skill_validation_pct * 0.4
    )

    base_score = (
        skills_keywords_pct * 0.40 +
        content_pct * 0.30 +
        formatting_pct * 0.15 +
        ats_compatibility_pct * 0.15
    )

    penalties = {}
    bonuses = {}
    score = base_score

    if grammar_results.get(
        'penalty_applied',
        0.0
    ) > 0:

        penalties['grammar'] = grammar_results[
            'penalty_applied'
        ]

    if location_results.get(
        'penalty_applied',
        0.0
    ) > 0:

        penalties['location_privacy'] = location_results[
            'penalty_applied'
        ]

    validation_pct = skill_validation_results.get(
        'validation_percentage',
        0.0
    )

    if validation_pct >= 0.9:
        bonuses['excellent_skill_validation'] = 2.0
        score += 2.0

    elif validation_pct >= 0.8:
        bonuses['good_skill_validation'] = 1.0
        score += 1.0

    if grammar_results.get(
        'total_errors',
        0
    ) == 0:

        bonuses['perfect_grammar'] = 1.0
        score += 1.0

    if jd_keywords and len(jd_keywords) > 0:
        all_resume_terms = list(
            set((keywords or []) + (skills or []))
        )

        fuzzy_result = fuzzy_match_keywords(
            all_resume_terms,
            jd_keywords,
            threshold=80
        )

        missing_pct = (
            len(fuzzy_result['missing']) /
            len(jd_keywords)
        )

        if missing_pct > 0.7:
            penalties['missing_jd_keywords'] = 15.0
            score -= 15.0

        elif missing_pct > 0.5:
            penalties['missing_jd_keywords'] = 10.0
            score -= 10.0

        elif missing_pct > 0.3:
            penalties['missing_jd_keywords'] = 5.0
            score -= 5.0

    overall_score = min(
        100.0,
        max(0.0, score)
    )

    interpretation = _generate_score_interpretation(
        overall_score
    )

    return {
        'overall_score': round(
            overall_score,
            1
        ),
        'formatting_score': round(
            formatting_score,
            1
        ),
        'keywords_score': round(
            keywords_score,
            1
        ),
        'content_score': round(
            content_score,
            1
        ),
        'skill_validation_score': round(
            skill_validation_score,
            1
        ),
        'ats_compatibility_score': round(
            ats_compatibility_score,
            1
        ),
        'overall_interpretation': interpretation,
        'penalties': penalties,
        'bonuses': bonuses,
    }


# Overall score calculation and interpretation
def generate_strengths(
    score_results: Dict,
    skill_validation_results: Dict,
    grammar_results: Dict,
) -> List[str]:

    strengths = []

    if score_results['formatting_score'] >= 16:
        strengths.append(
            ' Well-structured with clear sections and bullet points'
        )

    if score_results['keywords_score'] >= 20:
        strengths.append(
            ' Strong keyword optimization and skills presence'
        )

    if score_results['content_score'] >= 20:
        strengths.append(
            ' Excellent use of action verbs and quantifiable achievements'
        )

    if score_results['skill_validation_score'] >= 12:
        pct = (
            skill_validation_results.get(
                'validation_percentage',
                0
            ) * 100
        )

        strengths.append(
            f' {pct:.0f}% of skills are validated by projects'
        )

    if score_results['ats_compatibility_score'] >= 13:
        strengths.append(
            ' Excellent ATS compatibility with clean formatting'
        )

    if grammar_results.get(
        'total_errors',
        0
    ) == 0:

        strengths.append(
            ' Error-free grammar and spelling'
        )

    if not strengths:
        strengths.append(
            'Your resume has potential - focus on the recommendations below'
        )

    return strengths


# Critical issues that could cause ATS rejection
def generate_critical_issues(
    score_results: Dict,
    grammar_results: Dict,
    location_results: Dict,
) -> List[str]:

    issues = []

    critical_errors = len(
        grammar_results.get(
            'critical_errors',
            []
        )
    )

    if critical_errors > 0:
        issues.append(
            f' {critical_errors} critical grammar/spelling error(s) detected'
        )

    if location_results.get(
        'privacy_risk'
    ) == 'high':

        issues.append(
            'High privacy risk: Remove detailed location information'
        )

    if score_results['formatting_score'] < 10:
        issues.append(
            ' Poor formatting: Add clear sections and bullet points'
        )

    if score_results['keywords_score'] < 12:
        issues.append(
            ' Insufficient keywords and skills'
        )

    if score_results['skill_validation_score'] < 7:
        issues.append(
            ' Most skills lack supporting evidence in projects'
        )

    return issues


# Actionable improvements to enhance ATS performance
def generate_improvements(
    score_results: Dict,
    skill_validation_results: Dict,
) -> List[str]:

    improvements = []

    if 12 <= score_results['formatting_score'] < 16:
        improvements.append(
            'Add more bullet points and improve section organization'
        )

    if 14 <= score_results['keywords_score'] < 20:
        improvements.append(
            'Include more relevant keywords and technical skills'
        )

    if 14 <= score_results['content_score'] < 20:
        improvements.append(
            'Add more quantifiable achievements and action verbs'
        )

    if 7 <= score_results['skill_validation_score'] < 12:
        unvalidated_count = len(
            skill_validation_results.get(
                'unvalidated_skills',
                []
            )
        )

        improvements.append(
            f'Validate {unvalidated_count} skill(s) by adding relevant project details'
        )

    if 9 <= score_results['ats_compatibility_score'] < 13:
        improvements.append(
            'Simplify formatting for better ATS compatibility'
        )

    return improvements


# Interpretation of overall score
def _generate_score_interpretation(
    overall_score: float
) -> str:

    if overall_score >= 90:
        return (
            'Excellent! Your resume is highly optimized for ATS systems.'
        )

    elif overall_score >= 80:
        return (
            'Great! Your resume should perform well with most ATS systems.'
        )

    elif overall_score >= 70:
        return (
            'Good! Your resume is ATS-friendly with room for minor improvements.'
        )

    elif overall_score >= 60:
        return (
            'Fair. Your resume needs some improvements to be fully ATS-compatible.'
        )

    elif overall_score >= 50:
        return (
            'Below Average. Significant improvements needed for ATS compatibility.'
        )

    else:
        return (
            'Poor. Your resume requires major revisions to pass ATS screening.'
        )