from backend.services.jd_matcher import analyze_skills_gap, evaluate_jd_gaps

resume_skills = [
    "Python",
    "Machine Learning",
    "Deep Learning",
    "Generative AI",
    "FastAPI",
    "LangChain",
    "OpenAI API",
    "Groq API",
    "PyTorch",
]

resume_keywords = [
    "Large Language Models",
    "Retrieval-Augmented Generation",
]

jd_skills = [
    "Python",
    "Machine Learning",
    "Generative AI",
    "Large Language Models",
    "Retrieval-Augmented Generation",
    "FastAPI",
    "Agentic AI",
    "AI Agents",
    "LangGraph",
    "Docker",
]


actual_gaps = analyze_skills_gap(
    resume_skills=resume_skills,
    resume_keywords=resume_keywords,
    jd_skills=jd_skills,
)

expected_gaps = {
    "Agentic AI",
    "AI Agents",
    "LangGraph",
    "Docker",
}

evaluation = evaluate_jd_gaps(
    actual_gaps=actual_gaps,
    expected_gaps=expected_gaps,
)

matched = evaluation["matched"]
precision = evaluation["precision"]
recall = evaluation["recall"]
print("\nEvaluation:")
print("Matched:", sorted(matched))
print("Precision:", round(precision, 2))
print("Recall:", round(recall, 2))

print("Actual JD gaps:")
for skill in actual_gaps:
    print("-", skill)

assert precision == 1.0
assert recall == 1.0

print("\nTest PASSED ✅")

# False-positive test
resume_skills_2 = [
    "Python",
    "Machine Learning",
    "Docker",
]

jd_skills_2 = [
    "Python",
    "Machine Learning",
    "Docker",
]

actual_gaps_2 = analyze_skills_gap(
    resume_skills=resume_skills_2,
    resume_keywords=[],
    jd_skills=jd_skills_2,
)

print("\nFalse-positive test:")
print("Actual gaps:", actual_gaps_2)

assert actual_gaps_2 == []

print("False-positive test PASSED ✅")


# Alias normalization test
resume_skills_3 = [
    "LLM",
    "RAG",
]

jd_skills_3 = [
    "Large Language Models",
    "Retrieval-Augmented Generation",
]

actual_gaps_3 = analyze_skills_gap(
    resume_skills=resume_skills_3,
    resume_keywords=[],
    jd_skills=jd_skills_3,
)

print("\nAlias normalization test:")
print("Actual gaps:", actual_gaps_3)

assert actual_gaps_3 == []

print("Alias normalization test PASSED ✅")