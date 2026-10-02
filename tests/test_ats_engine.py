"""
Unit tests for ATSEngine module.
"""

import pytest
from backend.ats_engine import ats_engine


def test_keyword_matching_score():
    candidate_skills = ["Python", "FastAPI", "Docker", "PostgreSQL"]
    jd_skills = ["Python", "Docker", "AWS", "Kubernetes"]

    score, matched, missing = ats_engine.compute_keyword_score(candidate_skills, jd_skills)

    # 2 out of 4 skills matched = 50%
    assert score == 50.0
    assert "Python" in matched
    assert "Docker" in matched
    assert "AWS" in missing
    assert "Kubernetes" in missing


def test_experience_scoring():
    # Exactly meets requirement
    score_exact = ats_engine.compute_experience_score(candidate_exp=5.0, required_exp=5.0)
    assert score_exact >= 95.0

    # Exceeds requirement
    score_exceed = ats_engine.compute_experience_score(candidate_exp=7.0, required_exp=5.0)
    assert score_exceed >= 95.0

    # Shortfall
    score_short = ats_engine.compute_experience_score(candidate_exp=2.5, required_exp=5.0)
    assert score_short < 90.0


def test_education_scoring():
    # Equal degree
    score_equal = ats_engine.compute_education_score("Bachelor's Degree", "Bachelor's Degree")
    assert score_equal == 100.0

    # Exceeding degree (Master's applying for Bachelor's)
    score_higher = ats_engine.compute_education_score("Master's Degree", "Bachelor's Degree")
    assert score_higher == 100.0

    # Lower degree
    score_lower = ats_engine.compute_education_score("Diploma / Associate", "Bachelor's Degree")
    assert score_lower == 75.0


def test_ats_formula_weights():
    cand_data = {
        "skills": ["Python", "Docker", "AWS", "FastAPI"],
        "raw_text": "Experienced Python Engineer building cloud microservices with Docker and FastAPI on AWS.",
        "experience_years": 5.0,
        "highest_education": "Bachelor's Degree",
        "certifications": ["AWS Certified Solutions Architect"],
    }
    jd_data = {
        "skills": ["Python", "Docker", "AWS", "FastAPI"],
        "raw_text": "Looking for a Python Developer with Docker, AWS, and FastAPI experience.",
        "experience_required": 5.0,
        "min_education": "Bachelor's Degree",
    }

    result = ats_engine.evaluate_candidate(cand_data, jd_data)

    ats_score = result["ats_score"]
    assert 0.0 <= ats_score <= 100.0
    # Perfect alignment should yield a high score (> 85)
    assert ats_score >= 85.0
    assert "breakdown" in result
    assert result["breakdown"]["keyword_matching"] == 100.0


def test_candidate_ranking():
    candidates = [
        {"name": "Cand C", "scores": {"ats_score": 62.0, "breakdown": {"keyword_matching": 60.0}}},
        {"name": "Cand A", "scores": {"ats_score": 91.5, "breakdown": {"keyword_matching": 90.0}}},
        {"name": "Cand B", "scores": {"ats_score": 78.0, "breakdown": {"keyword_matching": 75.0}}},
    ]

    ranked = ats_engine.rank_candidates(candidates)

    assert ranked[0]["name"] == "Cand A"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["name"] == "Cand B"
    assert ranked[1]["rank"] == 2
    assert ranked[2]["name"] == "Cand C"
    assert ranked[2]["rank"] == 3
