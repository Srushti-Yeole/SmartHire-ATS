"""
Unit and integration tests for Side-by-Side Candidate Comparison.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import SessionLocal, save_evaluation_record
from frontend.components import create_multi_candidate_radar_chart, render_pillar_progress_bars

client = TestClient(app)


def test_candidate_comparison_endpoint():
    db = SessionLocal()
    # Create two dummy evaluation records
    cand1_data = {
        "name": "Candidate Alpha",
        "email": "alpha@example.com",
        "experience_years": 5.0,
        "highest_education": "Master of Science",
        "skills": ["Python", "FastAPI", "Docker"],
        "certifications": ["AWS Certified"],
    }
    cand2_data = {
        "name": "Candidate Beta",
        "email": "beta@example.com",
        "experience_years": 3.0,
        "highest_education": "Bachelor of Science",
        "skills": ["Python", "SQL"],
        "certifications": [],
    }
    jd_data = {
        "title": "Backend Lead",
        "company": "Tech Corp",
        "experience_required": 4.0,
        "skills": ["Python", "FastAPI", "Docker", "Kubernetes"],
    }
    scores1 = {
        "ats_score": 85.0,
        "match_percentage": 85.0,
        "keyword_score": 90.0,
        "semantic_score": 82.0,
        "experience_score": 85.0,
        "education_score": 100.0,
        "certification_score": 70.0,
    }
    scores2 = {
        "ats_score": 68.0,
        "match_percentage": 68.0,
        "keyword_score": 60.0,
        "semantic_score": 75.0,
        "experience_score": 70.0,
        "education_score": 80.0,
        "certification_score": 50.0,
    }

    eval1 = save_evaluation_record(db, cand1_data, jd_data, scores1, {"matched_skills": ["Python", "FastAPI"], "missing_skills": ["Kubernetes"]}, {})
    eval2 = save_evaluation_record(db, cand2_data, jd_data, scores2, {"matched_skills": ["Python"], "missing_skills": ["FastAPI", "Docker"]}, {})

    # Call comparison endpoint
    payload = {"evaluation_ids": [eval1.id, eval2.id]}
    res = client.post("/api/v1/candidates/compare", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_compared"] == 2
    assert len(data["candidates"]) == 2
    # Verify descending score sort
    assert data["candidates"][0]["ats_score"] >= data["candidates"][1]["ats_score"]
    assert data["candidates"][0]["name"] == "Candidate Alpha"

    db.close()


def test_multi_candidate_radar_chart():
    candidates = [
        {
            "name": "Alice",
            "ats_score": 90.0,
            "keyword_score": 95.0,
            "semantic_score": 88.0,
            "experience_score": 90.0,
            "education_score": 100.0,
            "certification_score": 80.0,
        },
        {
            "name": "Bob",
            "ats_score": 75.0,
            "keyword_score": 70.0,
            "semantic_score": 80.0,
            "experience_score": 75.0,
            "education_score": 80.0,
            "certification_score": 60.0,
        },
    ]
    fig = create_multi_candidate_radar_chart(candidates)
    assert fig is not None
    assert len(fig.data) == 2


def test_render_pillar_progress_bars():
    breakdown = {
        "keyword_matching": 85.0,
        "semantic_similarity": 78.5,
        "experience_matching": 90.0,
        "education_matching": 100.0,
        "certification_matching": 60.0,
    }
    html = render_pillar_progress_bars(breakdown)
    assert "Keyword Matching" in html
    assert "85.0%" in html
    assert "Semantic Relevance" in html
    assert "Experience Alignment" in html
