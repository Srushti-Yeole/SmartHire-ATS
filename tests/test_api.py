"""
Integration tests for FastAPI REST endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "sentence-transformers" in data["nlp_engine"]


def test_parse_job_description_endpoint():
    payload = {
        "jd_text": (
            "Position: Senior Backend Engineer\n"
            "Experience Required: 5+ years\n"
            "Education: Bachelor's Degree\n"
            "Requirements: Strong knowledge of Python, FastAPI, Docker, and PostgreSQL."
        ),
        "jd_title": "Senior Backend Engineer",
        "experience_required": 5.0,
    }
    response = client.post("/api/v1/parse/job-description", data=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Senior Backend Engineer"
    assert data["experience_required"] == 5.0
    assert "Python" in data["skills"]
    assert "Docker" in data["skills"]


def test_evaluate_single_endpoint_with_text():
    payload = {
        "resume_text": (
            "David Miller\n"
            "Email: david.miller@example.com | Phone: (555) 123-4567\n"
            "PROFESSIONAL SUMMARY\n"
            "Software engineer with 5 years experience in Python, FastAPI, and Docker.\n"
            "TECHNICAL SKILLS\n"
            "Python, FastAPI, Docker, PostgreSQL, Git\n"
            "EDUCATION\n"
            "Bachelor of Science in Computer Science\n"
            "CERTIFICATIONS\n"
            "AWS Certified Solutions Architect\n"
        ),
        "job_description_text": (
            "Looking for a Python Backend Developer.\n"
            "Experience: 4+ years.\n"
            "Must know Python, FastAPI, Docker, and PostgreSQL."
        ),
        "jd_title": "Python Developer",
        "experience_required": 4.0,
    }
    response = client.post("/api/v1/evaluate/single", data=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["candidate"]["name"] == "David Miller"
    assert data["candidate"]["email"] == "david.miller@example.com"
    assert data["score_breakdown"]["ats_score"] > 70.0
    assert "Python" in data["skill_gap"]["matched_skills"]
    assert "FastAPI" in data["skill_gap"]["matched_skills"]


def test_model_benchmark_endpoint():
    response = client.get("/api/v1/models/benchmark")
    assert response.status_code == 200
    data = response.json()
    assert "Sentence Transformers" in data["selected_method"]
    assert len(data["benchmarks"]) >= 2
