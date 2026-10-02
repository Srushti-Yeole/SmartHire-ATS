"""
Unit tests for SkillExtractor module.
"""

import pytest
from backend.skill_extractor import skill_extractor


def test_extract_skills():
    resume_snippet = (
        "Experienced software developer proficient in Python, FastAPI, Docker, and PostgreSQL. "
        "Built responsive web applications with React and TypeScript. "
        "Familiar with K8s and Postgres."
    )
    skills = skill_extractor.extract_skills(resume_snippet)

    assert "Python" in skills
    assert "FastAPI" in skills
    assert "Docker" in skills
    assert "PostgreSQL" in skills
    assert "React" in skills
    assert "TypeScript" in skills
    # Test alias resolution
    assert "Kubernetes" in skills  # Resolved from 'K8s'


def test_extract_education():
    text_phd = "Earned a PhD in Artificial Intelligence from MIT."
    tier, entries = skill_extractor.extract_education(text_phd)
    assert tier == "PhD / Doctorate"

    text_ms = "Holds a Master of Science in Computer Science."
    tier, entries = skill_extractor.extract_education(text_ms)
    assert tier == "Master's Degree"

    text_bs = "Graduated with a Bachelor of Technology in Electronics."
    tier, entries = skill_extractor.extract_education(text_bs)
    assert tier == "Bachelor's Degree"


def test_extract_experience_years():
    text_explicit = "Senior Developer with 7+ years of experience in distributed systems."
    years = skill_extractor.extract_experience_years(text_explicit)
    assert years == 7.0

    text_dates = (
        "Staff Engineer | CloudScale Inc. (2019 - 2023)\n"
        "Software Engineer | Acme Corp (2015 - 2019)"
    )
    years_calc = skill_extractor.extract_experience_years(text_dates)
    assert years_calc >= 7.0


def test_extract_certifications():
    text_certs = (
        "Certifications:\n"
        "- AWS Certified Solutions Architect\n"
        "- Certified Kubernetes Administrator (CKA)\n"
        "- PMP"
    )
    certs = skill_extractor.extract_certifications(text_certs)
    assert any("AWS Certified" in c for c in certs)
    assert any("Kubernetes" in c for c in certs)
