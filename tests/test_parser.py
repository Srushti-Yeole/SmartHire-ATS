"""
Unit tests for DocumentParser module.
"""

import os
import pytest
from backend.parser import document_parser


def test_extract_contact_info():
    sample_text = (
        "John Doe\n"
        "Email: john.doe@techcorp.com\n"
        "Phone: (123) 456-7890\n"
        "LinkedIn: https://www.linkedin.com/in/johndoe-eng\n"
        "GitHub: https://github.com/johndoe-code\n"
    )
    contact = document_parser.extract_contact_info(sample_text)

    assert contact["email"] == "john.doe@techcorp.com"
    assert contact["phone"] == "(123) 456-7890"
    assert "linkedin.com/in/johndoe-eng" in contact["linkedin"]
    assert "github.com/johndoe-code" in contact["github"]


def test_extract_candidate_name():
    sample_text = (
        "Alexander Hamilton\n"
        "Senior Software Engineer\n"
        "Email: alex@example.com\n"
    )
    name = document_parser.extract_candidate_name(sample_text)
    assert name == "Alexander Hamilton"


def test_segment_sections():
    sample_resume = (
        "Alice Walker\n"
        "PROFESSIONAL SUMMARY\n"
        "Experienced Cloud Engineer with 5 years in AWS.\n\n"
        "TECHNICAL SKILLS\n"
        "Python, Docker, Kubernetes, Terraform\n\n"
        "WORK EXPERIENCE\n"
        "Lead DevOps Engineer | Acorn Tech (2020 - Present)\n"
        "- Automated deployment pipelines.\n\n"
        "EDUCATION\n"
        "Bachelor of Science in Computer Science | MIT\n"
    )
    sections = document_parser.segment_sections(sample_resume)

    assert "summary" in sections
    assert "skills" in sections
    assert "experience" in sections
    assert "education" in sections
    assert "Python, Docker, Kubernetes" in sections["skills"]


def test_parse_job_description():
    jd_text = (
        "Job Title: Lead Data Scientist\n"
        "Experience Required: 6+ years of experience in machine learning.\n"
        "Education: Master's Degree in Computer Science or Mathematics.\n"
        "We are looking for an experienced ML engineer to lead generative AI research."
    )
    parsed = document_parser.parse_job_description(jd_text)

    assert "Lead Data Scientist" in parsed["title"]
    assert parsed["experience_required"] == 6.0
    assert parsed["min_education"] == "Master's Degree"


def test_extract_text_from_sample_files():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample_resumes")
    txt_path = os.path.join(data_dir, "Alex_Chen_Senior_ML_Engineer.txt")
    pdf_path = os.path.join(data_dir, "Alex_Chen_Senior_ML_Engineer.pdf")
    docx_path = os.path.join(data_dir, "Alex_Chen_Senior_ML_Engineer.docx")

    if os.path.exists(txt_path):
        txt_content = document_parser.extract_text(txt_path)
        assert "Alex Chen" in txt_content
        assert "Machine Learning" in txt_content

    if os.path.exists(pdf_path):
        pdf_content = document_parser.extract_text(pdf_path)
        assert len(pdf_content) > 100

    if os.path.exists(docx_path):
        docx_content = document_parser.extract_text(docx_path)
        assert len(docx_content) > 100
