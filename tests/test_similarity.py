"""
Unit tests for SimilarityEngine module.
"""

import pytest
from backend.similarity_engine import similarity_engine


def test_tfidf_similarity():
    text_a = "Senior Machine Learning Engineer specializing in Python, PyTorch, and NLP."
    text_b = "Looking for a Machine Learning Engineer with Python and PyTorch skills."
    sim = similarity_engine.compute_tfidf_similarity(text_a, text_b)

    assert 0.0 < sim <= 1.0
    assert sim > 0.3


def test_jaccard_similarity():
    text_a = "python docker kubernetes aws"
    text_b = "python docker react nodejs"
    sim = similarity_engine.compute_jaccard_similarity(text_a, text_b)

    # 2 common tokens (python, docker), 6 union tokens
    assert round(sim, 2) == 0.33


def test_sbert_similarity():
    text_a = "Natural Language Processing and Generative AI"
    text_b = "NLP algorithms and Large Language Models"
    sim = similarity_engine.compute_sbert_similarity(text_a, text_b)

    assert 0.0 <= sim <= 1.0
    # Semantic similarity should recognize semantic overlap
    assert sim > 0.4


def test_empty_text_similarity():
    sim = similarity_engine.calculate_similarity("", "Sample job description")
    assert sim == 0.0
