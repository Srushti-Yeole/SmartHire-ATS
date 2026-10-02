"""
Unit tests for NLPPipeline module.
"""

import pytest
from backend.nlp_pipeline import nlp_pipeline


def test_clean_text():
    raw_text = "Visit https://google.com for info!! Python & C++ are great. \n\t Contact: test@mail.com"
    cleaned = nlp_pipeline.clean_text(raw_text)

    assert "https" not in cleaned
    assert "google.com" not in cleaned
    assert "python" in cleaned
    assert "c++" in cleaned


def test_tokenize():
    text = "FastAPI and PyTorch enable scalable microservices."
    tokens = nlp_pipeline.tokenize(text)

    assert "fastapi" in tokens
    assert "pytorch" in tokens
    assert len(tokens) >= 5


def test_preprocess_tokens():
    text = "The quick brown fox is running and jumping over the lazy dogs in Python."
    tokens = nlp_pipeline.preprocess_tokens(text, remove_stops=True, lemmatize=True)

    # Stop words like 'the', 'is', 'and', 'in' should be eliminated
    assert "the" not in tokens
    assert "is" not in tokens
    assert "python" in tokens
    assert "run" in tokens or "running" in tokens


def test_extract_candidate_keywords():
    text = "Machine learning and natural language processing with deep neural networks."
    keywords = nlp_pipeline.extract_candidate_keywords(text)

    assert any("machine learning" in kw for kw in keywords)
    assert any("natural language" in kw for kw in keywords)
