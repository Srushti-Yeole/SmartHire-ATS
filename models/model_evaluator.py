"""
Model Evaluator Module.
Provides quantitative evaluation metrics to compare text representation algorithms
(TF-IDF, Jaccard Index, and Sentence Transformers) on resume-to-job-description matching.
"""

import time
from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ModelEvaluator:
    """Evaluates and compares NLP matching architectures."""

    def __init__(self):
        pass

    def evaluate_semantic_gap(
        self,
        synonym_pairs: List[Tuple[str, str]]
    ) -> Dict[str, float]:
        """
        Measures how well models score pairs of phrases that convey the exact same
        underlying skill using different vocabulary (e.g. 'Natural Language Processing' vs 'NLP').
        A higher score means the model correctly recognizes semantic equivalence.
        """
        from backend.similarity_engine import similarity_engine

        tfidf_scores = []
        sbert_scores = []

        for term_a, term_b in synonym_pairs:
            tfidf_sim = similarity_engine.compute_tfidf_similarity(term_a, term_b)
            sbert_sim = similarity_engine.compute_sbert_similarity(term_a, term_b)
            tfidf_scores.append(tfidf_sim)
            sbert_scores.append(sbert_sim)

        return {
            "tfidf_average_synonym_score": float(np.mean(tfidf_scores)) if tfidf_scores else 0.0,
            "sbert_average_synonym_score": float(np.mean(sbert_scores)) if sbert_scores else 0.0,
            "semantic_gain_percentage": float(
                ((np.mean(sbert_scores) - np.mean(tfidf_scores)) / (np.mean(tfidf_scores) + 1e-6)) * 100
            ) if tfidf_scores else 0.0
        }

    def measure_latency_distribution(
        self,
        text_pairs: List[Tuple[str, str]],
        iterations: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """Measures p50, p95, and mean inference latency in milliseconds."""
        from backend.similarity_engine import similarity_engine

        tfidf_times = []
        sbert_times = []

        for _ in range(iterations):
            for t_a, t_b in text_pairs:
                # TF-IDF
                t0 = time.perf_counter()
                similarity_engine.compute_tfidf_similarity(t_a, t_b)
                tfidf_times.append((time.perf_counter() - t0) * 1000)

                # SBERT
                t0 = time.perf_counter()
                similarity_engine.compute_sbert_similarity(t_a, t_b)
                sbert_times.append((time.perf_counter() - t0) * 1000)

        return {
            "tfidf": {
                "mean_ms": float(np.mean(tfidf_times)),
                "p50_ms": float(np.percentile(tfidf_times, 50)),
                "p95_ms": float(np.percentile(tfidf_times, 95)),
            },
            "sentence_transformers": {
                "mean_ms": float(np.mean(sbert_times)),
                "p50_ms": float(np.percentile(sbert_times, 50)),
                "p95_ms": float(np.percentile(sbert_times, 95)),
            }
        }


# Global evaluator instance
model_evaluator = ModelEvaluator()
