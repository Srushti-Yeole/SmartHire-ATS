"""
ATS Scoring Engine Module.
Implements the calibrated 5-factor scoring model:
  - Keyword Matching: 40%
  - Semantic Similarity: 30%
  - Experience Matching: 15%
  - Education Matching: 5%
  - Certification Matching: 10%

Provides multi-candidate ranking and percentile calculation.
"""

from typing import Any, Dict, List, Optional, Tuple
from backend.similarity_engine import similarity_engine


class ATSEngine:
    """Production ATS evaluation and scoring engine."""

    # Weights defined by project requirements
    WEIGHT_KEYWORD = 0.40
    WEIGHT_SEMANTIC = 0.30
    WEIGHT_EXPERIENCE = 0.15
    WEIGHT_EDUCATION = 0.05
    WEIGHT_CERTIFICATION = 0.10

    EDUCATION_RANKS = {
        "phd / doctorate": 4,
        "phd": 4,
        "doctorate": 4,
        "master's degree": 3,
        "master": 3,
        "ms": 3,
        "m.tech": 3,
        "mba": 3,
        "bachelor's degree": 2,
        "bachelor": 2,
        "bs": 2,
        "b.tech": 2,
        "b.e": 2,
        "diploma / associate": 1,
        "associate": 1,
        "diploma": 1,
        "not specified": 0,
    }

    def compute_keyword_score(
        self, candidate_skills: List[str], jd_skills: List[str]
    ) -> Tuple[float, List[str], List[str]]:
        """
        Calculates Keyword Matching Score (0 - 100) and extracts matched & missing skills.
        """
        if not jd_skills:
            # If JD has no explicit skills parsed, give baseline 75 if candidate has any skills
            return (80.0 if candidate_skills else 50.0), candidate_skills, []

        candidate_set = {s.lower().strip() for s in candidate_skills}
        matched = []
        missing = []

        for skill in jd_skills:
            if skill.lower().strip() in candidate_set:
                matched.append(skill)
            else:
                missing.append(skill)

        match_ratio = len(matched) / len(jd_skills)
        score = min(100.0, match_ratio * 100.0)
        return round(score, 2), matched, missing

    def compute_semantic_score(self, resume_text: str, jd_text: str) -> float:
        """
        Calculates Semantic Similarity Score (0 - 100) using Sentence Transformers / TF-IDF.
        """
        raw_sim = similarity_engine.calculate_similarity(resume_text, jd_text, method="sentence_transformer")
        # Scale to 0-100
        score = min(100.0, max(0.0, raw_sim * 100.0))
        return round(score, 2)

    def compute_experience_score(
        self, candidate_exp: float, required_exp: float
    ) -> float:
        """
        Calculates Experience Matching Score (0 - 100).
        """
        if required_exp <= 0.0:
            return 100.0 if candidate_exp > 0 else 85.0

        if candidate_exp >= required_exp:
            # Fully meets or exceeds requirement
            # Slight bonus for seniority, capped at 100
            surplus = candidate_exp - required_exp
            bonus = min(5.0, surplus * 1.0)
            return min(100.0, 95.0 + bonus)

        # Candidate has less experience than required
        ratio = candidate_exp / required_exp
        score = max(25.0, ratio * 90.0)
        return round(score, 2)

    def compute_education_score(
        self, candidate_edu: str, required_edu: str
    ) -> float:
        """
        Calculates Education Matching Score (0 - 100) based on qualification tiers.
        """
        cand_rank = self.EDUCATION_RANKS.get(candidate_edu.lower().strip(), 1)
        req_rank = self.EDUCATION_RANKS.get(required_edu.lower().strip(), 2)

        if cand_rank >= req_rank:
            return 100.0
        elif cand_rank == req_rank - 1:
            return 75.0
        elif cand_rank == req_rank - 2:
            return 50.0
        else:
            return 30.0

    def compute_certification_score(
        self, candidate_certs: List[str], jd_skills: List[str]
    ) -> float:
        """
        Calculates Certification Matching Score (0 - 100).
        """
        if not candidate_certs:
            # If no certifications are listed, return base credit since not all roles require them
            return 40.0

        num_certs = len(candidate_certs)
        # Check relevance
        jd_terms = " ".join(jd_skills).lower()
        has_relevant = any(
            any(w in cert.lower() for w in ["aws", "azure", "gcp", "cloud", "kubernetes", "ml", "pmp", "scrum", "data"])
            for cert in candidate_certs
        )

        if num_certs >= 2 and has_relevant:
            return 100.0
        elif num_certs >= 1 and has_relevant:
            return 85.0
        elif num_certs >= 1:
            return 70.0
        return 40.0

    def evaluate_candidate(
        self,
        candidate_data: Dict[str, Any],
        jd_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Runs the full 5-pillar ATS scoring algorithm on a candidate against a job description.
        """
        candidate_skills = candidate_data.get("skills", [])
        jd_skills = jd_data.get("skills", [])
        resume_text = candidate_data.get("raw_text", "")
        jd_text = jd_data.get("raw_text", "")
        candidate_exp = float(candidate_data.get("experience_years", 0.0))
        required_exp = float(jd_data.get("experience_required", 0.0))
        candidate_edu = candidate_data.get("highest_education", "Not specified")
        required_edu = jd_data.get("min_education", "Bachelor's Degree")
        candidate_certs = candidate_data.get("certifications", [])

        # 1. Keyword Matching (40%)
        keyword_score, matched_skills, missing_skills = self.compute_keyword_score(
            candidate_skills, jd_skills
        )

        # 2. Semantic Similarity (30%)
        semantic_score = self.compute_semantic_score(resume_text, jd_text)

        # 3. Experience Matching (15%)
        experience_score = self.compute_experience_score(candidate_exp, required_exp)

        # 4. Education Matching (5%)
        education_score = self.compute_education_score(candidate_edu, required_edu)

        # 5. Certification Matching (10%)
        certification_score = self.compute_certification_score(candidate_certs, jd_skills)

        # Composite ATS Score
        ats_score = (
            (self.WEIGHT_KEYWORD * keyword_score)
            + (self.WEIGHT_SEMANTIC * semantic_score)
            + (self.WEIGHT_EXPERIENCE * experience_score)
            + (self.WEIGHT_EDUCATION * education_score)
            + (self.WEIGHT_CERTIFICATION * certification_score)
        )
        ats_score = round(min(100.0, max(0.0, ats_score)), 2)

        # Overall Match Percentage
        match_percentage = ats_score

        # Skill match percentage
        total_jd_skills = len(jd_skills) if jd_skills else 1
        skill_overlap_pct = round((len(matched_skills) / total_jd_skills) * 100.0, 2) if jd_skills else keyword_score

        return {
            "ats_score": ats_score,
            "match_percentage": match_percentage,
            "breakdown": {
                "keyword_matching": keyword_score,
                "semantic_similarity": semantic_score,
                "experience_matching": experience_score,
                "education_matching": education_score,
                "certification_matching": certification_score,
            },
            "skill_gap": {
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "skill_match_percentage": skill_overlap_pct,
            },
        }

    def rank_candidates(
        self, evaluated_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Ranks a batch of evaluated candidate records by total ATS Score in descending order.
        Adds rank (1-indexed) and percentile metrics.
        """
        if not evaluated_candidates:
            return []

        # Sort descending by ATS Score, breaking ties with keyword score
        sorted_candidates = sorted(
            evaluated_candidates,
            key=lambda x: (
                x.get("scores", {}).get("ats_score", 0.0),
                x.get("scores", {}).get("breakdown", {}).get("keyword_matching", 0.0),
            ),
            reverse=True,
        )

        n = len(sorted_candidates)
        for idx, cand in enumerate(sorted_candidates):
            rank = idx + 1
            cand["rank"] = rank
            # Percentile: (N - rank + 1) / N * 100
            cand["percentile"] = round(((n - rank + 1) / n) * 100.0, 1)

        return sorted_candidates


# Global ATS engine instance
ats_engine = ATSEngine()
