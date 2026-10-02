"""
API Client Bridge for Streamlit Frontend.
Handles REST requests to FastAPI backend with automatic, seamless fallback
to direct local Python module execution when backend server is not active.
"""

import io
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import requests

logger = logging.getLogger(__name__)

BACKEND_BASE_URL = "http://localhost:8000"


class ATSApiClient:
    """Client for communicating with the ATS scoring backend."""

    def __init__(self, base_url: str = BACKEND_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def is_backend_online(self) -> bool:
        """Checks if the FastAPI backend is running and healthy."""
        try:
            res = requests.get(f"{self.base_url}/api/v1/health", timeout=1.5)
            return res.status_code == 200
        except Exception:
            return False

    def evaluate_single(
        self,
        resume_file: Optional[Any] = None,
        resume_text: Optional[str] = None,
        jd_text: str = "",
        jd_title: str = "Target Role",
        experience_required: Optional[float] = None
    ) -> Dict[str, Any]:
        """Evaluates a single resume via REST API or local in-memory fallback."""
        if self.is_backend_online():
            try:
                data = {
                    "job_description_text": jd_text,
                    "jd_title": jd_title,
                }
                if experience_required is not None:
                    data["experience_required"] = str(experience_required)

                files = {}
                if resume_file is not None:
                    # Streamlit UploadedFile object
                    files["resume_file"] = (
                        resume_file.name,
                        resume_file.getvalue(),
                        resume_file.type or "application/octet-stream",
                    )
                elif resume_text:
                    data["resume_text"] = resume_text

                res = requests.post(f"{self.base_url}/api/v1/evaluate/single", data=data, files=files, timeout=60)
                if res.status_code == 200:
                    return res.json()
            except Exception as e:
                logger.warning(f"Backend request failed ({e}), using local execution fallback.")

        # In-memory fallback
        return self._local_evaluate_single(resume_file, resume_text, jd_text, jd_title, experience_required)

    def evaluate_batch(
        self,
        resume_files: List[Any],
        jd_text: str = "",
        jd_title: str = "Target Role",
        experience_required: Optional[float] = None
    ) -> Dict[str, Any]:
        """Evaluates multiple resumes via REST API or local in-memory fallback."""
        if self.is_backend_online():
            try:
                data = {
                    "job_description_text": jd_text,
                    "jd_title": jd_title,
                }
                if experience_required is not None:
                    data["experience_required"] = str(experience_required)

                files = [
                    ("resume_files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                    for f in resume_files
                ]

                res = requests.post(f"{self.base_url}/api/v1/evaluate/batch", data=data, files=files, timeout=120)
                if res.status_code == 200:
                    return res.json()
            except Exception as e:
                logger.warning(f"Batch backend request failed ({e}), using local fallback.")

        # In-memory fallback
        return self._local_evaluate_batch(resume_files, jd_text, jd_title, experience_required)

    def register(
        self,
        email: str,
        password: str,
        role: str = "candidate",
        company_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Registers a user via API or local database fallback."""
        if self.is_backend_online():
            try:
                res = requests.post(
                    f"{self.base_url}/api/v1/auth/register",
                    json={
                        "email": email,
                        "password": password,
                        "role": role,
                        "company_name": company_name,
                    },
                    timeout=10,
                )
                if res.status_code == 200:
                    return {"success": True, "user": res.json()}
                return {"success": False, "error": res.json().get("detail", "Registration failed.")}
            except Exception as e:
                logger.warning(f"Backend register call failed ({e}), using local DB fallback.")

        # Local database fallback
        from backend.database import SessionLocal, create_user
        db = SessionLocal()
        try:
            user = create_user(db, email=email, password=password, role=role, company_name=company_name)
            return {
                "success": True,
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "role": user.role,
                    "company_name": user.company_name,
                    "created_at": user.created_at.isoformat(),
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            db.close()

    def login(self, email: str, password: str) -> Dict[str, Any]:
        """Authenticates user via API or local database fallback."""
        if self.is_backend_online():
            try:
                res = requests.post(
                    f"{self.base_url}/api/v1/auth/login",
                    json={"email": email, "password": password},
                    timeout=10,
                )
                if res.status_code == 200:
                    return {"success": True, "user": res.json()}
                return {"success": False, "error": res.json().get("detail", "Invalid email or password.")}
            except Exception as e:
                logger.warning(f"Backend login call failed ({e}), using local DB fallback.")

        # Local database fallback
        from backend.database import SessionLocal, authenticate_user
        db = SessionLocal()
        try:
            user = authenticate_user(db, email=email, password=password)
            if not user:
                return {"success": False, "error": "Invalid email or password."}
            return {
                "success": True,
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "role": user.role,
                    "company_name": user.company_name,
                    "created_at": user.created_at.isoformat(),
                },
            }
        finally:
            db.close()

    def compare_candidates(self, evaluation_ids: List[int]) -> Dict[str, Any]:
        """Compares multiple candidates side-by-side."""
        if self.is_backend_online():
            try:
                res = requests.post(
                    f"{self.base_url}/api/v1/candidates/compare",
                    json={"evaluation_ids": evaluation_ids},
                    timeout=10,
                )
                if res.status_code == 200:
                    return res.json()
            except Exception as e:
                logger.warning(f"Backend comparison failed ({e}), using local fallback.")

        # Local fallback
        from backend.database import SessionLocal, Evaluation
        db = SessionLocal()
        try:
            evaluations = db.query(Evaluation).filter(Evaluation.id.in_(evaluation_ids)).all()
            evaluations.sort(key=lambda x: x.ats_score, reverse=True)
            items = []
            job_title = "Candidate Comparison"
            for ev in evaluations:
                if ev.job_description and ev.job_description.title:
                    job_title = ev.job_description.title
                candidate = ev.candidate
                items.append({
                    "name": candidate.name if candidate else "Candidate",
                    "email": candidate.email if candidate else None,
                    "ats_score": ev.ats_score,
                    "match_percentage": ev.match_percentage,
                    "keyword_score": ev.keyword_score,
                    "semantic_score": ev.semantic_score,
                    "experience_score": ev.experience_score,
                    "education_score": ev.education_score,
                    "certification_score": ev.certification_score,
                    "experience_years": candidate.experience_years if candidate else 0.0,
                    "highest_education": candidate.highest_education if candidate else "Not specified",
                    "matched_skills": json.loads(ev.matched_skills or "[]"),
                    "missing_skills": json.loads(ev.missing_skills or "[]"),
                })
            return {
                "job_title": job_title,
                "total_compared": len(items),
                "candidates": items,
            }
        finally:
            db.close()

    def _local_evaluate_single(
        self,
        resume_file: Optional[Any],
        resume_text: Optional[str],
        jd_text: str,
        jd_title: str,
        experience_required: Optional[float]
    ) -> Dict[str, Any]:
        """Runs the complete parsing, scoring, and recommendation pipeline in-process."""
        from backend.parser import document_parser
        from backend.skill_extractor import skill_extractor
        from backend.ats_engine import ats_engine
        from backend.recommendation_engine import recommendation_engine

        if resume_file:
            raw_text = document_parser.extract_text(resume_file.getvalue(), filename=resume_file.name)
        else:
            raw_text = resume_text or ""

        contact = document_parser.extract_contact_info(raw_text)
        name = document_parser.extract_candidate_name(raw_text)
        if name == "Candidate" and resume_file:
            name = resume_file.name.rsplit(".", 1)[0].replace("_", " ").title()

        sections = document_parser.segment_sections(raw_text)
        skills = skill_extractor.extract_skills(raw_text)
        edu_highest, edu_entries = skill_extractor.extract_education(raw_text)
        certs = skill_extractor.extract_certifications(raw_text)
        exp_years = skill_extractor.extract_experience_years(raw_text)

        parsed_jd = document_parser.parse_job_description(jd_text, default_title=jd_title)
        jd_skills = skill_extractor.extract_skills(jd_text)
        exp_req = experience_required if experience_required is not None else parsed_jd.get("experience_required", 0.0)

        candidate_data = {
            "name": name,
            "email": contact.get("email"),
            "phone": contact.get("phone"),
            "linkedin": contact.get("linkedin"),
            "github": contact.get("github"),
            "experience_years": exp_years,
            "highest_education": edu_highest,
            "skills": skills,
            "certifications": certs,
            "projects": [p.strip() for p in sections.get("projects", "").split("\n") if len(p.strip()) > 5][:5],
            "education_entries": edu_entries,
            "sections": sections,
            "raw_text": raw_text,
        }

        jd_data = {
            "title": parsed_jd.get("title", jd_title),
            "company": "Company",
            "experience_required": exp_req,
            "min_education": parsed_jd.get("min_education", "Bachelor's Degree"),
            "skills": jd_skills,
            "raw_text": jd_text,
        }

        eval_res = ats_engine.evaluate_candidate(candidate_data, jd_data)
        skill_gap = eval_res["skill_gap"]
        recs = recommendation_engine.generate_full_recommendations(
            candidate_data,
            eval_res["ats_score"],
            skill_gap["matched_skills"],
            skill_gap["missing_skills"],
        )

        return {
            "evaluation_id": 1,
            "candidate": candidate_data,
            "job_description": jd_data,
            "score_breakdown": {
                "keyword_matching": eval_res["breakdown"]["keyword_matching"],
                "semantic_similarity": eval_res["breakdown"]["semantic_similarity"],
                "experience_matching": eval_res["breakdown"]["experience_matching"],
                "education_matching": eval_res["breakdown"]["education_matching"],
                "certification_matching": eval_res["breakdown"]["certification_matching"],
                "ats_score": eval_res["ats_score"],
                "match_percentage": eval_res["match_percentage"],
            },
            "skill_gap": {
                "matched_skills": skill_gap["matched_skills"],
                "missing_skills": skill_gap["missing_skills"],
                "recommended_skills": recs.get("recommended_skills", []),
                "skill_match_percentage": skill_gap["skill_match_percentage"],
            },
            "recommendations": recs,
            "candidate_rank": 1,
        }

    def _local_evaluate_batch(
        self,
        resume_files: List[Any],
        jd_text: str,
        jd_title: str,
        experience_required: Optional[float]
    ) -> Dict[str, Any]:
        """Runs batch evaluation locally."""
        from backend.ats_engine import ats_engine
        import uuid

        candidates_eval = []
        for file in resume_files:
            single = self._local_evaluate_single(
                resume_file=file,
                resume_text=None,
                jd_text=jd_text,
                jd_title=jd_title,
                experience_required=experience_required,
            )
            candidates_eval.append(single)

        # Rank candidates
        ranked_payloads = []
        for c in candidates_eval:
            ranked_payloads.append({
                "candidate": c["candidate"],
                "scores": {
                    "ats_score": c["score_breakdown"]["ats_score"],
                    "match_percentage": c["score_breakdown"]["match_percentage"],
                    "breakdown": c["score_breakdown"],
                },
                "skill_gap": c["skill_gap"],
                "recommendations": c["recommendations"],
                "evaluation_id": c.get("evaluation_id", 1),
            })

        ranked = ats_engine.rank_candidates(ranked_payloads)

        results = []
        for r in ranked:
            results.append({
                "evaluation_id": r.get("evaluation_id", 1),
                "candidate": r["candidate"],
                "job_description": candidates_eval[0]["job_description"] if candidates_eval else {},
                "score_breakdown": r["scores"]["breakdown"],
                "skill_gap": r["skill_gap"],
                "recommendations": r["recommendations"],
                "candidate_rank": r["rank"],
            })

        return {
            "batch_id": f"batch_{uuid.uuid4().hex[:8]}",
            "total_candidates": len(results),
            "job_title": jd_title,
            "rankings": results,
        }


api_client = ATSApiClient()
