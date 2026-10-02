"""
Batch candidate ranking verification script.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.parser import document_parser
from backend.skill_extractor import skill_extractor
from backend.ats_engine import ats_engine
from backend.recommendation_engine import recommendation_engine
from backend.report_generator import report_generator


def test_batch_ranking_order():
    resumes_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample_resumes")
    jd_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_job_descriptions", "Job_Senior_ML_Engineer.txt")

    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    parsed_jd = document_parser.parse_job_description(jd_text)
    jd_skills = skill_extractor.extract_skills(jd_text)

    jd_data = {
        "title": parsed_jd["title"],
        "experience_required": parsed_jd["experience_required"],
        "min_education": parsed_jd["min_education"],
        "skills": jd_skills,
        "raw_text": jd_text,
    }

    files = [
        "Alex_Chen_Senior_ML_Engineer.pdf",
        "Sarah_Johnson_FullStack_Developer.docx",
        "Michael_Patel_DevOps_Cloud_Architect.pdf",
        "Emily_Davis_Junior_Data_Analyst.txt",
    ]

    evaluated_batch = []
    for fname in files:
        fpath = os.path.join(resumes_dir, fname)
        raw_text = document_parser.extract_text(fpath)
        contact = document_parser.extract_contact_info(raw_text)
        name = document_parser.extract_candidate_name(raw_text)
        skills = skill_extractor.extract_skills(raw_text)
        edu_tier, _ = skill_extractor.extract_education(raw_text)
        certs = skill_extractor.extract_certifications(raw_text)
        exp_years = skill_extractor.extract_experience_years(raw_text)

        cand_data = {
            "name": name,
            "email": contact["email"],
            "phone": contact["phone"],
            "experience_years": exp_years,
            "highest_education": edu_tier,
            "skills": skills,
            "certifications": certs,
            "raw_text": raw_text,
        }

        eval_res = ats_engine.evaluate_candidate(cand_data, jd_data)
        recs = recommendation_engine.generate_full_recommendations(
            cand_data,
            eval_res["ats_score"],
            eval_res["skill_gap"]["matched_skills"],
            eval_res["skill_gap"]["missing_skills"],
        )

        evaluated_batch.append({
            "candidate": cand_data,
            "scores": eval_res,
            "skill_gap": eval_res["skill_gap"],
            "recommendations": recs,
        })

    ranked = ats_engine.rank_candidates(evaluated_batch)

    # For Senior ML Engineer, Alex Chen MUST be ranked #1
    assert ranked[0]["candidate"]["name"] == "Alex Chen"
    assert ranked[0]["rank"] == 1
    assert ranked[0]["scores"]["ats_score"] > 80.0

    print("\n" + "=" * 75)
    print("  BATCH RANKING VERIFIED FOR: Senior Machine Learning Engineer")
    print("=" * 75)
    for r in ranked:
        print(f"Rank #{r['rank']} | {r['candidate']['name']:<22} | ATS Score: {r['scores']['ats_score']:<5.1f} | Match %: {r['scores']['match_percentage']:<5.1f} | Percentile: {r['percentile']}%")
    print("=" * 75)


if __name__ == "__main__":
    test_batch_ranking_order()
