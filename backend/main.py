"""
FastAPI Main Application for AI Resume Screening & ATS Score Predictor.
Exposes REST endpoints for parsing, ATS scoring, multi-resume batch ranking,
report generation, and model benchmarking.
"""

import json
import os
import uuid
from typing import List, Optional
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

from backend.database import (
    Candidate,
    Evaluation,
    JobDescription,
    User,
    authenticate_user,
    create_user,
    get_all_evaluations,
    get_db,
    get_evaluation_by_id,
    get_evaluations_by_batch,
    get_user_by_email,
    init_db,
    save_evaluation_record,
)
from backend.schemas import (
    ATSScoreBreakdown,
    BatchEvaluationResponse,
    CandidateComparisonItem,
    CandidateComparisonResponse,
    CandidateCompareRequest,
    CandidateParsed,
    EvaluationResponse,
    JobDescriptionParsed,
    ModelBenchmarkItem,
    ModelComparisonResponse,
    RecommendationResult,
    SingleEvaluationRequest,
    SkillGapAnalysis,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
)
from backend.parser import document_parser
from backend.nlp_pipeline import nlp_pipeline
from backend.skill_extractor import skill_extractor
from backend.similarity_engine import similarity_engine
from backend.ats_engine import ats_engine
from backend.recommendation_engine import recommendation_engine
from backend.report_generator import report_generator

# Initialize database
init_db()

app = FastAPI(
    title="AI Resume Screening & ATS Score Predictor API",
    version="1.0.0",
    description="End-to-end NLP & Machine Learning system to screen resumes, compute ATS scores, rank candidates, and generate improvement recommendations.",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend and external integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
def health_check():
    """Returns system status and model readiness."""
    return {
        "status": "healthy",
        "service": "AI Resume Screening & ATS Score Predictor",
        "nlp_engine": "sentence-transformers/all-MiniLM-L6-v2",
        "sbert_available": similarity_engine.use_sbert,
        "database": "SQLite (operational)"
    }


@app.post("/api/v1/auth/register", response_model=UserResponse)
def register_user_endpoint(payload: UserRegisterRequest, db: Session = Depends(get_db)):
    """Registers a new candidate or recruiter user account."""
    role = payload.role.strip().lower()
    if role not in ["candidate", "recruiter"]:
        raise HTTPException(status_code=400, detail="Role must be either 'candidate' or 'recruiter'.")
    try:
        user = create_user(
            db=db,
            email=payload.email,
            password=payload.password,
            role=role,
            company_name=payload.company_name,
        )
        return UserResponse(
            id=user.id,
            email=user.email,
            role=user.role,
            company_name=user.company_name,
            created_at=user.created_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/auth/login", response_model=UserResponse)
def login_user_endpoint(payload: UserLoginRequest, db: Session = Depends(get_db)):
    """Authenticates a candidate or recruiter and returns profile data."""
    user = authenticate_user(db=db, email=payload.email, password=payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return UserResponse(
        id=user.id,
        email=user.email,
        role=user.role,
        company_name=user.company_name,
        created_at=user.created_at,
    )


@app.post("/api/v1/candidates/compare", response_model=CandidateComparisonResponse)
def compare_candidates_endpoint(payload: CandidateCompareRequest, db: Session = Depends(get_db)):
    """Compares 2 or more candidates side-by-side across ATS dimensions."""
    if not payload.evaluation_ids or len(payload.evaluation_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 evaluation IDs are required for comparison.")

    evaluations = db.query(Evaluation).filter(Evaluation.id.in_(payload.evaluation_ids)).all()
    if not evaluations:
        raise HTTPException(status_code=404, detail="No evaluations found matching the specified IDs.")

    evaluations.sort(key=lambda x: x.ats_score, reverse=True)

    items = []
    job_title = "Candidate Comparison"
    for ev in evaluations:
        if ev.job_description and ev.job_description.title:
            job_title = ev.job_description.title
        candidate = ev.candidate
        matched = json.loads(ev.matched_skills or "[]")
        missing = json.loads(ev.missing_skills or "[]")
        items.append(
            CandidateComparisonItem(
                name=candidate.name if candidate else "Candidate",
                email=candidate.email if candidate else None,
                ats_score=ev.ats_score,
                match_percentage=ev.match_percentage,
                keyword_score=ev.keyword_score,
                semantic_score=ev.semantic_score,
                experience_score=ev.experience_score,
                education_score=ev.education_score,
                certification_score=ev.certification_score,
                experience_years=candidate.experience_years if candidate else 0.0,
                highest_education=candidate.highest_education if candidate else "Not specified",
                matched_skills=matched,
                missing_skills=missing,
            )
        )

    return CandidateComparisonResponse(
        job_title=job_title,
        total_compared=len(items),
        candidates=items,
    )


@app.post("/api/v1/parse/resume", response_model=CandidateParsed)
async def parse_resume_endpoint(file: UploadFile = File(...)):
    """Extracts raw text, structured sections, candidate details, skills, and qualifications."""
    content = await file.read()
    raw_text = document_parser.extract_text(content, filename=file.filename or "resume.txt")

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from the uploaded file.")

    contact = document_parser.extract_contact_info(raw_text)
    name = document_parser.extract_candidate_name(raw_text)
    sections = document_parser.segment_sections(raw_text)

    skills = skill_extractor.extract_skills(raw_text)
    edu_highest, edu_entries = skill_extractor.extract_education(raw_text)
    certs = skill_extractor.extract_certifications(raw_text)
    exp_years = skill_extractor.extract_experience_years(raw_text)

    # Extract project names/lines
    proj_lines = [p.strip() for p in sections.get("projects", "").split("\n") if len(p.strip()) > 5][:5]

    return CandidateParsed(
        name=name,
        email=contact.get("email"),
        phone=contact.get("phone"),
        linkedin=contact.get("linkedin"),
        github=contact.get("github"),
        experience_years=exp_years,
        highest_education=edu_highest,
        skills=skills,
        certifications=certs,
        projects=proj_lines,
        education_entries=edu_entries,
        sections=sections,
        raw_text=raw_text,
    )


@app.post("/api/v1/parse/job-description", response_model=JobDescriptionParsed)
def parse_job_description_endpoint(
    jd_text: str = Form(...),
    jd_title: Optional[str] = Form("Target Position"),
    experience_required: Optional[float] = Form(None)
):
    """Parses raw job description text to detect title, experience, minimum education, and required skills."""
    parsed = document_parser.parse_job_description(jd_text, default_title=jd_title or "Target Position")
    skills = skill_extractor.extract_skills(jd_text)

    exp_req = experience_required if experience_required is not None else parsed.get("experience_required", 0.0)

    return JobDescriptionParsed(
        title=parsed.get("title", jd_title),
        company="Company",
        experience_required=exp_req,
        min_education=parsed.get("min_education", "Bachelor's Degree"),
        skills=skills,
        raw_text=jd_text,
    )


@app.post("/api/v1/evaluate/single", response_model=EvaluationResponse)
async def evaluate_single_endpoint(
    resume_file: Optional[UploadFile] = File(None),
    resume_text: Optional[str] = Form(None),
    job_description_text: str = Form(...),
    jd_title: Optional[str] = Form("Target Role"),
    experience_required: Optional[float] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Evaluates a single resume against a job description.
    Supports either file upload or raw text submission.
    Computes 5-pillar ATS score, detects skill gaps, and stores record in SQLite.
    """
    # 1. Parse Resume
    if resume_file:
        content = await resume_file.read()
        extracted_text = document_parser.extract_text(content, filename=resume_file.filename or "")
    elif resume_text:
        extracted_text = resume_text
    else:
        raise HTTPException(status_code=400, detail="Must provide either resume_file or resume_text.")

    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="Empty resume text provided.")

    contact = document_parser.extract_contact_info(extracted_text)
    name = document_parser.extract_candidate_name(extracted_text)
    sections = document_parser.segment_sections(extracted_text)
    cand_skills = skill_extractor.extract_skills(extracted_text)
    edu_highest, edu_entries = skill_extractor.extract_education(extracted_text)
    certs = skill_extractor.extract_certifications(extracted_text)
    exp_years = skill_extractor.extract_experience_years(extracted_text)
    proj_lines = [p.strip() for p in sections.get("projects", "").split("\n") if len(p.strip()) > 5][:5]

    candidate_data = {
        "name": name,
        "email": contact.get("email"),
        "phone": contact.get("phone"),
        "linkedin": contact.get("linkedin"),
        "github": contact.get("github"),
        "experience_years": exp_years,
        "highest_education": edu_highest,
        "skills": cand_skills,
        "certifications": certs,
        "projects": proj_lines,
        "education_entries": edu_entries,
        "sections": sections,
        "raw_text": extracted_text,
    }

    # 2. Parse Job Description
    parsed_jd = document_parser.parse_job_description(job_description_text, default_title=jd_title or "Target Role")
    jd_skills = skill_extractor.extract_skills(job_description_text)
    exp_req = experience_required if experience_required is not None else parsed_jd.get("experience_required", 0.0)

    jd_data = {
        "title": parsed_jd.get("title", jd_title),
        "company": "Hiring Team",
        "experience_required": exp_req,
        "min_education": parsed_jd.get("min_education", "Bachelor's Degree"),
        "skills": jd_skills,
        "raw_text": job_description_text,
    }

    # 3. Calculate ATS Score
    eval_result = ats_engine.evaluate_candidate(candidate_data, jd_data)
    scores = {
        "ats_score": eval_result["ats_score"],
        "match_percentage": eval_result["match_percentage"],
        "keyword_score": eval_result["breakdown"]["keyword_matching"],
        "semantic_score": eval_result["breakdown"]["semantic_similarity"],
        "experience_score": eval_result["breakdown"]["experience_matching"],
        "education_score": eval_result["breakdown"]["education_matching"],
        "certification_score": eval_result["breakdown"]["certification_matching"],
    }
    skill_gap = eval_result["skill_gap"]

    # 4. Generate Recommendations
    recs = recommendation_engine.generate_full_recommendations(
        candidate_data,
        eval_result["ats_score"],
        skill_gap["matched_skills"],
        skill_gap["missing_skills"],
    )

    # 5. Persist to Database
    db_record = save_evaluation_record(
        db=db,
        candidate_data=candidate_data,
        jd_data=jd_data,
        scores=scores,
        skill_gap=skill_gap,
        recommendations=recs,
    )

    return EvaluationResponse(
        evaluation_id=db_record.id,
        candidate=CandidateParsed(**candidate_data),
        job_description=JobDescriptionParsed(**jd_data),
        score_breakdown=ATSScoreBreakdown(
            keyword_matching=scores["keyword_score"],
            semantic_similarity=scores["semantic_score"],
            experience_matching=scores["experience_score"],
            education_matching=scores["education_score"],
            certification_matching=scores["certification_score"],
            ats_score=scores["ats_score"],
            match_percentage=scores["match_percentage"],
        ),
        skill_gap=SkillGapAnalysis(
            matched_skills=skill_gap["matched_skills"],
            missing_skills=skill_gap["missing_skills"],
            recommended_skills=recs.get("recommended_skills", []),
            skill_match_percentage=skill_gap["skill_match_percentage"],
        ),
        recommendations=RecommendationResult(
            missing_keywords=recs.get("missing_keywords", []),
            suggested_certifications=recs.get("suggested_certifications", []),
            suggested_projects=recs.get("suggested_projects", []),
            resume_improvement_recommendations=recs.get("resume_improvement_recommendations", []),
        ),
        candidate_rank=1,
        created_at=db_record.created_at,
    )


@app.post("/api/v1/evaluate/batch", response_model=BatchEvaluationResponse)
async def evaluate_batch_endpoint(
    resume_files: List[UploadFile] = File(...),
    job_description_text: str = Form(...),
    jd_title: Optional[str] = Form("Target Role"),
    experience_required: Optional[float] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Evaluates multiple resumes simultaneously, computes ATS scores,
    and ranks candidates in descending order.
    """
    if not resume_files:
        raise HTTPException(status_code=400, detail="No resume files uploaded.")

    # Parse JD once
    parsed_jd = document_parser.parse_job_description(job_description_text, default_title=jd_title or "Target Role")
    jd_skills = skill_extractor.extract_skills(job_description_text)
    exp_req = experience_required if experience_required is not None else parsed_jd.get("experience_required", 0.0)

    jd_data = {
        "title": parsed_jd.get("title", jd_title),
        "company": "Hiring Team",
        "experience_required": exp_req,
        "min_education": parsed_jd.get("min_education", "Bachelor's Degree"),
        "skills": jd_skills,
        "raw_text": job_description_text,
    }

    batch_id = f"batch_{uuid.uuid4().hex[:10]}"
    evaluated_list = []

    for file in resume_files:
        content = await file.read()
        extracted_text = document_parser.extract_text(content, filename=file.filename or "")
        if not extracted_text.strip():
            continue

        contact = document_parser.extract_contact_info(extracted_text)
        name = document_parser.extract_candidate_name(extracted_text)
        if name == "Candidate":
            # Use base filename as candidate name fallback
            base_name = os.path.splitext(file.filename)[0].replace("_", " ").title()
            name = base_name

        sections = document_parser.segment_sections(extracted_text)
        cand_skills = skill_extractor.extract_skills(extracted_text)
        edu_highest, edu_entries = skill_extractor.extract_education(extracted_text)
        certs = skill_extractor.extract_certifications(extracted_text)
        exp_years = skill_extractor.extract_experience_years(extracted_text)
        proj_lines = [p.strip() for p in sections.get("projects", "").split("\n") if len(p.strip()) > 5][:5]

        candidate_data = {
            "name": name,
            "email": contact.get("email"),
            "phone": contact.get("phone"),
            "linkedin": contact.get("linkedin"),
            "github": contact.get("github"),
            "experience_years": exp_years,
            "highest_education": edu_highest,
            "skills": cand_skills,
            "certifications": certs,
            "projects": proj_lines,
            "education_entries": edu_entries,
            "sections": sections,
            "raw_text": extracted_text,
        }

        eval_result = ats_engine.evaluate_candidate(candidate_data, jd_data)
        skill_gap = eval_result["skill_gap"]
        recs = recommendation_engine.generate_full_recommendations(
            candidate_data,
            eval_result["ats_score"],
            skill_gap["matched_skills"],
            skill_gap["missing_skills"],
        )

        evaluated_list.append({
            "candidate": candidate_data,
            "scores": eval_result,
            "skill_gap": skill_gap,
            "recommendations": recs,
        })

    # Rank candidates
    ranked = ats_engine.rank_candidates(evaluated_list)

    final_responses: List[EvaluationResponse] = []
    for cand_entry in ranked:
        scores_dict = cand_entry["scores"]
        scores_flat = {
            "ats_score": scores_dict["ats_score"],
            "match_percentage": scores_dict["match_percentage"],
            "keyword_score": scores_dict["breakdown"]["keyword_matching"],
            "semantic_score": scores_dict["breakdown"]["semantic_similarity"],
            "experience_score": scores_dict["breakdown"]["experience_matching"],
            "education_score": scores_dict["breakdown"]["education_matching"],
            "certification_score": scores_dict["breakdown"]["certification_matching"],
        }

        # Save each to DB with batch_id and rank
        db_rec = save_evaluation_record(
            db=db,
            candidate_data=cand_entry["candidate"],
            jd_data=jd_data,
            scores=scores_flat,
            skill_gap=cand_entry["skill_gap"],
            recommendations=cand_entry["recommendations"],
            batch_id=batch_id,
            rank=cand_entry["rank"],
        )

        final_responses.append(
            EvaluationResponse(
                evaluation_id=db_rec.id,
                candidate=CandidateParsed(**cand_entry["candidate"]),
                job_description=JobDescriptionParsed(**jd_data),
                score_breakdown=ATSScoreBreakdown(
                    keyword_matching=scores_flat["keyword_score"],
                    semantic_similarity=scores_flat["semantic_score"],
                    experience_matching=scores_flat["experience_score"],
                    education_matching=scores_flat["education_score"],
                    certification_matching=scores_flat["certification_score"],
                    ats_score=scores_flat["ats_score"],
                    match_percentage=scores_flat["match_percentage"],
                ),
                skill_gap=SkillGapAnalysis(
                    matched_skills=cand_entry["skill_gap"]["matched_skills"],
                    missing_skills=cand_entry["skill_gap"]["missing_skills"],
                    recommended_skills=cand_entry["recommendations"].get("recommended_skills", []),
                    skill_match_percentage=cand_entry["skill_gap"]["skill_match_percentage"],
                ),
                recommendations=RecommendationResult(
                    missing_keywords=cand_entry["recommendations"].get("missing_keywords", []),
                    suggested_certifications=cand_entry["recommendations"].get("suggested_certifications", []),
                    suggested_projects=cand_entry["recommendations"].get("suggested_projects", []),
                    resume_improvement_recommendations=cand_entry["recommendations"].get("resume_improvement_recommendations", []),
                ),
                candidate_rank=cand_entry["rank"],
                created_at=db_rec.created_at,
            )
        )

    return BatchEvaluationResponse(
        batch_id=batch_id,
        total_candidates=len(final_responses),
        job_title=jd_data["title"],
        rankings=final_responses,
    )


@app.get("/api/v1/reports/pdf/{evaluation_id}")
def download_pdf_report_endpoint(evaluation_id: int, db: Session = Depends(get_db)):
    """Generates and serves a downloadable executive PDF report for a candidate evaluation."""
    record = get_evaluation_by_id(db, evaluation_id)
    if not record:
        raise HTTPException(status_code=404, detail="Evaluation record not found.")

    cand = record.candidate
    jd = record.job_description

    eval_payload = {
        "candidate": {
            "name": cand.name,
            "email": cand.email,
            "phone": cand.phone,
            "experience_years": cand.experience_years,
            "highest_education": cand.highest_education,
        },
        "job_description": {
            "title": jd.title,
            "experience_required": jd.experience_required,
            "min_education": jd.min_education,
        },
        "score_breakdown": {
            "ats_score": record.ats_score,
            "match_percentage": record.match_percentage,
            "keyword_matching": record.keyword_score,
            "semantic_similarity": record.semantic_score,
            "experience_matching": record.experience_score,
            "education_matching": record.education_score,
            "certification_matching": record.certification_score,
        },
        "skill_gap": {
            "matched_skills": json.loads(record.matched_skills or "[]"),
            "missing_skills": json.loads(record.missing_skills or "[]"),
        },
        "recommendations": json.loads(record.recommendations or "{}"),
    }

    pdf_file = report_generator.generate_pdf_report(eval_payload)
    filename = os.path.basename(pdf_file)
    return FileResponse(pdf_file, media_type="application/pdf", filename=filename)


@app.get("/api/v1/reports/excel/{batch_id}")
def download_excel_report_endpoint(batch_id: str, db: Session = Depends(get_db)):
    """Generates and serves a downloadable Excel leaderboard report for a batch evaluation."""
    records = get_evaluations_by_batch(db, batch_id)
    if not records:
        raise HTTPException(status_code=404, detail="No evaluations found for this batch ID.")

    rankings = []
    job_title = "Position"
    for r in records:
        job_title = r.job_description.title
        rankings.append({
            "candidate_rank": r.candidate_rank,
            "candidate": {
                "name": r.candidate.name,
                "email": r.candidate.email,
                "phone": r.candidate.phone,
                "experience_years": r.candidate.experience_years,
                "highest_education": r.candidate.highest_education,
            },
            "score_breakdown": {
                "ats_score": r.ats_score,
                "match_percentage": r.match_percentage,
                "keyword_matching": r.keyword_score,
                "semantic_similarity": r.semantic_score,
                "experience_matching": r.experience_score,
                "education_matching": r.education_score,
                "certification_matching": r.certification_score,
            },
            "skill_gap": {
                "matched_skills": json.loads(r.matched_skills or "[]"),
                "missing_skills": json.loads(r.missing_skills or "[]"),
                "skill_match_percentage": round((r.keyword_score), 2),
            },
            "recommendations": json.loads(r.recommendations or "{}"),
        })

    excel_file = report_generator.generate_excel_batch_report(rankings, job_title=job_title)
    filename = os.path.basename(excel_file)
    return FileResponse(
        excel_file,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename,
    )


@app.get("/api/v1/models/benchmark", response_model=ModelComparisonResponse)
def model_benchmark_endpoint():
    """Runs live performance comparison of TF-IDF, Jaccard, and Sentence-Transformers."""
    sample_resume = (
        "Senior Machine Learning Engineer with 6 years experience specializing in Natural Language Processing, "
        "PyTorch, Large Language Models, Fine-tuning BERT and Transformers, FastAPI, Docker, and Kubernetes. "
        "Designed and deployed real-time semantic search and retrieval-augmented generation systems on AWS."
    )
    sample_jd = (
        "Looking for a Senior NLP Data Scientist to build generative AI solutions. "
        "Requirements: 5+ years of experience with Python, PyTorch, Deep Learning, LLMs, Vector Databases, "
        "and deploying high-throughput microservices using Docker and Kubernetes in cloud environments (AWS/GCP)."
    )

    benchmarks = similarity_engine.benchmark_comparison(sample_resume, sample_jd)
    items = [
        ModelBenchmarkItem(
            method=v["method"],
            latency_ms=v["latency_ms"],
            similarity_score=v["similarity_score"],
            description=v["description"],
            pros=v["pros"],
            cons=v["cons"],
        )
        for k, v in benchmarks.items()
    ]

    justification = (
        "Sentence-Transformers (all-MiniLM-L6-v2) was selected as our primary semantic engine because it operates "
        "in a continuous 384-dimensional dense metric space, successfully bridging vocabulary mismatches "
        "(e.g., matching 'Natural Language Processing' with 'NLP' and 'Generative AI' with 'LLMs') while achieving "
        "sub-50ms CPU inference latencies. TF-IDF is retained as a robust baseline and offline fallback."
    )

    return ModelComparisonResponse(
        selected_method="Sentence Transformers (all-MiniLM-L6-v2)",
        justification=justification,
        benchmarks=items,
    )
