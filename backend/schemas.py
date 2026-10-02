"""
Pydantic schemas for request validation and response serialization.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


def get_utc_now() -> datetime:
    """Returns the current UTC datetime."""
    return datetime.now(timezone.utc)


class CandidateParsed(BaseModel):
    name: str = Field(default="Anonymous Candidate", description="Extracted candidate name")
    email: Optional[str] = Field(default=None, description="Extracted email address")
    phone: Optional[str] = Field(default=None, description="Extracted telephone number")
    linkedin: Optional[str] = Field(default=None, description="Extracted LinkedIn profile URL")
    github: Optional[str] = Field(default=None, description="Extracted GitHub profile URL")
    experience_years: float = Field(default=0.0, description="Calculated years of experience")
    highest_education: str = Field(default="Not specified", description="Highest detected educational degree")
    skills: List[str] = Field(default_factory=list, description="Categorized skills list")
    certifications: List[str] = Field(default_factory=list, description="Extracted certifications")
    projects: List[str] = Field(default_factory=list, description="Identified project titles or summaries")
    education_entries: List[str] = Field(default_factory=list, description="Detected education entries")
    sections: Dict[str, str] = Field(default_factory=dict, description="Parsed section text mapping")
    raw_text: str = Field(default="", description="Original sanitized raw text")


class JobDescriptionParsed(BaseModel):
    title: str = Field(default="Software Professional", description="Job title")
    company: Optional[str] = Field(default="Company", description="Company or hiring team name")
    experience_required: float = Field(default=0.0, description="Minimum years of experience required")
    min_education: str = Field(default="Bachelor's Degree", description="Minimum degree required")
    skills: List[str] = Field(default_factory=list, description="Extracted required skills")
    raw_text: str = Field(default="", description="Raw job description text")


class ATSScoreBreakdown(BaseModel):
    keyword_matching: float = Field(..., description="Keyword Matching (40% weight)")
    semantic_similarity: float = Field(..., description="Semantic Similarity (30% weight)")
    experience_matching: float = Field(..., description="Experience Matching (15% weight)")
    education_matching: float = Field(..., description="Education Matching (5% weight)")
    certification_matching: float = Field(..., description="Certification Matching (10% weight)")
    ats_score: float = Field(..., description="Total Composite ATS Score [0 - 100]")
    match_percentage: float = Field(..., description="Overall Match Percentage")


class SkillGapAnalysis(BaseModel):
    matched_skills: List[str] = Field(default_factory=list, description="Skills present in both JD and Resume")
    missing_skills: List[str] = Field(default_factory=list, description="Required skills absent from Resume")
    recommended_skills: List[str] = Field(default_factory=list, description="Adjacent skills beneficial for role")
    skill_match_percentage: float = Field(..., description="Percentage of JD skills fulfilled")


class RecommendationResult(BaseModel):
    missing_keywords: List[str] = Field(default_factory=list, description="Keywords to insert into resume")
    suggested_certifications: List[str] = Field(default_factory=list, description="Certifications to bridge skill gap")
    suggested_projects: List[str] = Field(default_factory=list, description="Hands-on projects to demonstrate mastery")
    resume_improvement_recommendations: List[str] = Field(default_factory=list, description="Actionable bullet point advice")


class SingleEvaluationRequest(BaseModel):
    resume_text: Optional[str] = Field(default=None, description="Raw text of resume (if not uploading file)")
    job_description_text: str = Field(..., description="Raw text of job description")
    jd_title: Optional[str] = Field(default="Target Role", description="Job position title")
    experience_required: Optional[float] = Field(default=None, description="Experience in years required")


class EvaluationResponse(BaseModel):
    evaluation_id: Optional[int] = None
    candidate: CandidateParsed
    job_description: JobDescriptionParsed
    score_breakdown: ATSScoreBreakdown
    skill_gap: SkillGapAnalysis
    recommendations: RecommendationResult
    candidate_rank: Optional[int] = None
    created_at: Optional[datetime] = None


class BatchEvaluationResponse(BaseModel):
    batch_id: str
    total_candidates: int
    job_title: str
    rankings: List[EvaluationResponse]
    created_at: datetime = Field(default_factory=get_utc_now)


class ModelBenchmarkItem(BaseModel):
    method: str
    latency_ms: float
    similarity_score: float
    description: str
    pros: List[str]
    cons: List[str]


class ModelComparisonResponse(BaseModel):
    timestamp: datetime = Field(default_factory=get_utc_now)
    selected_method: str
    justification: str
    benchmarks: List[ModelBenchmarkItem]



class UserRegisterRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=4, description="User password")
    role: str = Field(default="candidate", description="User role: 'candidate' or 'recruiter'")
    company_name: Optional[str] = Field(default=None, description="Company name for recruiters")


class UserLoginRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class UserResponse(BaseModel):
    id: int
    email: str
    role: str
    company_name: Optional[str] = None
    created_at: datetime


class CandidateComparisonItem(BaseModel):
    name: str
    email: Optional[str] = None
    ats_score: float
    match_percentage: float
    keyword_score: float
    semantic_score: float
    experience_score: float
    education_score: float
    certification_score: float
    experience_years: float
    highest_education: str
    matched_skills: List[str]
    missing_skills: List[str]


class CandidateCompareRequest(BaseModel):
    evaluation_ids: List[int] = Field(..., description="List of evaluation IDs to compare side-by-side")


class CandidateComparisonResponse(BaseModel):
    job_title: str
    total_compared: int
    candidates: List[CandidateComparisonItem]


