"""
Database module for AI Resume Screening & ATS Score Predictor.
Uses SQLite and SQLAlchemy ORM for persisting candidates, JDs, and ATS evaluation records.
"""

import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
from typing import Generator, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

# Path to persistent SQLite DB
DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "ats_screening.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_utc_now() -> datetime:
    """Returns the current UTC datetime without deprecation."""
    return datetime.now(timezone.utc)


def hash_password(password: str) -> str:
    """Hashes a password using PBKDF2-HMAC-SHA256 with a unique salt."""
    salt = secrets.token_hex(16)
    pw_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000).hex()
    return f"{salt}${pw_hash}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against the stored salt$hash."""
    try:
        salt, expected_hash = hashed_password.split("$", 1)
        actual_hash = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt.encode("utf-8"), 100000).hex()
        return secrets.compare_digest(actual_hash, expected_hash)
    except Exception:
        return False


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="candidate")  # 'candidate' or 'recruiter'
    company_name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=get_utc_now)



class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(255), nullable=False, default="Anonymous")
    email = Column(String(255), nullable=True, index=True)
    phone = Column(String(50), nullable=True)
    linkedin = Column(String(255), nullable=True)
    github = Column(String(255), nullable=True)
    experience_years = Column(Float, default=0.0)
    highest_education = Column(String(100), nullable=True)
    detected_skills = Column(Text, default="[]")  # JSON string
    certifications = Column(Text, default="[]")    # JSON string
    raw_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=get_utc_now)

    evaluations = relationship("Evaluation", back_populates="candidate", cascade="all, delete-orphan")


class JobDescription(Base):
    __tablename__ = "job_descriptions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String(255), nullable=False, default="Software Professional")
    company = Column(String(255), nullable=True, default="Company")
    experience_required = Column(Float, default=0.0)
    min_education = Column(String(100), nullable=True)
    required_skills = Column(Text, default="[]")  # JSON string
    raw_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=get_utc_now)

    evaluations = relationship("Evaluation", back_populates="job_description", cascade="all, delete-orphan")


class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)
    job_description_id = Column(Integer, ForeignKey("job_descriptions.id"), nullable=False)

    ats_score = Column(Float, nullable=False)
    match_percentage = Column(Float, nullable=False)

    # 5-factor breakdown
    keyword_score = Column(Float, nullable=False)
    semantic_score = Column(Float, nullable=False)
    experience_score = Column(Float, nullable=False)
    education_score = Column(Float, nullable=False)
    certification_score = Column(Float, nullable=False)

    matched_skills = Column(Text, default="[]")      # JSON string
    missing_skills = Column(Text, default="[]")      # JSON string
    recommended_skills = Column(Text, default="[]")  # JSON string
    recommendations = Column(Text, default="{}")     # JSON string

    batch_id = Column(String(100), nullable=True, index=True)
    candidate_rank = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=get_utc_now)

    candidate = relationship("Candidate", back_populates="evaluations")
    job_description = relationship("JobDescription", back_populates="evaluations")


def init_db():
    """Initializes the database schema."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency for obtaining a SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_evaluation_record(
    db: Session,
    candidate_data: dict,
    jd_data: dict,
    scores: dict,
    skill_gap: dict,
    recommendations: dict,
    batch_id: Optional[str] = None,
    rank: Optional[int] = None,
) -> Evaluation:
    """Helper to persist a full screening evaluation record to the database."""
    # 1. Upsert / Create Candidate
    candidate = Candidate(
        name=candidate_data.get("name", "Unknown Candidate"),
        email=candidate_data.get("email"),
        phone=candidate_data.get("phone"),
        linkedin=candidate_data.get("linkedin"),
        github=candidate_data.get("github"),
        experience_years=float(candidate_data.get("experience_years", 0.0)),
        highest_education=candidate_data.get("highest_education", "Not specified"),
        detected_skills=json.dumps(candidate_data.get("skills", [])),
        certifications=json.dumps(candidate_data.get("certifications", [])),
        raw_text=candidate_data.get("raw_text", ""),
    )
    db.add(candidate)
    db.flush()

    # 2. Upsert / Create Job Description
    jd = JobDescription(
        title=jd_data.get("title", "Position Opening"),
        company=jd_data.get("company", "Hiring Team"),
        experience_required=float(jd_data.get("experience_required", 0.0)),
        min_education=jd_data.get("min_education", "Bachelor's Degree"),
        required_skills=json.dumps(jd_data.get("skills", [])),
        raw_text=jd_data.get("raw_text", ""),
    )
    db.add(jd)
    db.flush()

    # 3. Create Evaluation Record
    eval_rec = Evaluation(
        candidate_id=candidate.id,
        job_description_id=jd.id,
        ats_score=float(scores.get("ats_score", 0.0)),
        match_percentage=float(scores.get("match_percentage", 0.0)),
        keyword_score=float(scores.get("keyword_score", 0.0)),
        semantic_score=float(scores.get("semantic_score", 0.0)),
        experience_score=float(scores.get("experience_score", 0.0)),
        education_score=float(scores.get("education_score", 0.0)),
        certification_score=float(scores.get("certification_score", 0.0)),
        matched_skills=json.dumps(skill_gap.get("matched_skills", [])),
        missing_skills=json.dumps(skill_gap.get("missing_skills", [])),
        recommended_skills=json.dumps(skill_gap.get("recommended_skills", [])),
        recommendations=json.dumps(recommendations),
        batch_id=batch_id,
        candidate_rank=rank,
    )
    db.add(eval_rec)
    db.commit()
    db.refresh(eval_rec)
    return eval_rec


def get_all_evaluations(db: Session, limit: int = 50) -> List[Evaluation]:
    """Fetch recent evaluations ordered by date descending."""
    return db.query(Evaluation).order_by(Evaluation.created_at.desc()).limit(limit).all()


def get_evaluation_by_id(db: Session, eval_id: int) -> Optional[Evaluation]:
    """Retrieve an evaluation record by ID."""
    return db.query(Evaluation).filter(Evaluation.id == eval_id).first()


def get_evaluations_by_batch(db: Session, batch_id: str) -> List[Evaluation]:
    """Retrieve all evaluation records for a specific batch run."""
    return (
        db.query(Evaluation)
        .filter(Evaluation.batch_id == batch_id)
        .order_by(Evaluation.ats_score.desc())
        .all()
    )


def create_user(
    db: Session,
    email: str,
    password: str,
    role: str = "candidate",
    company_name: Optional[str] = None
) -> User:
    """Registers a new candidate or recruiter user."""
    normalized_email = email.strip().lower()
    existing = db.query(User).filter(User.email == normalized_email).first()
    if existing:
        raise ValueError(f"User with email '{normalized_email}' already exists.")
    user = User(
        email=normalized_email,
        hashed_password=hash_password(password),
        role=role.strip().lower(),
        company_name=company_name.strip() if company_name else None,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticates user credentials."""
    normalized_email = email.strip().lower()
    user = db.query(User).filter(User.email == normalized_email).first()
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Finds user by email."""
    return db.query(User).filter(User.email == email.strip().lower()).first()


def get_evaluations_for_candidate(db: Session, email: str, limit: int = 10) -> List[Evaluation]:
    """Retrieves evaluations matching candidate email."""
    return (
        db.query(Evaluation)
        .join(Candidate)
        .filter(Candidate.email == email.strip().lower())
        .order_by(Evaluation.created_at.desc())
        .limit(limit)
        .all()
    )


# Automatically initialize tables on import
try:
    init_db()
except Exception as e:
    print(f"[Warning] Database initialization deferred: {e}")

