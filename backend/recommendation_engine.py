"""
Recommendation Engine Module.
Generates skill gap recommendations, suggested certifications,
portfolio project ideas, and structural resume enhancements.
"""

from typing import Any, Dict, List, Set


class RecommendationEngine:
    """Production recommendation engine to guide candidates and recruiters."""

    # Skill to Certification Mapping
    SKILL_CERT_MAP = {
        "aws": "AWS Certified Solutions Architect - Associate",
        "amazon web services": "AWS Certified Solutions Architect - Associate",
        "azure": "Microsoft Certified: Azure Solutions Architect Expert",
        "microsoft azure": "Microsoft Certified: Azure Solutions Architect Expert",
        "gcp": "Google Cloud Professional Cloud Architect",
        "google cloud platform": "Google Cloud Professional Cloud Architect",
        "kubernetes": "Certified Kubernetes Administrator (CKA)",
        "docker": "Docker Certified Associate (DCA)",
        "terraform": "HashiCorp Certified: Terraform Associate",
        "machine learning": "AWS Certified Machine Learning - Specialty or TensorFlow Developer Certificate",
        "deep learning": "Deep Learning Specialization (DeepLearning.AI)",
        "nlp": "Natural Language Processing Specialization (DeepLearning.AI)",
        "project management": "Project Management Professional (PMP)",
        "scrum": "Certified ScrumMaster (CSM)",
        "agile": "PMI Agile Certified Practitioner (PMI-ACP)",
        "security": "CompTIA Security+ or CISSP",
        "sql": "Oracle Certified Professional: MySQL Database Administrator",
        "data engineering": "Google Cloud Professional Data Engineer"
    }

    # Skill to Project Ideas Mapping
    SKILL_PROJECT_MAP = {
        "fastapi": "Develop a high-throughput async REST API microservice with FastAPI, Pydantic v2, and JWT authentication.",
        "docker": "Containerize a multi-service application with Docker Compose, implementing multi-stage builds and health checks.",
        "kubernetes": "Deploy and manage a microservices cluster on Kubernetes (Minikube/EKS) with Helm charts and Ingress controllers.",
        "machine learning": "Build and deploy an end-to-end ML classification/regression pipeline with Scikit-Learn, MLflow tracking, and FastAPI serving.",
        "nlp": "Create a Retrieval-Augmented Generation (RAG) system using Sentence-Transformers, ChromaDB vector database, and LangChain.",
        "react": "Build a responsive dashboard using React, Redux Toolkit, and Tailwind CSS with real-time WebSocket updates.",
        "aws": "Architect a serverless event-driven workflow using AWS Lambda, S3, SQS, and DynamoDB with Terraform IaC.",
        "ci/cd": "Create a GitHub Actions CI/CD pipeline featuring automated unit testing, linting, Docker image pushing, and staging deployment.",
        "kafka": "Implement a real-time event streaming pipeline using Apache Kafka with producers, consumers, and topic partitioning.",
        "postgresql": "Design an optimized PostgreSQL schema with indexes, connection pooling, and migration management using Alembic."
    }

    # Skill to Adjacent Recommendations
    SKILL_ADJACENT_MAP = {
        "python": ["FastAPI", "PyTest", "Docker", "Poetry"],
        "react": ["TypeScript", "Next.js", "Tailwind CSS"],
        "machine learning": ["MLflow", "ONNX", "FastAPI", "Docker"],
        "docker": ["Kubernetes", "Helm", "Terraform", "GitHub Actions"],
        "aws": ["Terraform", "CloudWatch", "Lambda", "IAM"],
        "fastapi": ["Pydantic", "Uvicorn", "PostgreSQL", "Redis"],
        "sql": ["PostgreSQL", "Redis", "Database Indexing", "SQLAlchemy"]
    }

    def get_recommended_skills(
        self, matched_skills: List[str], missing_skills: List[str]
    ) -> List[str]:
        """Suggests adjacent, high-demand skills that complement the target role."""
        adjacent_suggestions: Set[str] = set()
        combined = set(matched_skills + missing_skills)

        for skill in combined:
            lower = skill.lower()
            if lower in self.SKILL_ADJACENT_MAP:
                for adj in self.SKILL_ADJACENT_MAP[lower]:
                    if adj.lower() not in {s.lower() for s in combined}:
                        adjacent_suggestions.add(adj)

        return sorted(list(adjacent_suggestions))[:6]

    def get_suggested_certifications(self, missing_skills: List[str]) -> List[str]:
        """Identifies industry certifications to directly bridge detected skill gaps."""
        suggested_certs: Set[str] = set()

        for skill in missing_skills:
            lower = skill.lower()
            for key, cert in self.SKILL_CERT_MAP.items():
                if key in lower:
                    suggested_certs.add(cert)

        # Fallback general recommendation if no domain-specific cert matched
        if not suggested_certs:
            if any("cloud" in s.lower() for s in missing_skills):
                suggested_certs.add("AWS Certified Solutions Architect - Associate")
            elif any("data" in s.lower() or "ml" in s.lower() for s in missing_skills):
                suggested_certs.add("Databricks Certified Data Engineer or AWS Machine Learning")
            else:
                suggested_certs.add("HashiCorp Certified Terraform Associate")

        return sorted(list(suggested_certs))[:4]

    def get_suggested_projects(self, missing_skills: List[str]) -> List[str]:
        """Recommends portfolio projects demonstrating mastery of missing skills."""
        suggested_projs: List[str] = []

        for skill in missing_skills:
            lower = skill.lower()
            for key, proj in self.SKILL_PROJECT_MAP.items():
                if key in lower and proj not in suggested_projs:
                    suggested_projs.append(proj)

        # Default fallback projects if none triggered
        if not suggested_projs:
            suggested_projs.append(
                "Build an end-to-end full-stack web application featuring RESTful API integration, containerization, and unit tests."
            )
            suggested_projs.append(
                "Create a cloud-deployed microservice with automated GitHub Actions CI/CD pipeline and monitoring."
            )

        return suggested_projs[:3]

    def get_structural_recommendations(
        self,
        candidate_data: Dict[str, Any],
        ats_score: float,
        missing_skills: List[str]
    ) -> List[str]:
        """
        Generates actionable resume formatting and structural recommendations
        following modern ATS best practices.
        """
        tips: List[str] = []

        # 1. Action Verbs & X-Y-Z formula
        tips.append(
            "Utilize the Google 'X-Y-Z' formula for bullet points: 'Accomplished [X] as measured by [Y], by doing [Z]' (e.g., 'Reduced API latency by 35% by implementing Redis caching')."
        )

        # 2. Missing Skills In-Context
        if missing_skills:
            top_missing = ", ".join(missing_skills[:5])
            tips.append(
                f"Directly integrate top missing keywords ({top_missing}) into past job responsibilities and project descriptions rather than listing them solely in a skills summary."
            )

        # 3. Experience Quantification
        raw_text = candidate_data.get("raw_text", "")
        has_metrics = bool(any(char in raw_text for char in ["%", "$", "M", "K"]) or any(w in raw_text.lower() for w in ["reduced", "increased", "saved", "improved", "scaled"]))
        if not has_metrics:
            tips.append(
                "Incorporate quantifiable metrics (e.g., revenue generated, throughput handled, deployment frequency increase, percentage error reduction) across each work experience entry."
            )

        # 4. Standard Headings
        sections = candidate_data.get("sections", {})
        if "experience" not in sections and "work experience" not in str(sections).lower():
            tips.append(
                "Ensure standard ATS section headers are used (e.g., 'Work Experience', 'Technical Skills', 'Education', 'Projects') to avoid misclassification by automated parsers."
            )

        # 5. Length & Contact Advice
        if len(raw_text.split()) > 1000:
            tips.append(
                "Your resume appears longer than 2 pages. Consider condensing older experience entries to keep key qualifications prominent and within 1-2 pages."
            )

        # 6. ATS Score specific advice
        if ats_score < 70.0:
            tips.append(
                "Tailor your resume summary and skill bullets to mirror the exact terminology and phrasing found in the target Job Description."
            )
        else:
            tips.append(
                "Strong candidate alignment! Double-check that your LinkedIn and GitHub links are active and reflect the achievements showcased in your resume."
            )

        return tips

    def generate_full_recommendations(
        self,
        candidate_data: Dict[str, Any],
        ats_score: float,
        matched_skills: List[str],
        missing_skills: List[str]
    ) -> Dict[str, Any]:
        """Aggregates all recommendations into a unified payload."""
        return {
            "missing_keywords": missing_skills,
            "suggested_certifications": self.get_suggested_certifications(missing_skills),
            "suggested_projects": self.get_suggested_projects(missing_skills),
            "recommended_skills": self.get_recommended_skills(matched_skills, missing_skills),
            "resume_improvement_recommendations": self.get_structural_recommendations(
                candidate_data, ats_score, missing_skills
            ),
        }


# Global recommendation engine instance
recommendation_engine = RecommendationEngine()
