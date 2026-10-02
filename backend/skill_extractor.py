"""
Skill and Qualification Extractor Module.
Leverages hierarchical taxonomy matching, regex NER, alias normalization,
and contextual rules to accurately extract skills, education, certifications, and experience.
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Set, Tuple


class SkillExtractor:
    """Production skill and qualification extractor for resumes and job descriptions."""

    def __init__(self, taxonomy_path: str = None):
        if not taxonomy_path:
            taxonomy_path = os.path.join(os.path.dirname(__file__), "..", "data", "skills_taxonomy.json")

        self.taxonomy = self._load_taxonomy(taxonomy_path)
        self.aliases: Dict[str, str] = {k.lower(): v for k, v in self.taxonomy.get("aliases", {}).items()}
        self._build_skill_lookup()

    def _load_taxonomy(self, path: str) -> Dict:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Warning] Failed to read taxonomy from {path}: {e}")

        # Fallback embedded taxonomy
        return {
            "categories": {
                "programming_languages": ["Python", "Java", "C++", "C#", "JavaScript", "TypeScript", "Go", "Rust", "SQL"],
                "frameworks_and_libraries": ["FastAPI", "Flask", "Django", "React", "Node.js", "PyTorch", "TensorFlow", "Pandas", "Scikit-Learn"],
                "cloud_and_devops": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "CI/CD", "Linux"],
                "databases_and_big_data": ["PostgreSQL", "MySQL", "MongoDB", "Redis", "Kafka", "Spark", "Snowflake"],
                "ai_and_data_science": ["Machine Learning", "Deep Learning", "NLP", "LLM", "Computer Vision", "Prompt Engineering"],
                "tools_and_practices": ["Git", "GitHub", "Jira", "Postman", "Agile", "REST API", "Microservices", "PyTest"],
                "soft_skills": ["Leadership", "Communication", "Problem Solving", "Collaboration", "Teamwork"],
                "certifications": ["AWS Certified", "Azure Solutions Architect", "CKA", "PMP"],
                "education_degrees": ["PhD", "Master of Science", "MS", "Bachelor of Science", "BS", "B.Tech", "B.E."]
            },
            "aliases": {
                "k8s": "Kubernetes",
                "postgres": "PostgreSQL",
                "js": "JavaScript",
                "ts": "TypeScript"
            }
        }

    def _build_skill_lookup(self):
        """Constructs an inverted index mapping lowercased skill names to canonical forms."""
        self.skill_to_canonical: Dict[str, str] = {}
        self.skill_to_category: Dict[str, str] = {}

        categories = self.taxonomy.get("categories", {})
        for cat_name, skill_list in categories.items():
            if cat_name in ["education_degrees"]:
                continue
            for skill in skill_list:
                clean_skill = skill.strip()
                lower_skill = clean_skill.lower()
                self.skill_to_canonical[lower_skill] = clean_skill
                self.skill_to_category[lower_skill] = cat_name

    def extract_skills(self, text: str) -> List[str]:
        """
        Extracts and normalizes skills mentioned in the given text.
        Handles short abbreviations safely (like 'C', 'R', 'Go') with strict boundaries.
        """
        if not text:
            return []

        lower_text = " " + text.lower() + " "
        detected_skills: Set[str] = set()

        # 1. Direct multi-word and single-word lookup
        for lower_skill, canonical in self.skill_to_canonical.items():
            # Special handling for single or very short tokens to prevent false positives
            if lower_skill in {"c", "r"}:
                pattern = rf"(?:\b|\s){re.escape(lower_skill)}(?:\s*,\s*|\s+programming|\s+language|\s*[/\-]\s*)"
                if re.search(pattern, lower_text):
                    detected_skills.add(canonical)
            elif lower_skill == "go":
                pattern = r"\bgo\s+language\b|\bgolang\b|\bgo\s+developer\b|\bprogramming\s+in\s+go\b"
                if re.search(pattern, lower_text):
                    detected_skills.add("Go")
            elif lower_skill == "sql":
                if re.search(r"\bsql\b", lower_text):
                    detected_skills.add("SQL")
            elif lower_skill == "r":
                if re.search(r"\br\s+programming\b|\br\s+language\b|\busing\s+r\b", lower_text):
                    detected_skills.add("R")
            else:
                # Standard regex word-boundary match
                # Escape characters like +, #, .
                pattern = rf"(?<!\w){re.escape(lower_skill)}(?!\w)"
                if re.search(pattern, lower_text):
                    detected_skills.add(canonical)

        # 2. Check aliases
        for alias, canonical in self.aliases.items():
            pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
            if re.search(pattern, lower_text):
                detected_skills.add(canonical)

        # Return sorted list for determinism
        return sorted(list(detected_skills))

    def extract_education(self, text: str) -> Tuple[str, List[str]]:
        """
        Extracts educational qualifications and determines highest achieved degree tier.
        Tiers: PhD (Rank 4) > Master's (Rank 3) > Bachelor's (Rank 2) > Diploma/Associate (Rank 1).
        """
        found_entries = []
        highest_tier = "Not specified"
        highest_rank = 0

        lower_text = text.lower()

        degree_hierarchy = [
            ("PhD / Doctorate", 4, [r"\bph\.?d\b", r"\bdoctor of philosophy\b", r"\bdoctorate\b"]),
            ("Master's Degree", 3, [r"\bmaster(?:'s)?\b", r"\bm\.?s\b", r"\bm\.?sc\b", r"\bm\.?tech\b", r"\bmba\b", r"\bmca\b"]),
            ("Bachelor's Degree", 2, [r"\bbachelor(?:'s)?\b", r"\bb\.?s\b", r"\bb\.?sc\b", r"\bb\.?tech\b", r"\bb\.?e\b", r"\bbca\b"]),
            ("Diploma / Associate", 1, [r"\bassociate(?:'s)?\b", r"\bdiploma\b"])
        ]

        for name, rank, patterns in degree_hierarchy:
            for pattern in patterns:
                matches = re.finditer(pattern, lower_text)
                for m in matches:
                    snippet = text[max(0, m.start() - 20) : min(len(text), m.end() + 40)].strip()
                    found_entries.append(snippet)
                    if rank > highest_rank:
                        highest_rank = rank
                        highest_tier = name

        return highest_tier, found_entries[:5]

    def extract_certifications(self, text: str) -> List[str]:
        """Extracts recognizable industry certifications."""
        certs = self.taxonomy.get("categories", {}).get("certifications", [])
        found_certs = set()
        lower_text = text.lower()

        for cert in certs:
            if re.search(rf"\b{re.escape(cert.lower())}\b", lower_text):
                found_certs.add(cert)

        # Additional regex patterns for generic certified credentials
        generic_patterns = [
            r"(?:aws\s+certified\s+[a-zA-Z\s]+)",
            r"(?:azure\s+certified\s+[a-zA-Z\s]+)",
            r"(?:google\s+cloud\s+certified\s+[a-zA-Z\s]+)",
            r"(?:certified\s+kubernetes\s+[a-zA-Z]+)",
            r"\bpmp\b",
            r"\bcsm\b"
        ]
        for pat in generic_patterns:
            matches = re.findall(pat, lower_text)
            for m in matches:
                clean_cert = m.strip().title()
                if len(clean_cert) > 3:
                    found_certs.add(clean_cert)

        return sorted(list(found_certs))

    def extract_experience_years(self, text: str) -> float:
        """
        Calculates or estimates total years of professional experience.
        1. Checks for explicit mentions like '5+ years of experience'.
        2. Parses date intervals like '2018 - 2023', 'Jan 2019 - Present'.
        """
        # Explicit statement check
        explicit_patterns = [
            r"(\d+(?:\.\d+)?)\s*\+?\s*years?(?:\s+of)?\s+(?:professional\s+)?experience",
            r"(?:over|more\s+than)\s+(\d+(?:\.\d+)?)\s*years",
            r"experience:\s*(\d+(?:\.\d+)?)\s*years"
        ]
        for pat in explicit_patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass

        # Date range parsing (e.g. 2018 - 2022, 2019 - Present)
        current_year = datetime.now().year
        year_ranges = re.findall(r"\b(19\d\d|20\d\d)\s*(?:-|–|to)\s*(19\d\d|20\d\d|present|current|now)\b", text, re.IGNORECASE)

        total_years = 0.0
        for start_str, end_str in year_ranges:
            try:
                start_yr = int(start_str)
                if end_str.lower() in ["present", "current", "now"]:
                    end_yr = current_year
                else:
                    end_yr = int(end_str)

                if start_yr <= end_yr <= current_year + 1:
                    duration = end_yr - start_yr
                    # Avoid adding duplicate or negative intervals
                    if 0 <= duration <= 30:
                        total_years += duration
            except Exception:
                continue

        # Cap estimated total years at 40
        return round(min(total_years, 40.0), 1)


# Global extractor instance
skill_extractor = SkillExtractor()
