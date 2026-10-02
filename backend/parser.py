"""
Document Parser Module for AI Resume Screening & ATS Score Predictor.
Extracts text from PDF, DOCX, and TXT files.
Segments resumes into structured sections and detects candidate metadata.
"""

import io
import os
import re
from typing import BinaryIO, Dict, List, Optional, Tuple, Union

# Attempt PDF readers
try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# Attempt DOCX reader
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


class DocumentParser:
    """Production-grade document parser for resumes and job descriptions."""

    SECTION_PATTERNS = {
        "summary": [
            r"\b(professional\s+summary|summary|profile|about\s+me|career\s+objective|objective)\b"
        ],
        "experience": [
            r"\b(work\s+experience|professional\s+experience|experience|employment\s+history|career\s+history|work\s+history)\b"
        ],
        "education": [
            r"\b(education|academic\s+background|academic\s+qualifications|academics|qualifications|educational\s+history)\b"
        ],
        "skills": [
            r"\b(technical\s+skills|core\s+competencies|skills|technologies|tools\s+and\s+technologies|skills\s+and\s+abilities|key\s+skills)\b"
        ],
        "projects": [
            r"\b(projects|key\s+projects|personal\s+projects|academic\s+projects|notable\s+projects)\b"
        ],
        "certifications": [
            r"\b(certifications|certificates|licenses\s+and\s+certifications|professional\s+certifications|training)\b"
        ],
    }

    EMAIL_REGEX = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    PHONE_REGEX = r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    LINKEDIN_REGEX = r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w\-_%]+"
    GITHUB_REGEX = r"(?:https?://)?(?:www\.)?github\.com/[\w\-_]+"

    def extract_text_from_pdf(self, stream_or_path: Union[str, BinaryIO, bytes]) -> str:
        """Extracts text from a PDF file with robust error recovery."""
        if isinstance(stream_or_path, str):
            with open(stream_or_path, "rb") as f:
                content = f.read()
            stream = io.BytesIO(content)
        elif isinstance(stream_or_path, bytes):
            stream = io.BytesIO(stream_or_path)
        else:
            stream = stream_or_path

        extracted_text = []

        if PYPDF_AVAILABLE:
            try:
                reader = pypdf.PdfReader(stream)
                for page_idx, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    extracted_text.append(page_text)
            except Exception as e:
                extracted_text.append(f"[PDF Extraction Error: {str(e)}]")
        else:
            extracted_text.append(
                "[Warning: pypdf library is not installed. PDF text could not be extracted directly.]"
            )

        return "\n".join(extracted_text).strip()

    def extract_text_from_docx(self, stream_or_path: Union[str, BinaryIO, bytes]) -> str:
        """Extracts paragraphs and table contents from a DOCX document."""
        if isinstance(stream_or_path, str):
            file_ref = stream_or_path
        elif isinstance(stream_or_path, bytes):
            file_ref = io.BytesIO(stream_or_path)
        else:
            file_ref = stream_or_path

        text_pieces = []
        if DOCX_AVAILABLE:
            try:
                doc = docx.Document(file_ref)
                for para in doc.paragraphs:
                    if para.text.strip():
                        text_pieces.append(para.text.strip())

                # Also extract text from tables
                for table in doc.tables:
                    for row in table.rows:
                        row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if row_text:
                            text_pieces.append(" | ".join(row_text))
            except Exception as e:
                text_pieces.append(f"[DOCX Extraction Error: {str(e)}]")
        else:
            text_pieces.append(
                "[Warning: python-docx library is not installed. DOCX text could not be extracted directly.]"
            )

        return "\n".join(text_pieces).strip()

    def extract_text(self, file_content: Union[bytes, BinaryIO, str], filename: str = "") -> str:
        """Determines file type and delegates to the appropriate extractor."""
        if isinstance(file_content, str) and not os.path.exists(file_content):
            # Already raw string content
            return file_content

        if isinstance(file_content, str) and os.path.exists(file_content) and not filename:
            filename = file_content

        ext = os.path.splitext(filename)[1].lower() if filename else ""

        if ext == ".pdf":
            return self.extract_text_from_pdf(file_content)
        elif ext in [".docx", ".doc"]:
            return self.extract_text_from_docx(file_content)
        else:
            # Assume text/plain
            if isinstance(file_content, bytes):
                for encoding in ["utf-8", "latin-1", "cp1252"]:
                    try:
                        return file_content.decode(encoding)
                    except UnicodeDecodeError:
                        continue
                return file_content.decode("utf-8", errors="ignore")
            elif isinstance(file_content, str) and os.path.exists(file_content):
                with open(file_content, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            elif isinstance(file_content, io.IOBase):
                return file_content.read().decode("utf-8", errors="ignore")
            return str(file_content)

    def extract_contact_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extracts email, phone, LinkedIn, and GitHub links via regex."""
        email_match = re.search(self.EMAIL_REGEX, text)
        phone_match = re.search(self.PHONE_REGEX, text)
        linkedin_match = re.search(self.LINKEDIN_REGEX, text, re.IGNORECASE)
        github_match = re.search(self.GITHUB_REGEX, text, re.IGNORECASE)

        return {
            "email": email_match.group(0) if email_match else None,
            "phone": phone_match.group(0) if phone_match else None,
            "linkedin": linkedin_match.group(0) if linkedin_match else None,
            "github": github_match.group(0) if github_match else None,
        }

    def extract_candidate_name(self, text: str) -> str:
        """
        Infers candidate name from the top non-empty lines using heuristics.
        ATS resumes typically feature candidate names in the first 1-3 lines.
        """
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines[:5]:
            # Skip lines containing email, phone, URLs or section headers
            if re.search(self.EMAIL_REGEX, line) or re.search(self.PHONE_REGEX, line):
                continue
            if re.search(r"https?://", line, re.IGNORECASE):
                continue
            if any(
                re.search(pat, line, re.IGNORECASE)
                for pat_list in self.SECTION_PATTERNS.values()
                for pat in pat_list
            ):
                continue

            # Check if line looks like a person's name (2-4 capitalized words, no punctuation)
            words = line.split()
            if 2 <= len(words) <= 4 and all(w.replace(".", "").isalpha() for w in words):
                return line

        return "Candidate"

    def segment_sections(self, text: str) -> Dict[str, str]:
        """
        Segments a resume into standardized structural sections:
        Summary, Experience, Education, Skills, Projects, Certifications.
        """
        lines = text.split("\n")
        sections: Dict[str, List[str]] = {
            "header": [],
            "summary": [],
            "experience": [],
            "education": [],
            "skills": [],
            "projects": [],
            "certifications": [],
            "other": [],
        }

        current_section = "header"

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Detect section header boundary
            detected_new_section = None
            if len(stripped) < 45:  # Headers are generally brief
                for sec_name, pattern_list in self.SECTION_PATTERNS.items():
                    for pattern in pattern_list:
                        if re.search(f"^{pattern}\\b", stripped, re.IGNORECASE) or re.search(
                            f"^{pattern}:?", stripped, re.IGNORECASE
                        ):
                            detected_new_section = sec_name
                            break
                    if detected_new_section:
                        break

            if detected_new_section:
                current_section = detected_new_section
                continue

            sections[current_section].append(stripped)

        return {k: "\n".join(v).strip() for k, v in sections.items() if v}

    def parse_job_description(self, jd_text: str, default_title: str = "Target Position") -> Dict:
        """
        Parses a Job Description into required qualifications, years of experience,
        and minimum degree.
        """
        clean_jd = jd_text.strip()

        # Extract Experience Requirement (e.g., "5+ years", "3-5 years of experience")
        exp_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:\+|-\s*\d+)?\s*(?:to\s*\d+)?\s*years?(?:\s+of)?\s+experience",
            clean_jd,
            re.IGNORECASE,
        )
        experience_req = float(exp_match.group(1)) if exp_match else 0.0

        # Extract Education Requirement
        min_edu = "Bachelor's Degree"
        if re.search(r"\b(phd|doctorate|ph\.d)\b", clean_jd, re.IGNORECASE):
            min_edu = "PhD"
        elif re.search(r"\b(master'?s?|ms|m\.s|m\.tech|mba)\b", clean_jd, re.IGNORECASE):
            min_edu = "Master's Degree"
        elif re.search(r"\b(bachelor'?s?|bs|b\.s|b\.tech|b\.e)\b", clean_jd, re.IGNORECASE):
            min_edu = "Bachelor's Degree"

        # Detect Job Title from first line or headers
        title = default_title
        lines = [line.strip() for line in clean_jd.split("\n") if line.strip()]
        if lines and len(lines[0]) < 60:
            title = lines[0].replace("Job Title:", "").replace("Position:", "").strip()

        return {
            "title": title,
            "raw_text": clean_jd,
            "experience_required": experience_req,
            "min_education": min_edu,
        }


# Global parser instance
document_parser = DocumentParser()
