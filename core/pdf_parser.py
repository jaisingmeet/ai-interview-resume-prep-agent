from __future__ import annotations

import io
import re
from pathlib import Path

from pypdf import PdfReader

from .schemas import ResumeProfile


class ResumeParseError(ValueError):
    """Raised when an uploaded resume cannot be converted to usable text."""


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_resume_text(file_bytes: bytes, filename: str = "resume.pdf") -> str:
    if not file_bytes:
        raise ResumeParseError("The uploaded file is empty.")
    if not filename.lower().endswith(".pdf"):
        raise ResumeParseError("Please upload a PDF resume.")
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [(page.extract_text() or "") for page in reader.pages]
        text = _clean_text("\n\n".join(pages))
    except Exception as exc:  # pypdf raises several parser-specific exceptions.
        raise ResumeParseError(f"The PDF could not be read: {exc}") from exc
    if len(text) < 40:
        raise ResumeParseError(
            "This PDF contains little or no selectable text. Please export a text-based PDF or run OCR before uploading."
        )
    return text


def parse_resume(file_bytes: bytes, filename: str = "resume.pdf") -> ResumeProfile:
    text = extract_resume_text(file_bytes, filename)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    phone_match = re.search(r"(?:\+?\d[\d ()-]{8,}\d)", text)

    heading_patterns = {
        "skills": re.compile(r"^(technical\s+skills?|skills?|technologies|tools)[:\s]*$", re.I),
        "projects": re.compile(r"^(projects?|academic\s+projects?)[:\s]*$", re.I),
        "experience": re.compile(r"^(experience|work\s+experience|employment)[:\s]*$", re.I),
        "education": re.compile(r"^(education|academic\s+background)[:\s]*$", re.I),
    }
    sections: dict[str, list[str]] = {key: [] for key in heading_patterns}
    current: str | None = None
    for line in lines:
        matched = next((key for key, pattern in heading_patterns.items() if pattern.match(line)), None)
        if matched:
            current = matched
            continue
        if current:
            sections[current].append(line)

    name = lines[0] if lines else "Candidate"
    if email_match and name == email_match.group(0):
        name = "Candidate"

    return ResumeProfile(
        name=name[:100],
        email=email_match.group(0) if email_match else "",
        phone=phone_match.group(0).strip() if phone_match else "",
        education=sections["education"][:12],
        skills=sections["skills"][:30],
        projects=sections["projects"][:15],
        experience=sections["experience"][:15],
        summary=" ".join(lines[:5])[:600],
        raw_text=text,
    )
