from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResumeProfile:
    name: str = "Candidate"
    email: str = ""
    phone: str = ""
    education: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    experience: list[str] = field(default_factory=list)
    summary: str = ""
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "education": self.education,
            "skills": self.skills,
            "projects": self.projects,
            "experience": self.experience,
            "summary": self.summary,
        }


@dataclass
class ResumeAnalysis:
    fit_score: int = 0
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    improvement_plan: list[str] = field(default_factory=list)
    recruiter_summary: str = ""


@dataclass
class InterviewQuestion:
    question: str
    category: str = "technical"
    difficulty: str = "medium"
    why_asked: str = ""
    answer_framework: str = ""


@dataclass
class AnswerEvaluation:
    overall_score: int = 0
    clarity_score: int = 0
    correctness_score: int = 0
    structure_score: int = 0
    strengths: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    model_answer_outline: str = ""
    feedback: str = ""


@dataclass
class SessionState:
    resume: ResumeProfile | None = None
    role: str = ""
    company: str = ""
    job_description: str = ""
    analysis: ResumeAnalysis | None = None
    questions: list[InterviewQuestion] = field(default_factory=list)
    evaluations: list[AnswerEvaluation] = field(default_factory=list)
