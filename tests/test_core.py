from __future__ import annotations

import io
import json

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from agents.workflow import InterviewAgents
from core.pdf_parser import ResumeParseError, extract_resume_text, parse_resume
from core.rag import InterviewRetriever
from core.schemas import InterviewQuestion, ResumeProfile


@pytest.fixture
def resume_pdf() -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.drawString(72, 780, "Meet Sharma")
    pdf.drawString(72, 760, "meet@example.com")
    pdf.drawString(72, 720, "Skills")
    pdf.drawString(72, 700, "Python, Pandas, scikit-learn, Streamlit")
    pdf.drawString(72, 660, "Projects")
    pdf.drawString(72, 640, "Agentic Music Recommender")
    pdf.drawString(72, 600, "Education")
    pdf.drawString(72, 580, "MSc Artificial Intelligence")
    pdf.save()
    return buffer.getvalue()


def test_resume_parser_extracts_profile(resume_pdf: bytes) -> None:
    profile = parse_resume(resume_pdf)
    assert profile.name == "Meet Sharma"
    assert profile.email == "meet@example.com"
    assert any("Python" in skill for skill in profile.skills)
    assert any("Music" in project for project in profile.projects)


def test_empty_and_image_only_inputs_fail() -> None:
    with pytest.raises(ResumeParseError):
        extract_resume_text(b"")
    with pytest.raises(ResumeParseError):
        extract_resume_text(b"not a real PDF")


def test_rag_returns_relevant_domain() -> None:
    records = [
        {"id": "1", "domain": "AI/ML", "question": "What is overfitting?", "answer_framework": "Discuss generalization and regularization", "topic": "overfitting"},
        {"id": "2", "domain": "Software/Backend", "question": "How do database indexes work?", "answer_framework": "Discuss read speed and write cost", "topic": "indexes"},
    ]
    retriever = InterviewRetriever(records)
    results = retriever.search("regularization to prevent overfitting", k=1)
    assert results[0]["id"] == "1"
    assert "similarity" in results[0]


def test_agents_have_offline_fallback() -> None:
    records = [{"id": "1", "domain": "AI/ML", "question": "Explain RAG", "answer_framework": "Retrieve and ground", "topic": "RAG"}]
    agents = InterviewAgents(InterviewRetriever(records))
    resume = ResumeProfile(name="Candidate", skills=["Python"], projects=["ML project"])
    analysis = agents.analyze(resume, "AI/ML Engineer", "", "")
    assert 0 <= analysis.fit_score <= 100
    questions = agents.generate_questions(resume, "AI/ML Engineer", "", "")
    assert questions
    evaluation = agents.evaluate(InterviewQuestion("Explain RAG", answer_framework="Retrieve and ground"), "I retrieve relevant context and ground the answer.")
    assert 0 <= evaluation.overall_score <= 100
