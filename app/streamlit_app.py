from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.workflow import InterviewAgents
from core.llm import LLMConfigurationError, LLMRequestError, LLMClient
from core.pdf_parser import ResumeParseError, parse_resume
from core.rag import InterviewRetriever
from core.reporting import build_readiness_pdf
from core.schemas import AnswerEvaluation, InterviewQuestion, ResumeAnalysis, ResumeProfile


st.set_page_config(page_title="PrepPilot | AI Interview Coach", page_icon="◆", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root { --ink:#183B56; --muted:#536471; --teal:#0B7285; --mint:#E8F1F5; --orange:#FFB703; }
.block-container { padding-top: 2rem; max-width: 1180px; }
.hero { padding: 2rem 2.2rem; border-radius: 22px; background: linear-gradient(120deg,#183B56 0%,#0B7285 100%); color: white; margin-bottom: 1.3rem; }
.hero h1 { color: white; margin: 0 0 .3rem 0; font-size: 2.35rem; }
.hero p { color:#DDEEF2; font-size:1.06rem; margin:0; }
.kicker { text-transform:uppercase; letter-spacing:.12em; font-size:.73rem; font-weight:700; color:#A8DADC; }
.metric-card { padding:1rem; border-radius:14px; background:#F4F8FA; border:1px solid #DDE7EC; }
.metric-card h3 { color:var(--ink); margin:.2rem 0; font-size:1.65rem; }
.metric-card p { color:var(--muted); margin:0; font-size:.85rem; }
.badge { display:inline-block; padding:.25rem .55rem; margin:.15rem; border-radius:999px; background:#E8F1F5; color:#183B56; font-size:.78rem; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


def load_streamlit_secrets() -> None:
    for key in ("LLM_PROVIDER", "GROQ_API_KEY", "GROQ_MODEL", "GEMINI_API_KEY", "GEMINI_MODEL"):
        if key not in os.environ:
            try:
                if key in st.secrets:
                    os.environ[key] = str(st.secrets[key])
            except Exception:
                pass


@st.cache_resource(show_spinner=False)
def get_retriever() -> InterviewRetriever:
    with (ROOT / "data" / "knowledge_base.json").open(encoding="utf-8") as handle:
        return InterviewRetriever(json.load(handle))


def init_state() -> None:
    defaults = {"resume": None, "role": "", "company": "", "job_description": "", "analysis": None, "questions": [], "current_idx": 0, "evaluations": [], "answers": {}}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def metric(label: str, value: str, hint: str) -> None:
    st.markdown(f'<div class="metric-card"><p>{label}</p><h3>{value}</h3><p>{hint}</p></div>', unsafe_allow_html=True)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## PrepPilot")
        st.caption("AI-powered interview practice for placement preparation")
        st.divider()
        provider = os.getenv("LLM_PROVIDER", "groq").upper()
        status = "Connected" if LLMClient().configured else "Demo mode"
        st.markdown(f"**Provider:** `{provider}`  \n**Status:** `{status}`")
        st.info("Your resume is processed in the active app session. Do not upload confidential documents you are not comfortable sending to your configured model provider.")
        st.divider()
        st.markdown("**Portfolio skills demonstrated**")
        for item in ["PDF parsing", "RAG retrieval", "LangGraph workflow", "LLM API integration", "Prompt engineering", "Evaluation UX"]:
            st.markdown(f"<span class='badge'>{item}</span>", unsafe_allow_html=True)
        if st.button("Start a new session", use_container_width=True):
            for key in ["resume", "role", "company", "job_description", "analysis", "questions", "current_idx", "evaluations", "answers"]:
                st.session_state[key] = [] if key in {"questions", "evaluations"} else ({} if key == "answers" else (0 if key == "current_idx" else None if key in {"resume", "analysis"} else ""))
            st.rerun()


def render_setup(agents: InterviewAgents) -> None:
    st.markdown("### 1. Build your interview brief")
    st.write("Upload a text-based PDF resume, add the role you want, and optionally paste the job description for targeted gap analysis.")
    uploaded = st.file_uploader("Resume PDF", type=["pdf"], help="Text-based PDFs work best. Scanned image-only PDFs need OCR first.")
    col1, col2 = st.columns(2)
    with col1:
        role = st.text_input("Target role", placeholder="e.g., AI/ML Engineer")
    with col2:
        company = st.text_input("Target company (optional)", placeholder="e.g., product startup")
    jd = st.text_area("Job description (optional)", height=150, placeholder="Paste the job description to make skill-gap analysis and questions more specific.")
    st.session_state.role, st.session_state.company, st.session_state.job_description = role, company, jd
    if st.button("Analyze my resume", type="primary", use_container_width=True, disabled=uploaded is None or not role.strip()):
        try:
            with st.spinner("Parsing your resume and mapping it to the target role..."):
                resume = parse_resume(uploaded.getvalue(), uploaded.name)
                analysis = agents.analyze(resume, role.strip(), company.strip(), jd.strip())
                st.session_state.resume = resume
                st.session_state.analysis = analysis
                st.session_state.questions = []
                st.session_state.evaluations = []
                st.session_state.current_idx = 0
            st.success("Resume analysis is ready.")
        except ResumeParseError as exc:
            st.error(str(exc))
        except (LLMConfigurationError, LLMRequestError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected error while analyzing the resume: {exc}")


def render_analysis() -> None:
    analysis: ResumeAnalysis = st.session_state.analysis
    resume: ResumeProfile = st.session_state.resume
    st.markdown("### 2. Resume fit analysis")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric("Role fit", f"{analysis.fit_score}/100", "LLM-assisted estimate")
    with c2: metric("Skills found", str(len(analysis.matched_skills)), "Matched to target")
    with c3: metric("Priority gaps", str(len(analysis.missing_skills)), "Worth addressing")
    with c4: metric("Projects detected", str(len(resume.projects)), "From uploaded resume")
    st.write(analysis.recruiter_summary)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Strengths")
        for item in analysis.strengths or ["No strengths returned yet."]: st.markdown(f"- {item}")
        st.markdown("#### Matched skills")
        st.write(", ".join(analysis.matched_skills) or "No explicit matches returned.")
    with right:
        st.markdown("#### Missing or under-evidenced skills")
        for item in analysis.missing_skills or ["No major gaps identified from the available context."]: st.markdown(f"- {item}")
        st.markdown("#### Improvement plan")
        for item in analysis.improvement_plan or ["Add quantified impact and deployment details to project bullets."]: st.markdown(f"- {item}")


def render_practice(agents: InterviewAgents) -> None:
    st.markdown("### 3. Personalized interview practice")
    if not st.session_state.questions:
        if st.button("Generate 9 personalized questions", type="primary"):
            with st.spinner("Retrieving relevant patterns and creating your practice set..."):
                try:
                    st.session_state.questions = agents.generate_questions(st.session_state.resume, st.session_state.role, st.session_state.company, st.session_state.job_description)
                except (LLMConfigurationError, LLMRequestError) as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Could not generate questions: {exc}")
            st.rerun()
        return

    questions: list[InterviewQuestion] = st.session_state.questions
    idx = min(st.session_state.current_idx, len(questions) - 1)
    question = questions[idx]
    st.progress((idx + 1) / len(questions), text=f"Question {idx + 1} of {len(questions)}")
    st.markdown(f"#### {question.question}")
    st.caption(f"{question.category.title()} · {question.difficulty.title()} · Why it matters: {question.why_asked}")
    with st.expander("See the answer framework", expanded=False):
        st.write(question.answer_framework)
    answer = st.text_area("Your answer", value=st.session_state.answers.get(idx, ""), height=220, key=f"answer_{idx}", placeholder="Use a specific example. For behavioral questions, structure it as Situation, Task, Action, Result.")
    st.session_state.answers[idx] = answer
    if st.button("Evaluate this answer", type="primary", disabled=len(answer.strip()) < 20):
        with st.spinner("Coaching your answer..."):
            try:
                evaluation = agents.evaluate(question, answer)
                existing = {i: item for i, item in enumerate(st.session_state.evaluations)}
                existing[idx] = evaluation
                st.session_state.evaluations = [existing[i] for i in sorted(existing)]
                st.session_state[f"evaluation_{idx}"] = evaluation
            except (LLMConfigurationError, LLMRequestError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not evaluate the answer: {exc}")
    evaluation: AnswerEvaluation | None = st.session_state.get(f"evaluation_{idx}")
    if evaluation:
        st.divider()
        a, b, c, d = st.columns(4)
        with a: metric("Overall", f"{evaluation.overall_score}/100", "Answer quality")
        with b: metric("Clarity", f"{evaluation.clarity_score}/10", "Communication")
        with c: metric("Correctness", f"{evaluation.correctness_score}/10", "Content")
        with d: metric("Structure", f"{evaluation.structure_score}/10", "Organization")
        st.markdown(f"**Coach feedback:** {evaluation.feedback}")
        l, r = st.columns(2)
        with l:
            st.markdown("**What worked**")
            for item in evaluation.strengths: st.markdown(f"- {item}")
        with r:
            st.markdown("**Make it stronger**")
            for item in evaluation.improvements: st.markdown(f"- {item}")
    prev_col, next_col = st.columns(2)
    with prev_col:
        if st.button("← Previous", disabled=idx == 0, use_container_width=True):
            st.session_state.current_idx -= 1
            st.rerun()
    with next_col:
        if st.button("Next →", disabled=idx == len(questions) - 1, use_container_width=True):
            st.session_state.current_idx += 1
            st.rerun()


def render_report() -> None:
    st.markdown("### 4. Readiness report")
    count = len(st.session_state.evaluations)
    avg = round(sum(item.overall_score for item in st.session_state.evaluations) / count) if count else None
    if avg is not None:
        metric("Practice readiness", f"{avg}/100", f"Based on {count} evaluated answer(s)")
    else:
        st.info("Evaluate at least one answer to see your practice readiness score.")
    pdf = build_readiness_pdf(st.session_state.resume, st.session_state.role, st.session_state.company, st.session_state.analysis, st.session_state.questions, st.session_state.evaluations)
    st.download_button("Download readiness report (PDF)", data=pdf, file_name="prep_pilot_readiness_report.pdf", mime="application/pdf", type="primary")


load_streamlit_secrets()
init_state()
render_sidebar()
st.markdown('<div class="hero"><div class="kicker">Placement preparation · AI/ML portfolio project</div><h1>PrepPilot</h1><p>A grounded, resume-aware interview coach that turns your projects into confident answers.</p></div>', unsafe_allow_html=True)
retriever = get_retriever()
agents = InterviewAgents(retriever)
if st.session_state.resume is None:
    render_setup(agents)
else:
    render_analysis()
    st.divider()
    render_practice(agents)
    st.divider()
    render_report()
