from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from core.llm import LLMClient
from core.rag import InterviewRetriever
from core.schemas import AnswerEvaluation, InterviewQuestion, ResumeAnalysis, ResumeProfile


class AgentState(TypedDict, total=False):
    resume: dict[str, Any]
    role: str
    company: str
    job_description: str
    analysis: dict[str, Any]
    questions: list[dict[str, Any]]
    question: dict[str, Any]
    answer: str
    evaluation: dict[str, Any]
    context: list[dict[str, Any]]


class InterviewAgents:
    def __init__(self, retriever: InterviewRetriever, llm: LLMClient | None = None):
        self.retriever = retriever
        self.llm = llm or LLMClient()

    def _resume_context(self, resume: dict[str, Any]) -> str:
        return json.dumps({key: value for key, value in resume.items() if key != "raw_text"}, ensure_ascii=False)

    def resume_analyzer(self, state: AgentState) -> AgentState:
        resume = state["resume"]
        role = state.get("role", "AI/ML Engineer")
        jd = state.get("job_description", "")
        prompt = f"""Analyze this candidate for the target role.
Target role: {role}
Company: {state.get('company', 'Not specified')}
Job description: {jd or 'Not provided'}
Resume profile: {self._resume_context(resume)}

Return ONLY valid JSON with exactly these keys:
fit_score (integer 0-100), matched_skills (array of strings), missing_skills (array of strings), strengths (array of strings), improvement_plan (array of strings), recruiter_summary (string).
Base claims on the resume. If the job description is absent, use common expectations for the role and state that assumption."""
        fallback = {
            "fit_score": 50,
            "matched_skills": resume.get("skills", [])[:8],
            "missing_skills": [],
            "strengths": resume.get("projects", [])[:4],
            "improvement_plan": ["Add a quantified impact statement to each major project."],
            "recruiter_summary": "The profile shows relevant project experience; add a target job description for a more precise gap analysis.",
        }
        result = self.llm.generate_json(
            "You are a careful resume analyst. Do not invent candidate experience, skills, or outcomes.", prompt, max_tokens=1200
        ) if self.llm.configured else fallback
        return {**state, "analysis": result}

    def question_generator(self, state: AgentState) -> AgentState:
        resume = state["resume"]
        role = state.get("role", "AI/ML Engineer")
        query = f"{role} {state.get('company', '')} {state.get('job_description', '')} {resume.get('skills', [])}"
        context = self.retriever.search(query, k=8)
        prompt = f"""Generate 9 personalized interview questions for a candidate targeting {role} at {state.get('company', 'a company')}.
Resume profile: {self._resume_context(resume)}
Job description: {state.get('job_description', 'Not provided')}
Retrieved interview patterns: {json.dumps(context, ensure_ascii=False)}

Mix AI/ML or role-technical questions with software/backend and behavioral questions when relevant. Reference the candidate's actual projects or skills in the questions, but do not invent details.
Return ONLY valid JSON with key questions, an array of exactly 9 objects. Each object must have: question, category, difficulty, why_asked, answer_framework."""
        fallback = {"questions": [
            {"question": f"Walk me through the most relevant project on your resume for a {role} role.", "category": "project", "difficulty": "medium", "why_asked": "Tests ownership and technical communication.", "answer_framework": "Context, your design choices, obstacles, measurable result, and one lesson."},
            {"question": "How would you evaluate and monitor an ML system after deployment?", "category": "technical", "difficulty": "hard", "why_asked": "Tests production ML maturity.", "answer_framework": "Offline metrics, slices, drift, latency/cost, feedback, alerts, and rollback."},
            {"question": "Explain one trade-off you made in a recent project.", "category": "behavioral", "difficulty": "medium", "why_asked": "Tests judgment and reflection.", "answer_framework": "Goal, constraints, options, decision, result, and what you would revisit."},
        ]}
        result = self.llm.generate_json("You are an expert technical interviewer. Personalize questions strictly from the provided context.", prompt, max_tokens=2000) if self.llm.configured else fallback
        return {**state, "questions": result.get("questions", fallback["questions"]), "context": context}

    def answer_evaluator(self, state: AgentState) -> AgentState:
        question = state["question"]
        answer = state.get("answer", "").strip()
        context = self.retriever.search(f"{question.get('question', '')} {question.get('category', '')}", k=4)
        prompt = f"""Evaluate this interview answer fairly.
Question: {json.dumps(question, ensure_ascii=False)}
Candidate answer: {answer}
Relevant ideal-answer patterns: {json.dumps(context, ensure_ascii=False)}

Score clarity, correctness, and structure from 0 to 10, then overall_score from 0 to 100. Do not grade the candidate for things not asked. Return ONLY valid JSON with exactly: overall_score, clarity_score, correctness_score, structure_score, strengths (array), improvements (array), model_answer_outline (string), feedback (string)."""
        fallback = {"overall_score": 50, "clarity_score": 5, "correctness_score": 5, "structure_score": 5, "strengths": ["You attempted the question."], "improvements": ["Use a clear beginning, middle, and outcome."], "model_answer_outline": question.get("answer_framework", "State context, action, and result."), "feedback": "Add a specific example and quantify the result where possible."}
        result = self.llm.generate_json("You are a constructive interview coach. Be specific, evidence-based, and encouraging.", prompt, max_tokens=1200) if self.llm.configured else fallback
        return {**state, "evaluation": result}

    def build_resume_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("resume_analyzer", self.resume_analyzer)
        graph.add_edge(START, "resume_analyzer")
        graph.add_edge("resume_analyzer", END)
        return graph.compile()

    def build_question_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("question_generator", self.question_generator)
        graph.add_edge(START, "question_generator")
        graph.add_edge("question_generator", END)
        return graph.compile()

    def build_evaluation_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("answer_evaluator", self.answer_evaluator)
        graph.add_edge(START, "answer_evaluator")
        graph.add_edge("answer_evaluator", END)
        return graph.compile()

    def analyze(self, resume: ResumeProfile, role: str, company: str, job_description: str) -> ResumeAnalysis:
        state = self.build_resume_graph().invoke({"resume": resume.to_dict(), "role": role, "company": company, "job_description": job_description})
        data = state["analysis"]
        return ResumeAnalysis(**{key: data.get(key, getattr(ResumeAnalysis(), key)) for key in ResumeAnalysis().__dict__})

    def generate_questions(self, resume: ResumeProfile, role: str, company: str, job_description: str) -> list[InterviewQuestion]:
        state = self.build_question_graph().invoke({"resume": resume.to_dict(), "role": role, "company": company, "job_description": job_description})
        return [InterviewQuestion(**item) for item in state.get("questions", [])]

    def evaluate(self, question: InterviewQuestion, answer: str) -> AnswerEvaluation:
        state = self.build_evaluation_graph().invoke({"question": question.__dict__, "answer": answer})
        data = state["evaluation"]
        return AnswerEvaluation(**{key: data.get(key, getattr(AnswerEvaluation(), key)) for key in AnswerEvaluation().__dict__})
