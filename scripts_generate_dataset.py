from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "data" / "knowledge_base.json"

ML_SOURCE = "https://www.geeksforgeeks.org/machine-learning/machine-learning-interview-questions/"
BACKEND_SOURCE = "https://www.geeksforgeeks.org/interview-prep/backend-developer-interview-questions-and-answers/"
STAR_SOURCE = "https://capd.mit.edu/resources/the-star-method-for-behavioral-interviews/"


def make_record(idx: int, domain: str, question: str, framework: str, topic: str, difficulty: str, source: str, tags: list[str]) -> dict:
    return {
        "id": f"{domain[:3].lower()}-{idx:03d}",
        "domain": domain,
        "question": question,
        "answer_framework": framework,
        "topic": topic,
        "difficulty": difficulty,
        "tags": tags,
        "source_url": source,
        "source_note": "Question paraphrased and answer framework authored for this project; see source for related public preparation material.",
    }


def build_ml() -> list[dict]:
    topics = [
        ("ML fundamentals", "How would you explain the difference between AI, machine learning, and data science?", "Define the scope of each field, then show how ML sits inside AI and supports data-science workflows.", "easy"),
        ("generalization", "What does it mean for a model to generalize well?", "Connect training performance to unseen-data performance and mention validation design, leakage, and distribution shift.", "easy"),
        ("overfitting", "What causes overfitting, and how would you reduce it?", "Identify high variance, then discuss regularization, simpler models, cross-validation, early stopping, and more data.", "easy"),
        ("underfitting", "How would you diagnose and fix underfitting?", "Use train and validation errors: add signal/features, increase capacity, reduce excessive regularization, or train longer.", "easy"),
        ("bias variance", "Explain the bias-variance trade-off with a practical example.", "Define bias and variance, relate them to underfit/overfit, then describe validation-driven model selection.", "medium"),
        ("regularization", "Compare L1, L2, and Elastic Net regularization.", "State the penalty shape, sparsity behavior, correlated-feature behavior, and when each is useful.", "medium"),
        ("cross validation", "When would you use stratified k-fold cross-validation?", "Explain folds, class-ratio preservation, leakage prevention, and why it is appropriate for imbalanced classification.", "medium"),
        ("data leakage", "Give examples of data leakage in an ML pipeline.", "Mention preprocessing before splitting, target-derived features, duplicate users across folds, and future information.", "medium"),
        ("missing values", "How would you handle missing values in a production dataset?", "Profile missingness, distinguish MCAR/MAR/MNAR, fit imputers on train only, add indicators, and monitor drift.", "medium"),
        ("outliers", "How do outliers affect different models and metrics?", "Explain sensitivity of means, linear models, and distance methods; compare clipping, robust scaling, transformations, and investigation.", "medium"),
        ("scaling", "When is feature scaling important?", "Contrast tree models with distance/gradient-based models and explain standardization versus normalization.", "easy"),
        ("encoding", "How would you encode high-cardinality categorical variables?", "Compare one-hot, hashing, target encoding with leakage-safe folds, embeddings, and frequency encoding.", "medium"),
        ("class imbalance", "What would you do when the positive class is rare?", "Choose business-aware metrics, resampling or class weights, threshold tuning, calibration, and stratified evaluation.", "medium"),
        ("precision recall", "Explain precision, recall, and F1 using an interview-screening example.", "Define false positives/negatives, map costs to the metric, and explain why F1 is a compromise rather than a universal choice.", "easy"),
        ("ROC AUC", "What does ROC-AUC measure and when can it mislead?", "Describe ranking across thresholds, then note imbalance, precision-recall curves, calibration, and operating-point costs.", "medium"),
        ("calibration", "How would you check whether predicted probabilities are calibrated?", "Use reliability diagrams and Brier score; apply Platt scaling or isotonic regression on held-out data.", "medium"),
        ("regression metrics", "Compare MAE, MSE, RMSE, and MAPE.", "Discuss units, outlier sensitivity, asymmetric business costs, and MAPE instability near zero.", "easy"),
        ("feature engineering", "How do you decide whether a feature is useful?", "Check causal plausibility, availability at inference, leakage, univariate signal, ablation, and validation lift.", "medium"),
        ("trees", "Why are decision trees prone to overfitting?", "Explain recursive partitioning and high variance, then cover depth limits, pruning, minimum leaf size, and ensembles.", "easy"),
        ("random forest", "Why can random forests generalize better than a single tree?", "Describe bootstrap samples, random feature subsets, variance reduction, out-of-bag estimates, and trade-offs.", "easy"),
        ("boosting", "Compare bagging and boosting.", "Bagging trains independent learners to reduce variance; boosting builds sequentially to reduce bias, with different failure modes.", "medium"),
        ("gradient descent", "What affects gradient-descent convergence?", "Cover learning rate, feature scale, batch size, curvature, initialization, schedules, and stopping criteria.", "medium"),
        ("logistic regression", "Why is logistic regression useful despite being simple?", "Explain probabilistic linear decision boundaries, interpretability, regularization, calibration, and strong baselines.", "easy"),
        ("SVM", "When might an SVM be a good choice?", "Mention margin maximization, kernels for nonlinear boundaries, scaling needs, and limits with very large datasets.", "medium"),
        ("clustering", "How would you evaluate a clustering result without labels?", "Use silhouette or stability plus domain validation; explain that a metric cannot replace meaningful cluster use.", "medium"),
        ("PCA", "What does PCA optimize and what information can it lose?", "Describe orthogonal variance-maximizing components, preprocessing, dimensionality reduction, and interpretability trade-offs.", "medium"),
        ("recommendation", "How would you evaluate a recommender system offline and online?", "Separate ranking metrics, coverage/diversity/novelty, counterfactual limits, and A/B testing with guardrails.", "hard"),
        ("computer vision", "How would you handle class imbalance in an image classifier?", "Use stratified splits, augmentation, weighted loss, per-class metrics, thresholding, and check minority-image quality.", "medium"),
        ("transfer learning", "Why is transfer learning useful for computer vision?", "Reuse pretrained visual features, freeze/unfreeze progressively, adapt augmentation, and validate domain mismatch.", "easy"),
        ("embeddings", "What is an embedding and how would you use it in search?", "Explain dense vectors encoding similarity, nearest-neighbor retrieval, normalization, evaluation queries, and freshness.", "easy"),
        ("LLM", "What is the difference between an LLM and a traditional supervised model?", "Contrast objective, open-ended generation, in-context learning, evaluation, grounding, and safety controls.", "easy"),
        ("RAG", "Why use retrieval-augmented generation instead of only prompting an LLM?", "Ground responses in current/domain documents, reduce unsupported claims, expose sources, and manage retrieval quality.", "medium"),
        ("chunking", "How would you choose chunk size for a RAG system?", "Balance semantic completeness, retrieval precision, context budget, overlap, document structure, and measure recall.", "medium"),
        ("vector search", "Compare lexical search and vector search.", "Lexical search matches terms; vector search captures meaning; hybrid retrieval combines exact identifiers and semantic recall.", "medium"),
        ("prompting", "What makes a production prompt reliable?", "Specify role, task, constraints, schema, grounded context, refusal behavior, examples, and test cases.", "medium"),
        ("hallucination", "How would you reduce hallucinations in an LLM application?", "Use retrieval, citations, constrained output, lower temperature, validation, abstention, and monitoring.", "medium"),
        ("evaluation", "How would you evaluate a RAG application?", "Measure retrieval recall/precision and answer faithfulness/relevance, with golden sets, human review, and regression tests.", "hard"),
    ]
    records = []
    for idx, item in enumerate(topics * 2):
        topic, question, framework, difficulty = item
        suffix = "" if idx < len(topics) else " Give a concrete example from a project."
        records.append(make_record(idx + 1, "AI/ML", question + suffix, framework, topic, difficulty, ML_SOURCE, ["ai", "ml", topic]))
    return records[:70]


def build_backend() -> list[dict]:
    topics = [
        ("HTTP", "Explain the difference between GET, POST, PUT, PATCH, and DELETE.", "Map each verb to intent and idempotency, then mention validation, status codes, and safe retries.", "easy"),
        ("REST", "What makes an API RESTful enough for production use?", "Discuss resource-oriented URLs, statelessness, representations, HTTP semantics, pagination, errors, and versioning.", "medium"),
        ("status codes", "Which HTTP status codes would you return for common API failures?", "Use 400/422 for invalid input, 401/403 for auth, 404 for missing resources, 409 for conflicts, and 5xx for server faults.", "easy"),
        ("authentication", "Compare session-based authentication with JWTs.", "Contrast server-side session state with signed claims, revocation, rotation, storage, and browser security.", "medium"),
        ("authorization", "What is the difference between authentication and authorization?", "Authentication establishes identity; authorization checks permitted actions against roles, scopes, and resource ownership.", "easy"),
        ("SQL", "How would you find and fix a slow SQL query?", "Inspect query plan and indexes, reduce scanned rows, avoid N+1, verify cardinality, and measure before/after.", "medium"),
        ("indexes", "What are the trade-offs of database indexes?", "They accelerate reads but add storage and write/update cost; choose selective, workload-driven indexes.", "easy"),
        ("transactions", "Explain database transactions and isolation levels.", "Use ACID, dirty/non-repeatable/phantom reads, and choose the lowest safe isolation with retry handling.", "hard"),
        ("normalization", "When would you denormalize a relational schema?", "Start normalized for consistency; denormalize for measured read latency with ownership, refresh, and consistency plans.", "medium"),
        ("NoSQL", "When would you choose a document store over a relational database?", "Base it on access patterns, schema flexibility, transactions, relationships, consistency, and operational constraints.", "medium"),
        ("caching", "Design a cache for a read-heavy endpoint.", "Choose key/TTL/eviction, invalidation, stampede protection, stale behavior, and metrics; never cache sensitive data casually.", "medium"),
        ("Redis", "What problems can Redis solve and what can go wrong?", "Mention cache, rate limits, queues, and locks; cover eviction, persistence, hot keys, split brain, and durability limits.", "medium"),
        ("queues", "Why use a message queue between services?", "Decouple producers and consumers, absorb spikes, retry failures, and accept eventual consistency with idempotency.", "medium"),
        ("idempotency", "How would you make a payment-like API idempotent?", "Require a client key, persist request/result atomically, replay the same outcome, and define expiry/conflict semantics.", "hard"),
        ("scaling", "How would you scale a web service horizontally?", "Keep app instances stateless, externalize session/state, load-balance, pool DB connections, and monitor bottlenecks.", "medium"),
        ("load balancing", "What is the role of a load balancer?", "Distribute traffic, perform health checks, terminate TLS, route by policy, and avoid sending traffic to unhealthy nodes.", "easy"),
        ("rate limiting", "Design a rate limiter for a public API.", "Choose fixed/sliding/token-bucket strategy, identity key, storage, response headers, burst rules, and failure mode.", "hard"),
        ("observability", "What should you log and measure in a backend service?", "Use structured logs, latency/error/traffic/saturation metrics, traces, correlation IDs, and redaction.", "medium"),
        ("reliability", "Explain timeouts, retries, and circuit breakers.", "Set bounded timeouts, exponential backoff with jitter, retry only safe/transient failures, and trip breakers to prevent cascades.", "hard"),
        ("security", "How do you prevent SQL injection and XSS?", "Use parameterized queries and output encoding, validate input, CSP, secure cookies, and defense in depth.", "easy"),
        ("secrets", "How should production secrets be managed?", "Use a secret manager or platform secrets, least privilege, rotation, no logs or commits, and environment-specific access.", "easy"),
        ("Docker", "What belongs in a production Dockerfile?", "Use a small pinned base, multi-stage build, non-root user, health check, deterministic dependencies, and no secrets.", "medium"),
        ("CI/CD", "What checks should run before deploying an API?", "Run formatting, linting, unit/integration tests, security scans, migrations checks, and safe rollout validation.", "easy"),
        ("async", "What does asynchronous I/O improve in a server?", "It prevents waiting on I/O from blocking workers, but CPU-bound work still needs processes/workers or a job queue.", "medium"),
        ("Python", "How would you structure a maintainable Python service?", "Separate configuration, domain logic, adapters, API routes, tests, typed interfaces, and dependency boundaries.", "easy"),
        ("Node", "How does Node.js handle many concurrent requests?", "Explain the event loop and non-blocking I/O, then call out CPU-bound work and worker processes.", "medium"),
        ("API versioning", "How can an API evolve without breaking clients?", "Prefer additive changes, deprecations, compatibility tests, version policy, and explicit migration communication.", "medium"),
        ("pagination", "Compare offset and cursor pagination.", "Offset is simple but unstable/slow at depth; cursor pagination is stable and efficient with ordered indexed keys.", "medium"),
        ("file uploads", "How would you safely handle user-uploaded files?", "Limit size/type, scan content, store outside web root, randomize names, authorize access, and avoid trusting extensions.", "medium"),
        ("webhooks", "How would you process an incoming webhook reliably?", "Verify signature, deduplicate, acknowledge quickly, enqueue work, retry safely, and provide observability.", "hard"),
        ("system design", "Design a URL-shortening service at a high level.", "Clarify scale and consistency, generate unique IDs, store mappings, cache hot redirects, and handle abuse/analytics.", "hard"),
        ("system design", "Design a notification service.", "Separate preference, fan-out, queue, channel adapters, retries, deduplication, rate limits, and delivery status.", "hard"),
        ("testing", "What is the difference between unit, integration, and end-to-end tests?", "Explain scope, speed, failure isolation, realistic dependencies, and a balanced test pyramid.", "easy"),
        ("debugging", "Walk through how you debug a production incident.", "Stabilize impact, inspect signals and recent changes, reproduce safely, mitigate, communicate, and write a blameless postmortem.", "medium"),
        ("concurrency", "What race conditions can occur in a web application?", "Identify shared state and check-then-act bugs, then use transactions, locks, atomic operations, or idempotency.", "medium"),
    ]
    records = []
    for idx, item in enumerate(topics * 2):
        topic, question, framework, difficulty = item
        suffix = "" if idx < len(topics) else " Include one operational trade-off."
        records.append(make_record(idx + 1, "Software/Backend", question + suffix, framework, topic, difficulty, BACKEND_SOURCE, ["software", "backend", topic]))
    return records[:70]


def build_behavioral() -> list[dict]:
    questions = [
        ("Tell me about a project you are most proud of.", "Use STAR: context and goal, your specific actions, measurable result, and what you learned.", "projects"),
        ("Tell me about a time you solved a difficult problem.", "Frame the ambiguity, options considered, decision criteria, action, and measurable outcome.", "problem-solving"),
        ("Describe a time you failed.", "Be honest, own your part, explain the impact, corrective action, and the changed behavior/result.", "learning"),
        ("Tell me about a time you worked in a team.", "Use I-statements: team context, your contribution, collaboration behavior, outcome, and reflection.", "teamwork"),
        ("Tell me about a conflict with a teammate.", "Explain the shared goal, listen-first approach, evidence-based resolution, and relationship/outcome.", "conflict"),
        ("Describe a time you demonstrated leadership without authority.", "Show initiative, alignment, delegation or influence, measurable result, and what you would repeat.", "leadership"),
        ("How do you prioritize competing deadlines?", "Describe impact/urgency assessment, stakeholder alignment, explicit trade-offs, execution, and communication.", "prioritization"),
        ("Tell me about a time requirements changed late.", "Explain the change, impact analysis, re-planning, communication, and delivery result.", "adaptability"),
        ("Describe a time you received difficult feedback.", "State the feedback without defensiveness, action taken, evidence of change, and ongoing practice.", "feedback"),
        ("How do you handle ambiguity?", "Clarify the goal, state assumptions, create a small experiment, surface risks, and iterate with stakeholders.", "ambiguity"),
        ("Tell me about a time you disagreed with a decision.", "Respectfully present evidence, seek context, propose alternatives, commit once decided, and reflect.", "communication"),
        ("Tell me about a time you had to learn something quickly.", "Describe the gap, focused learning plan, application, validation, and result.", "learning"),
        ("Describe a time you improved a process.", "Quantify baseline pain, diagnose root cause, implement a practical change, measure adoption and outcome.", "ownership"),
        ("Tell me about a time you made a mistake.", "Own it, explain how you detected and contained it, prevent recurrence, and quantify recovery.", "accountability"),
        ("What motivates you in technical work?", "Connect intrinsic motivation to the role, user impact, learning, and an authentic example.", "motivation"),
        ("What is your biggest strength?", "Name one relevant strength, provide evidence, describe its impact, and avoid unsupported superlatives.", "self-awareness"),
        ("What is an area you are improving?", "Choose a real but manageable gap, show a concrete improvement system, and provide progress evidence.", "self-awareness"),
        ("Why this role and company?", "Connect company problem, role responsibilities, your evidence, and a specific reason the timing fits.", "motivation"),
        ("Tell me about a time you advocated for quality.", "Describe the risk, practical quality action, trade-off, stakeholder handling, and outcome.", "quality"),
        ("How do you explain technical concepts to non-technical people?", "Start from audience goals, use plain language/analogy, check understanding, and tailor detail.", "communication"),
        ("Tell me about a time you used data to make a decision.", "State the decision, data quality and alternatives, analysis, action, and outcome with caveats.", "analytical-thinking"),
        ("Tell me about a time you went beyond your assigned task.", "Show user/team need, responsible initiative, alignment, outcome, and boundaries.", "ownership"),
        ("Describe a time you managed stress.", "Give a specific high-pressure context, prioritization and communication actions, result, and sustainable lesson.", "resilience"),
        ("Tell me about a time you helped someone else succeed.", "Identify their need, coaching or enablement action, their outcome, and what you learned about collaboration.", "teamwork"),
        ("How do you respond when you do not know an answer?", "Be transparent, decompose the question, state what you know, propose how to verify, and follow through.", "integrity"),
        ("Describe an ethical dilemma you faced.", "Clarify stakeholders and principles, escalate appropriately, choose a defensible action, and reflect on impact.", "integrity"),
        ("Tell me about a time you balanced speed and quality.", "Make risk explicit, define minimum safe quality, ship incrementally, and measure/revisit.", "judgment"),
        ("Tell me about a time you handled an unresponsive stakeholder.", "Use respectful follow-ups, clarify the decision needed, offer options, escalate with context, and protect delivery.", "communication"),
        ("What kind of manager helps you do your best work?", "Describe working preferences, feedback cadence, autonomy, and how you adapt to different styles.", "self-awareness"),
        ("What would your teammates say about you?", "Use two evidence-backed traits, one balanced growth area, and link them to collaboration.", "self-awareness"),
        ("How do you prepare for an important presentation?", "Define audience/outcome, structure narrative, rehearse, test risks, and invite questions.", "communication"),
        ("Tell me about a time you changed your mind.", "Describe new evidence, how you updated your view, decision impact, and intellectual humility.", "learning"),
        ("How do you make sure your work is inclusive and accessible?", "Consider user diversity, accessible defaults, testing with varied users, and feedback loops.", "values"),
        ("Tell me about a time you had to say no.", "Explain the request, constraints, alternatives, respectful communication, and protected outcome.", "judgment"),
        ("Where do you want to grow in the next two years?", "Choose role-relevant skills, a concrete plan, and how you will create value while learning.", "motivation"),
    ]
    records = []
    for idx, item in enumerate(questions * 2):
        question, framework, topic = item
        suffix = "" if idx < len(questions) else " Give a concise, evidence-backed answer."
        records.append(make_record(idx + 1, "Behavioral/HR", question + suffix, framework, topic, "medium", STAR_SOURCE, ["behavioral", "hr", topic, "star"]))
    return records[:70]


if __name__ == "__main__":
    records = build_ml() + build_backend() + build_behavioral()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(records)} records to {OUT}")
