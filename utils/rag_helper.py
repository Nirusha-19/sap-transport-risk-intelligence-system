# RAG engine with TF-IDF retrieval + prompt builder

from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Knowledge base ────────────────────────────────────────────────────────
KNOWLEDGE_BASE = [
    {
        "title": "HIGH risk FI transports",
        "content": (
            "Financial Accounting (FI) transports with more than 100 lines of code change "
            "carry HIGH risk because FI directly impacts general ledger, accounts payable, "
            "and financial reporting. A rollback may be impossible without data loss."
        ),
    },
    {
        "title": "Conflict impact",
        "content": (
            "Transport conflicts (two or more concurrent changes to the same object) are the "
            "strongest predictor of deployment failure. Even one conflict raises risk to MEDIUM; "
            "two or more escalates to HIGH regardless of module."
        ),
    },
    {
        "title": "History failures",
        "content": (
            "Transports from change requests with 3 or more historical failures "
            "are classified HIGH risk. Repeated failures suggest systemic design or dependency issues."
        ),
    },
    {
        "title": "Production stage policy",
        "content": (
            "All HIGH-risk transports targeting the Production system require dual sign-off from "
            "the basis team and the business owner, a mandatory downtime window, and a tested "
            "rollback script before the change window opens."
        ),
    },
    {
        "title": "MEDIUM risk mitigations",
        "content": (
            "MEDIUM-risk transports (lines_changed > 80 or one conflict) should be moved during "
            "off-peak hours, include unit-test evidence, and be monitored for 24 hours post-import."
        ),
    },
    {
        "title": "LOW risk fast-track",
        "content": (
            "LOW-risk transports (fewer than 60 lines, no conflicts, fewer than 2 failures) may "
            "be fast-tracked through Quality to Production with a single approver, reducing lead "
            "time from 5 days to same-day."
        ),
    },
    {
        "title": "MM and SD module specifics",
        "content": (
            "Materials Management (MM) transports affecting pricing procedures and SD transports "
            "touching pricing conditions require extra regression testing because they can silently "
            "alter customer-facing prices."
        ),
    },
    {
        "title": "HR module sensitivity",
        "content": (
            "Human Resources (HR) transports are GDPR-sensitive. Even LOW-risk HR changes must "
            "be reviewed by the data-privacy officer before promotion to Production."
        ),
    },
    {
        "title": "SHAP explainability",
        "content": (
            "SHAP (SHapley Additive exPlanations) values show each feature's contribution to "
            "the predicted risk. Positive SHAP pushes toward the predicted class; negative pushes away. "
            "The top drivers are conflicts, history_failures, and lines_changed."
        ),
    },
    {
        "title": "SMOTE training",
        "content": (
            "The model uses SMOTE to balance training classes, making it equally sensitive to LOW, "
            "MEDIUM, and HIGH risk instead of defaulting to the majority class."
        ),
    },
]


class RAGEngine:
    def __init__(self):
        self._texts  = [d["content"] for d in KNOWLEDGE_BASE]
        self._titles = [d["title"]   for d in KNOWLEDGE_BASE]
        self._tfidf  = TfidfVectorizer(stop_words="english")
        self._matrix = self._tfidf.fit_transform(self._texts)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        q_vec  = self._tfidf.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        idxs   = np.argsort(scores)[::-1][:top_k]
        return [
            {"title": self._titles[i], "content": self._texts[i],
             "score": float(scores[i])}
            for i in idxs if scores[i] > 0.0
        ]

    def build_prompt(self, query: str,
                     transport_context: dict | None = None,
                     shap_context: dict | None = None) -> str:
        docs = self.retrieve(query, top_k=3)
        context_block = "\n\n".join(
            f"[{d['title']}]\n{d['content']}" for d in docs
        )

        transport_block = ""
        if transport_context:
            transport_block = f"""
Transport under analysis:
  Module:           {transport_context.get('module')}
  Predicted Risk:   {transport_context.get('risk')}
  Stage:            {transport_context.get('stage')}
  Lines changed:    {transport_context.get('lines_changed')}
  Conflicts:        {transport_context.get('conflicts')}
  History failures: {transport_context.get('history_failures')}
"""

        shap_block = ""
        if shap_context:
            sorted_feats = sorted(shap_context.items(),
                                  key=lambda x: abs(x[1]), reverse=True)
            shap_lines = "\n".join(f"  {f}: {v:+.4f}"
                                   for f, v in sorted_feats[:5])
            shap_block = f"\nTop SHAP feature contributions:\n{shap_lines}\n"

        return f"""You are an SAP Basis consultant specialising in transport risk.
Use ONLY the retrieved knowledge below to answer the question.
If the knowledge does not cover it, say so honestly.

=== RETRIEVED KNOWLEDGE ===
{context_block}
{transport_block}{shap_block}
=== QUESTION ===
{query}

Provide a structured answer with:
1. Risk assessment rationale (citing which features drove the prediction)
2. Business impact
3. Recommended actions
"""

    def answer_no_llm(self, query: str) -> str:
        """Fallback when no API key is available — returns raw retrieved snippets."""
        docs = self.retrieve(query, top_k=3)
        if not docs:
            return "No relevant knowledge found."
        return "\n\n---\n\n".join(
            f"**{d['title']}** (relevance: {d['score']:.2f})\n{d['content']}"
            for d in docs
        )


# Singleton so streamlit doesn't rebuild it on every rerun
_engine: RAGEngine | None = None

def get_rag_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine
