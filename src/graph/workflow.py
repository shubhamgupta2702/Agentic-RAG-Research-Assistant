from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.graph.state import ResearchAssistantState
from src.rag.retriever import get_retriever
from src.tools.search_tool import search_tool
from src.utils.utils import get_llm
from src.logger.logger import logger
from langsmith import traceable
import os


# ---------------------------------------------------------------------------
# Prompt templates — defined once at module level, never rebuilt per call
# ---------------------------------------------------------------------------

ROUTER_PROMPT = PromptTemplate.from_template("""
You are a routing classifier.

Rules:
    - rag         → answer comes only from internal documents/PDFs/company data
    - web_search  → answer needs public or current information
    - both        → answer requires internal documents plus public/current information

Return exactly one token (no punctuation, no explanation):

    rag
    web_search
    both

Question: {question}
""")

GENERATE_PROMPT = PromptTemplate.from_template("""
You are an advanced Research Assistant.

Context:
{context}

Sources:
{sources}

Question:
{question}

Rules:
1. Prioritize internal RAG data over web information.
2. Cite source origins where possible.
3. If context is insufficient, say so explicitly.
4. Do NOT fabricate facts; only use what is in Context.

Refined Answer:
""")

GRADE_PROMPT = PromptTemplate.from_template("""
You are a grader checking whether an answer is grounded in the provided context.

Context:
{context}

Answer:
{generation}

Does the answer use information from the context, or does it introduce facts not present there?
Return exactly one token:

    grounded
    not_grounded

Verdict:
""")

# Maximum characters fed into the LLM to avoid context-window overflow
_MAX_CONTEXT_CHARS = 12_000


# ---------------------------------------------------------------------------
# Node: route_question
# ---------------------------------------------------------------------------

@traceable
def route_question(state: ResearchAssistantState) -> ResearchAssistantState:
    question = state.get("question", "").strip()
    logger.info(f"[Router] Analyzing: {question}")

    chain = ROUTER_PROMPT | get_llm() | StrOutputParser()
    route = chain.invoke({"question": question}).strip().lower()

    # Sanitise — default to "rag" (cheaper) rather than "both"
    if route not in ("rag", "web_search", "both"):
        logger.warning(f"[Router] Unexpected route '{route}', defaulting to 'rag'")
        route = "rag"

    logger.info(f"[Router] Route decided: {route}")
    return {"route": route}


# ---------------------------------------------------------------------------
# Node: retrieve
# ---------------------------------------------------------------------------

@traceable
def retrieve(state: ResearchAssistantState) -> ResearchAssistantState:
    question = state.get("question", "")
    logger.info("[Retriever] Fetching from Vector Store…")

    try:
        retriever = get_retriever()
        docs = retriever.invoke(question)
        context = [doc.page_content for doc in docs]
        sources = [doc.metadata.get("source", "Knowledge Base") for doc in docs]
        logger.info(f"[Retriever] Retrieved {len(docs)} document(s).")
    except Exception as e:
        logger.error(f"[Retriever] RAG error: {e}")
        context = []
        sources = ["Retrieval failed"]

    return {"context": context, "sources": sources}


# ---------------------------------------------------------------------------
# Node: generate
# ---------------------------------------------------------------------------

@traceable
def generate(state: ResearchAssistantState) -> ResearchAssistantState:
    """Synthesise a final response from all collected context."""
    question   = state.get("question", "")
    context    = state.get("context", [])
    sources    = state.get("sources", [])

    logger.info("[Generator] Synthesising final response…")

    # Deduplicate sources while preserving order
    unique_sources = list(dict.fromkeys(sources))

    # Cap context length to avoid LLM context-window overflow
    context_str = "\n\n---\n\n".join(context) if context else "No context available."
    context_str = context_str[:_MAX_CONTEXT_CHARS]

    chain = GENERATE_PROMPT | get_llm() | StrOutputParser()
    generation = chain.invoke({
        "context":  context_str,
        "sources":  ", ".join(unique_sources),
        "question": question,
    })

    return {"generation": generation}


# ---------------------------------------------------------------------------
# Node: grade_generation  (Corrective-RAG quality gate)
# ---------------------------------------------------------------------------

@traceable
def grade_generation(state: ResearchAssistantState) -> ResearchAssistantState:
    """
    Check whether the generated answer is actually grounded in the retrieved
    context.  Sets state['grade'] to 'grounded' | 'not_grounded'.
    """
    context    = state.get("context", [])
    generation = state.get("generation", "")

    logger.info("[Grader] Checking answer grounding…")

    context_str = "\n\n---\n\n".join(context) if context else "No context available."
    context_str = context_str[:_MAX_CONTEXT_CHARS]

    chain  = GRADE_PROMPT | get_llm() | StrOutputParser()
    grade  = chain.invoke({"context": context_str, "generation": generation}).strip().lower()

    if grade not in ("grounded", "not_grounded"):
        logger.warning(f"[Grader] Unexpected grade '{grade}', defaulting to 'grounded'")
        grade = "grounded"

    logger.info(f"[Grader] Grade: {grade}")
    return {"grade": grade}


# ---------------------------------------------------------------------------
# Routing helpers (used in add_conditional_edges)
# ---------------------------------------------------------------------------

def _route_after_router(state: ResearchAssistantState) -> str:
    return state["route"]


def _route_after_retrieve(state: ResearchAssistantState) -> str:
    """After RAG retrieval, either go to web_search (both) or straight to generate."""
    return state["route"]   # "rag" → generate | "both" → web_search


def _route_after_grade(state: ResearchAssistantState) -> str:
    """
    If the answer is grounded → END.
    If not grounded → trigger web_search as a fallback enrichment step,
    then regenerate.  Prevents infinite loops: we only retry once because
    the second pass always routes to generate → END regardless of grade.
    """
    retries = state.get("retry_count", 0)
    grade   = state.get("grade", "grounded")

    if grade == "grounded" or retries >= 1:
        return "end"
    return "retry"


# ---------------------------------------------------------------------------
# Workflow builder
# ---------------------------------------------------------------------------

@traceable
def build_workflow():
    workflow = StateGraph(ResearchAssistantState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    workflow.add_node("router",           route_question)
    workflow.add_node("retrieve",         retrieve)
    workflow.add_node("web_search",       search_tool)
    workflow.add_node("generate",         generate)
    workflow.add_node("grade_generation", grade_generation)

    # ── Entry point ────────────────────────────────────────────────────────
    workflow.set_entry_point("router")

    # ── router → retrieve | web_search ────────────────────────────────────
    workflow.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "rag":        "retrieve",
            "web_search": "web_search",
            "both":       "retrieve",
        },
    )

    # ── retrieve → generate (rag) | web_search (both) ─────────────────────
    workflow.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {
            "rag":  "generate",
            "both": "web_search",
        },
    )

    # ── web_search → generate (always) ────────────────────────────────────
    workflow.add_edge("web_search", "generate")

    # ── generate → grade ──────────────────────────────────────────────────
    workflow.add_edge("generate", "grade_generation")

    # ── grade → END (grounded) | web_search retry (not_grounded) ──────────
    workflow.add_conditional_edges(
        "grade_generation",
        _route_after_grade,
        {
            "end":   END,
            "retry": "web_search",   # enrich with web, then regenerate once
        },
    )

    return workflow.compile()


# ---------------------------------------------------------------------------
# Lazy singleton — only built when first accessed, not at import time
# ---------------------------------------------------------------------------

_workflow_instance = None


def get_workflow():
    """Return the compiled workflow, building it on first call."""
    global _workflow_instance
    if _workflow_instance is None:
        logger.info("[Workflow] Building workflow graph…")
        _workflow_instance = build_workflow()
    return _workflow_instance


# Keep `workflow` as a module-level name for backward compatibility,
# but resolve it lazily so importing this module never crashes.
class _LazyWorkflow:
    """Proxy that forwards all attribute access to the real compiled graph."""

    def __getattr__(self, name):
        return getattr(get_workflow(), name)

    def __call__(self, *args, **kwargs):
        return get_workflow()(*args, **kwargs)


workflow = _LazyWorkflow()