from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from src.graph.state import ResearchAssistantState
from src.rag.retriever import get_retriever
from src.tools.search_tool import search_tool
from src.logger.logger import logger
from langsmith import traceable
import os


@traceable(run_type="llm")
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)


@traceable
def route_question(state: ResearchAssistantState) -> ResearchAssistantState:
    question = state.get("question", "")
    logger.info(f"[Router] Analyzing: {question}")

    router_prompt = PromptTemplate.from_template("""
                You are a routing classifier.

                Rules:
                    - rag → answer comes only from internal documents/PDFs/company data
                    - web_search → answer needs public or current information
                    - both → answer requires internal documents plus public/current information

                Return exactly one token:

                    rag
                    web_search
                    both

                    Question: {question}
                    """)

    chain = router_prompt | get_llm() | StrOutputParser()
    route = chain.invoke({"question": question}).strip().lower()

    if route not in ["rag", "web_search", "both"]:
        route = "both"

    return {"route": route}


@traceable
def retrieve(state: ResearchAssistantState) -> ResearchAssistantState:
    question = state.get("question", "")
    logger.info("[Retriever] Fetching from Vector Store...")
    try:
        retriever = get_retriever()
        docs = retriever.invoke(question)
        context = [doc.page_content for doc in docs]
        sources = [doc.metadata.get("source", "Knowledge Base") for doc in docs]
        return {"context": context, "sources": sources}
    except Exception as e:
        logger.error(f"RAG Error: {e}")
        return {"context": [], "sources": ["Retrieval failed"]}


@traceable
def generate(state: ResearchAssistantState) -> ResearchAssistantState:
    """The LLM takes everything found so far and creates a refined response."""
    question = state.get("question", "")
    context = state.get("context", [])
    sources = state.get("sources", [])

    logger.info("[Generator] Refining final response with LLM synthesis...")

    generate_prompt = PromptTemplate.from_template("""
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

Refined Answer:
""")

    context_str = "\n\n---\n\n".join(context) if context else "No context available."
    chain = generate_prompt | get_llm() | StrOutputParser()
    generation = chain.invoke({
    "context": context_str,
    "sources": ", ".join(sources),
    "question": question
})

    return {"generation": generation}


@traceable
def build_workflow():
    workflow = StateGraph(ResearchAssistantState)

    workflow.add_node("router", route_question)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("web_search", search_tool)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("router")

    # Initial routing
    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {"rag": "retrieve", "web_search": "web_search", "both": "retrieve"},
    )

    # After retrieval decide where to go
    workflow.add_conditional_edges(
        "retrieve",
        lambda state: state["route"],
        {"rag": "generate", "both": "web_search"},
    )

    workflow.add_edge("web_search", "generate")

    workflow.add_edge("generate", END)

    return workflow.compile()


workflow = build_workflow()
