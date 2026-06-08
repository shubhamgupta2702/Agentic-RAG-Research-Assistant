from langchain_google_genai import ChatGoogleGenerativeAI

_llm_instance = None


def get_llm() -> ChatGoogleGenerativeAI:
    """Return a cached LLM instance (singleton pattern)."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0,
        )
    return _llm_instance