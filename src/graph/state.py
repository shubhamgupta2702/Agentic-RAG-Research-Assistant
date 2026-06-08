from typing import TypedDict, List, Annotated
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ResearchAssistantState(TypedDict, total=False):
    """
    State payload passed between LangGraph nodes.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    
    question: str

    context: Annotated[List[str], operator.add]

    route: str 
    
    generation: str

    sources: Annotated[List[str], operator.add]
    
    grade: str
    retry_count: int