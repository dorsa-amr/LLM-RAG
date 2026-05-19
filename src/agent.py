"""
LangChain agent setup for multi-step reasoning over documents.
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from src.config import settings
from src.retriever import get_retriever


class RetrievalTool:
    """Custom tool for document retrieval."""
    
    def __init__(self):
        self.retriever = get_retriever()
    
    def search_documents(self, query: str) -> str:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            
        Returns:
            Formatted string with retrieved documents
        """
        results = self.retriever.retrieve(query)
        
        if not results:
            return "No relevant documents found in the knowledge base. Answer using general knowledge if possible."
        
        formatted = "Retrieved relevant scientific articles:\n\n"
        for i, result in enumerate(results, 1):
            pmid = result['metadata'].get('pmid', 'N/A')
            title = result['metadata'].get('title', 'N/A')
            pub_date = result['metadata'].get('pub_date', 'N/A')
            
            formatted += f"[{i}] PMID:{pmid} | {title} ({pub_date})\n"
            formatted += f"Excerpt: {result['content'][:250]}...\n\n"
        
        return formatted


def create_qa_agent(memory_enabled: bool = True):
    """
    Create a question-answering agent with retrieval capabilities.
    
    Args:
        memory_enabled: Whether to enable conversation memory
        
    Returns:
        Initialized agent
    """
    
    llm = ChatOpenAI(
        openai_api_key=settings.openai_api_key,
        model=settings.model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    retrieval = RetrievalTool()

    @tool("document_search")
    def document_search(query: str) -> str:
        """Search for relevant scientific documents from PubMed by semantic similarity."""
        return retrieval.search_documents(query)

    @tool("reasoning")
    def reasoning(note: str) -> str:
        """Use this tool to structure and refine intermediate reasoning steps."""
        return note

    agent = create_agent(
        model=llm,
        tools=[document_search, reasoning],
        system_prompt=(
            "You are a biomedical QA assistant with access to PubMed articles. "
            "ALWAYS use document_search first to find relevant scientific evidence. "
            "If documents are retrieved, your answer MUST cite them using [PMID:xxxxx] format. "
            "If no documents are found, use your general knowledge but note that citations are unavailable. "
            "Be concise, accurate, and prioritize evidence from the retrieved documents."
        ),
        debug=False,
    )

    return agent


def query_agent(agent, question: str) -> str:
    """
    Query the agent with a question.
    
    Args:
        agent: The initialized agent
        question: User's question
        
    Returns:
        Agent's response
    """
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            return getattr(last, "content", str(last))
        return str(result)
    except Exception as e:
        return f"Error during agent execution: {str(e)}"
