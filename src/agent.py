"""
LangChain agent setup for multi-step reasoning over documents.
"""

import re
from typing import Any, Dict, List

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
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


def _extract_pmids(results: List[Dict[str, Any]]) -> List[str]:
    """Extract unique PMIDs from retrieval results."""
    seen = set()
    pmids = []
    for result in results:
        pmid = str(result.get("metadata", {}).get("pmid", "")).strip()
        if pmid and pmid != "N/A" and pmid not in seen:
            seen.add(pmid)
            pmids.append(pmid)
    return pmids


def _format_retrieval_context(results: List[Dict[str, Any]]) -> str:
    """Create a compact context block for grounded answer generation."""
    chunks = []
    for result in results:
        metadata = result.get("metadata", {})
        pmid = metadata.get("pmid", "N/A")
        title = metadata.get("title", "N/A")
        excerpt = result.get("content", "")[:350]
        chunks.append(f"PMID:{pmid} | {title}\n{excerpt}")
    return "\n\n".join(chunks)


def _normalize_inline_citations(answer: str, pmids: List[str]) -> str:
    """Convert numeric citations like [1] into [PMID:xxxxx] and drop unmatched numeric refs."""
    if not pmids:
        return answer

    def repl(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        if 1 <= idx <= len(pmids):
            return f"[PMID:{pmids[idx - 1]}]"
        return ""

    normalized = re.sub(r"\[(\d+)\]", repl, answer)
    # Collapse accidental double spaces left after removing unmatched numeric citations.
    normalized = re.sub(r" {2,}", " ", normalized)
    return normalized


def _ensure_citations(answer: str, pmids: List[str]) -> str:
    """Guarantee at least one PMID citation in grounded answers."""
    answer = _normalize_inline_citations(answer, pmids)

    if not pmids:
        return answer
    if any(f"[PMID:{pmid}]" in answer for pmid in pmids):
        return answer
    fallback = ", ".join(f"[PMID:{pmid}]" for pmid in pmids[:3])
    return f"{answer.rstrip()}\n\nSources: {fallback}"


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
        # Use a deterministic, grounded path when retrieval has evidence.
        retrieval_results = get_retriever().retrieve(question)
        if retrieval_results:
            pmids = _extract_pmids(retrieval_results)
            context = _format_retrieval_context(retrieval_results)

            llm = ChatOpenAI(
                openai_api_key=settings.openai_api_key,
                model=settings.model_name,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )

            grounded_prompt = (
                f"Question: {question}\n\n"
                f"Evidence from PubMed documents:\n{context}\n\n"
                "Instructions:\n"
                "- Answer based on the evidence above.\n"
                "- Cite supporting claims inline using [PMID:xxxxx].\n"
                "- Use only PMIDs that appear in the provided evidence.\n"
                "- Do not use numeric citation markers like [1], [2], or [3].\n"
                "- If evidence is insufficient, state uncertainty clearly."
            )

            response = llm.invoke(
                [
                    SystemMessage(content="You are a biomedical QA assistant that writes concise, evidence-grounded answers."),
                    HumanMessage(content=grounded_prompt),
                ]
            )
            content = getattr(response, "content", str(response))
            return _ensure_citations(content, pmids)

        # Fallback path when no retrieval evidence is available.
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            return getattr(last, "content", str(last))
        return str(result)
    except Exception as e:
        return f"Error during agent execution: {str(e)}"
