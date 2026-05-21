"""
LangChain agent setup for multi-step reasoning over documents.
"""

import json
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


def _normalize_text_tokens(text: str) -> List[str]:
    """Tokenize text to lowercase word tokens for lightweight lexical matching."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _result_key(result: Dict[str, Any]) -> str:
    """Build a stable key to deduplicate retrieval results across steps."""
    metadata = result.get("metadata", {})
    pmid = str(metadata.get("pmid", "")).strip()
    chunk_idx = str(metadata.get("chunk_idx", "")).strip()
    title = str(metadata.get("title", "")).strip()
    content_head = str(result.get("content", ""))[:120]
    return f"{pmid}|{chunk_idx}|{title}|{content_head}"


def _to_structured_evidence(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert retrieval output into a compact, structured evidence item."""
    metadata = result.get("metadata", {})
    snippet = re.sub(r"\s+", " ", str(result.get("content", ""))).strip()[:320]
    return {
        "pmid": str(metadata.get("pmid", "N/A")),
        "title": str(metadata.get("title", "N/A")),
        "pub_date": str(metadata.get("pub_date", "N/A")),
        "source_type": str(metadata.get("source_type", "N/A")),
        "score": float(result.get("score", 1.0)),
        "snippet": snippet,
    }


def _merge_unique_results(existing: List[Dict[str, Any]], new_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge retrieval result lists while keeping first-seen order."""
    merged = list(existing)
    seen = {_result_key(item) for item in merged}
    for item in new_items:
        key = _result_key(item)
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def _rerank_results(question: str, results: List[Dict[str, Any]], top_n: int = 8) -> List[Dict[str, Any]]:
    """Rerank results using vector distance and lexical overlap, then keep top_n."""
    if not results:
        return []

    q_tokens = set(_normalize_text_tokens(question))
    if not q_tokens:
        return results[:top_n]

    scored = []
    for item in results:
        evidence = _to_structured_evidence(item)
        text = f"{evidence['title']} {evidence['snippet']}"
        d_tokens = set(_normalize_text_tokens(text))
        overlap = len(q_tokens.intersection(d_tokens)) / max(len(q_tokens), 1)
        # Chroma returns lower distance as better; convert into bounded relevance signal.
        vector_relevance = 1.0 / (1.0 + max(evidence["score"], 0.0))
        combined = (0.7 * vector_relevance) + (0.3 * overlap)
        scored.append((combined, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:top_n]]


def _format_structured_context(results: List[Dict[str, Any]]) -> str:
    """Serialize evidence into JSON for more reliable grounded synthesis."""
    payload = [_to_structured_evidence(item) for item in results]
    return json.dumps(payload, ensure_ascii=True, indent=2)


def _parse_json_object(text: str) -> Dict[str, Any]:
    """Extract and parse first JSON object from model output."""
    content = (text or "").strip()
    try:
        loaded = json.loads(content)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass

    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        return {}
    try:
        loaded = json.loads(match.group(0))
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        return {}
    return {}


def _propose_next_query(
    llm: ChatOpenAI,
    original_question: str,
    current_query: str,
    latest_results: List[Dict[str, Any]],
) -> str:
    """Suggest one refined retrieval query to improve evidence coverage."""
    snippets = []
    for item in latest_results[:3]:
        metadata = item.get("metadata", {})
        pmid = metadata.get("pmid", "N/A")
        title = metadata.get("title", "N/A")
        snippets.append(f"PMID:{pmid} | {title}")
    evidence_preview = "\n".join(snippets) if snippets else "No evidence yet."

    prompt = (
        f"Original user question: {original_question}\n"
        f"Current retrieval query: {current_query}\n"
        f"Evidence preview:\n{evidence_preview}\n\n"
        "Task: Propose ONE improved PubMed-style retrieval query to find complementary, highly relevant evidence.\n"
        "Rules:\n"
        "- Return only the refined query text.\n"
        "- Keep it concise (max 12 words).\n"
        "- Avoid quotes and extra commentary."
    )

    response = llm.invoke(
        [
            SystemMessage(content="You optimize biomedical search queries."),
            HumanMessage(content=prompt),
        ]
    )
    return str(getattr(response, "content", "")).strip().replace("\n", " ")


def _evidence_sufficiency_gate(
    llm: ChatOpenAI,
    question: str,
    reranked_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Check whether current evidence is sufficient; return gate decision and missing focus."""
    context = _format_structured_context(reranked_results[:6])
    gate_prompt = (
        f"Question: {question}\n\n"
        f"Evidence JSON:\n{context}\n\n"
        "Decide if this evidence is sufficient for a high-quality grounded answer. "
        "Return JSON only with keys: sufficient (true/false), missing_focus (string), rationale (string)."
    )

    response = llm.invoke(
        [
            SystemMessage(content="You are an evidence sufficiency checker for biomedical QA."),
            HumanMessage(content=gate_prompt),
        ]
    )
    parsed = _parse_json_object(str(getattr(response, "content", "")))
    if not parsed:
        return {"sufficient": True, "missing_focus": "", "rationale": "fallback-accept"}
    parsed["sufficient"] = bool(parsed.get("sufficient", False))
    parsed["missing_focus"] = str(parsed.get("missing_focus", "")).strip()
    parsed["rationale"] = str(parsed.get("rationale", "")).strip()
    return parsed


def _multi_step_retrieve(question: str, max_steps: int = 3) -> List[Dict[str, Any]]:
    """Run a small retrieve-refine loop to gather stronger evidence before answering."""
    retriever = get_retriever()
    llm = ChatOpenAI(
        openai_api_key=settings.openai_api_key,
        model=settings.model_name,
        temperature=0,
        max_tokens=120,
    )

    current_query = question
    tried_queries = {current_query.lower()}
    all_results: List[Dict[str, Any]] = []

    for _ in range(max_steps):
        step_results = retriever.retrieve(current_query)
        all_results = _merge_unique_results(all_results, step_results)

        if len(_extract_pmids(all_results)) >= 4:
            break

        next_query = _propose_next_query(
            llm=llm,
            original_question=question,
            current_query=current_query,
            latest_results=step_results,
        )
        if not next_query:
            break

        normalized = next_query.lower()
        if normalized in tried_queries:
            break

        tried_queries.add(normalized)
        current_query = next_query

    reranked = _rerank_results(question, all_results, top_n=8)
    gate = _evidence_sufficiency_gate(llm, question, reranked)

    # One controlled extra retrieval pass if evidence appears insufficient.
    if not gate.get("sufficient", False):
        missing_focus = str(gate.get("missing_focus", "")).strip()
        supplemental_seed = missing_focus if missing_focus else question
        supplemental_query = _propose_next_query(
            llm=llm,
            original_question=question,
            current_query=supplemental_seed,
            latest_results=reranked,
        )
        if supplemental_query and supplemental_query.lower() not in tried_queries:
            supplemental = retriever.retrieve(supplemental_query)
            all_results = _merge_unique_results(all_results, supplemental)
            reranked = _rerank_results(question, all_results, top_n=8)

    return reranked


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
        # Minimal agentic loop: retrieve -> refine query -> retrieve (up to 3 steps).
        retrieval_results = _multi_step_retrieve(question, max_steps=3)
        if retrieval_results:
            pmids = _extract_pmids(retrieval_results)
            structured_context = _format_structured_context(retrieval_results)

            llm = ChatOpenAI(
                openai_api_key=settings.openai_api_key,
                model=settings.model_name,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )

            grounded_prompt = (
                f"Question: {question}\n\n"
                f"Evidence from PubMed documents (JSON):\n{structured_context}\n\n"
                "Instructions:\n"
                "- Answer based on the evidence above.\n"
                "- Cite supporting claims inline using [PMID:xxxxx].\n"
                "- Use only PMIDs that appear in the provided evidence.\n"
                "- Do not use numeric citation markers like [1], [2], or [3].\n"
                "- If studies disagree, explicitly mention the disagreement.\n"
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
