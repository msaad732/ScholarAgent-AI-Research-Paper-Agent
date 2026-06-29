"""LangGraph ReAct agent that autonomously orchestrates the four research tools."""
import re
import time
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, MAX_AGENT_ITERATIONS
from core.vector_store import VectorStoreManager
from tools.compare_tool import make_compare_tool
from tools.extract_tool import make_extract_tool
from tools.qa_tool import make_fetch_tool
from tools.search_tool import make_search_tool
from utils.helpers import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = f"""You are ScholarAgent, an AI research assistant with access to a knowledge base of academic papers and the ability to fetch new ones from arxiv.

Your tools:
1. search_knowledge_base — Search the existing indexed papers for relevant information. ALWAYS try this first.
2. fetch_new_papers — Search arxiv and download new papers on a topic. Use this when the knowledge base doesn't have enough information, or the user explicitly asks for more/new/recent papers.
3. compare_papers — Compare methodologies, results, or approaches across multiple papers in the knowledge base.
4. extract_metrics — Extract specific numerical results, metrics, or key findings from papers.

Rules:
- ALWAYS search the knowledge base first before fetching new papers.
- When you fetch new papers, search the knowledge base again to use them.
- Cite sources with arxiv IDs.
- If you're unsure, say so. Never fabricate information.
- Limit yourself to {MAX_AGENT_ITERATIONS} tool calls per question.
"""

_ARXIV_RE = re.compile(r"arxiv:([\w./-]+)", re.IGNORECASE)
# Bracketed blocks from search_tool / extract_tool:
#   [Paper: Title | arxiv:id | Section: X | relevance: 0.5]\n<snippet>
_BRACKET_RE = re.compile(
    r"\[(?:(?:YOUR PAPER|Paper):\s*)?(?P<title>[^|\]]+?)\s*\|\s*(?P<ident>[^|\]]+?)\s*"
    r"\|\s*Section:\s*(?P<section>[^|\]]+?)\s*(?:\|[^\]]*)?\]\s*\n(?P<snippet>.+?)"
    r"(?=\n\n---\n\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)
# Blocks from compare_tool:  Paper: Title (arxiv:id)\nRelevant excerpt: <snippet>
_COMPARE_RE = re.compile(
    r"(?:YOUR PAPER|Paper):\s*(?P<title>.+?)\s*\((?P<ident>[^)]+)\)\s*\n"
    r"Relevant excerpt:\s*(?P<snippet>.+?)(?=\n---\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)

_MAX_AGENT_RETRIES = 3  # retries when the LLM emits a malformed tool call


def _is_tool_use_error(exc: Exception) -> bool:
    """Return True if the exception is Groq's malformed-tool-call (400) error."""
    text = str(exc).lower()
    return "tool_use_failed" in text or "failed to call a function" in text


class AgentState(TypedDict):
    """LangGraph state: conversation messages plus a tool-call counter."""

    messages: Annotated[list, add_messages]
    tool_calls_count: int


def _build_llm_with_tools(tools: list):
    """Construct a ChatGroq client with the research tools bound."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file or Streamlit secrets."
        )
    from langchain_groq import ChatGroq

    llm = ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    return llm.bind_tools(tools)


def build_agent(vector_store: VectorStoreManager):
    """Build and compile the LangGraph agent bound to a vector store.

    Args:
        vector_store: Initialised vector store manager shared by all tools.

    Returns:
        A compiled LangGraph graph ready to invoke.
    """
    tools = [
        make_search_tool(vector_store),
        make_fetch_tool(vector_store),
        make_compare_tool(vector_store),
        make_extract_tool(vector_store),
    ]
    llm_with_tools = _build_llm_with_tools(tools)
    raw_tool_node = ToolNode(tools)

    def agent_node(state: AgentState) -> dict:
        """Call the LLM (with tools bound) on the current conversation."""
        messages = [SystemMessage(content=_SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def tool_node(state: AgentState) -> dict:
        """Execute the requested tool(s) and increment the iteration counter."""
        result = raw_tool_node.invoke(state)
        return {
            "messages": result["messages"],
            "tool_calls_count": state.get("tool_calls_count", 0) + 1,
        }

    def should_continue(state: AgentState) -> str:
        """Route to tools if the LLM requested one and we are under the cap."""
        last = state["messages"][-1]
        has_calls = isinstance(last, AIMessage) and bool(last.tool_calls)
        if has_calls and state.get("tool_calls_count", 0) < MAX_AGENT_ITERATIONS:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    compiled = graph.compile()
    logger.info("Agent graph compiled with %d tools", len(tools))
    return compiled


def run_agent(
    agent,
    user_query: str,
    chat_history: list | None = None,
    paper_context: str = "",
) -> dict:
    """Invoke the agent and return the answer, tool log, and cited sources.

    Args:
        agent: A compiled graph from ``build_agent``.
        user_query: The user's question.
        chat_history: Prior messages as ``[{"role": ..., "content": ...}, ...]``.
        paper_context: Optional note telling the agent which indexed paper is the
            user's own upload, so phrases like "my paper" resolve correctly.

    Returns:
        A dict with ``answer``, ``tool_calls`` and ``sources`` keys.
    """
    init_messages: list = []
    if paper_context:
        init_messages.append(SystemMessage(content=paper_context))
    init_messages.extend(_history_to_messages(chat_history or []))
    init_messages.append(HumanMessage(content=user_query))
    init_state: AgentState = {"messages": init_messages, "tool_calls_count": 0}

    # llama models on Groq intermittently emit a malformed tool-call format
    # ("tool_use_failed"). Retry a couple of times before giving up so the
    # caller can fall back to plain RAG.
    final_state = None
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_AGENT_RETRIES + 1):
        try:
            final_state = agent.invoke(init_state, config={"recursion_limit": 25})
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.error("Agent invocation failed (attempt %d): %s", attempt, exc)
            if _is_tool_use_error(exc) and attempt < _MAX_AGENT_RETRIES:
                time.sleep(1)
                continue
            break

    if final_state is None:
        return {
            "answer": f"An error occurred while running the agent: {last_exc}",
            "tool_calls": [],
            "sources": [],
            "error": True,
        }

    messages = final_state["messages"]
    tool_calls: list[dict] = []
    sources: dict[str, dict] = {}

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                tool_calls.append(
                    {"tool": call.get("name", "unknown"), "input": call.get("args", {})}
                )
        if isinstance(msg, ToolMessage):
            _harvest_sources(str(msg.content), sources)

    answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
            answer = msg.content
            break
    if not answer:
        answer = "The agent did not produce a final answer."

    return {"answer": answer, "tool_calls": tool_calls, "sources": list(sources.values())}


# NOTE: agent token-streaming was intentionally removed. LangGraph's "messages"
# stream mode streams the LLM's tool-deciding calls, and streaming a tool call
# makes llama on Groq emit a malformed function format (tool_use_failed). The
# agent therefore runs non-streamed; only single-call RAG answers are streamed
# (see core.rag_chain.stream_answer).


def _history_to_messages(chat_history: list) -> list:
    """Convert a list of role/content dicts into LangChain message objects."""
    messages = []
    for turn in chat_history:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _store_source(
    sources: dict[str, dict], title: str, ident: str, section: str, snippet: str
) -> None:
    """Add a parsed citation (title, id/section, snippet) to ``sources``."""
    ident = (ident or "").strip()
    low = ident.lower()
    is_own = "own upload" in low or low.startswith("upload")
    aid = ""
    if "arxiv:" in low:
        aid = ident[low.index("arxiv:") + 6 :].strip()
    elif not is_own and re.fullmatch(r"[\w.\-/]+", ident):
        aid = ident  # extract_tool passes a bare id
    key = aid if aid else f"own:{title.strip()}"
    if key not in sources:
        sources[key] = {
            "arxiv_id": aid,
            "title": title.strip(),
            "section": (section or "").strip() or "Unknown",
            "snippet": (snippet or "").strip()[:500],
            "is_own": is_own,
        }


def _harvest_sources(text: str, sources: dict[str, dict]) -> None:
    """Extract paper citations (with section + excerpt) from a tool result."""
    matched = False
    for m in _BRACKET_RE.finditer(text):
        matched = True
        _store_source(sources, m.group("title"), m.group("ident"), m.group("section"), m.group("snippet"))
    for m in _COMPARE_RE.finditer(text):
        matched = True
        _store_source(sources, m.group("title"), m.group("ident"), "", m.group("snippet"))
    if not matched:
        for m in _ARXIV_RE.finditer(text):
            aid = m.group(1).strip()
            sources.setdefault(
                aid,
                {"arxiv_id": aid, "title": "Unknown", "section": "Unknown",
                 "snippet": "", "is_own": False},
            )
