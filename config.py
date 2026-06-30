# Central config — all magic numbers and paths live here
import os

import streamlit as st
from dotenv import load_dotenv

# override=True so a rotated key in .env always wins over a stale OS env var.
load_dotenv(override=True)


# Support both .env (local) and Streamlit Cloud secrets (deployed)
def get_secret(key: str) -> str | None:
    """Return a secret value from Streamlit Cloud secrets, falling back to env vars.

    Args:
        key: Name of the secret to look up.

    Returns:
        The secret value, or None if it is not set in either location.
    """
    # Try Streamlit secrets first (deployed). Broad except: accessing st.secrets
    # without a secrets file raises different errors across Streamlit versions,
    # and must never crash import (e.g. in CI / headless test runs).
    try:
        return st.secrets[key]
    except Exception:  # noqa: BLE001
        pass
    # Fall back to environment variable (local)
    return os.getenv(key)


# API Keys
GROQ_API_KEY = get_secret("GROQ_API_KEY")

# Model Config
# llama-3.1-8b-instant is far more reliable at tool-calling than the 70b model on
# Groq (verified ~4/4 vs ~2-4/4) and has higher free-tier rate limits.
LLM_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.1
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking Config
CHUNK_SIZE = 1000          # characters per chunk
CHUNK_OVERLAP = 200        # overlap between chunks
MIN_CHUNK_SIZE = 100       # discard chunks smaller than this

# Vector Store
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "research_papers"

# Paper Storage
PAPERS_DIR = "./data/papers"
STATIC_DIR = "./static"   # served by Streamlit for in-browser PDF viewing

# Retrieval
TOP_K_RESULTS = 5          # number of chunks to retrieve (lower = fewer tokens/call)
SIMILARITY_THRESHOLD = 0.3  # minimum relevance score

# Reranking — a cross-encoder re-scores candidates for better precision.
# Disabled by default because the cross-encoder is a SECOND transformer model;
# on small/free hosts (e.g. Streamlit Community Cloud) the extra RAM can OOM the
# app. Set to True locally or on a larger instance for better retrieval quality.
RERANK_ENABLED = False
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 20     # candidates fetched before reranking down to TOP_K

# Agent
MAX_AGENT_ITERATIONS = 3   # prevent infinite loops (each iteration = 1 Groq request)
MAX_PAPERS_PER_SEARCH = 5  # papers to fetch per arxiv search

# Rate limiting (abuse protection + staying under Groq free-tier limits).
# NOTE: one user question triggers MULTIPLE Groq requests (the agent's ReAct
# loop), so question-level caps are deliberately well below Groq's request caps.
# Groq free tier (llama-3.3-70b-versatile): 30 req/min, 12K tokens/min,
# 1K req/day, 100K tokens/day. The token/day limit is the real ceiling.
RATE_LIMIT_WINDOW = 60          # seconds in the per-minute sliding window
RATE_LIMIT_PER_SESSION = 4      # max questions/min for a single user
RATE_LIMIT_GLOBAL = 5           # max questions/min across ALL users
FETCH_LIMIT_PER_SESSION = 3     # max arxiv fetch actions/min for a single user
DAILY_WINDOW = 86400            # seconds in the per-day sliding window
RATE_LIMIT_GLOBAL_DAILY = 50    # max questions/day across ALL users (token/day guard)
ANALYZE_LIMIT_PER_SESSION = 2   # max paper analyses/min for a single user

# Upload & Review
UPLOAD_RELATED_DEFAULT = 3      # related arxiv papers fetched per analysis
REVIEW_PAPER_CHAR_BUDGET = 6000  # cap on uploaded-paper text sent to the LLM
REVIEW_RELATED_TOP_K = 6        # related excerpts retrieved from the corpus

# Ensure runtime directories exist (Streamlit Cloud has ephemeral storage)
os.makedirs(PAPERS_DIR, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
