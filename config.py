"""Central configuration — edit SQL_SERVER_* to match your environment."""

import os

# ── Ollama ─────────────────────────────────────────────────────────────────────
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
DEFAULT_MODEL = os.getenv("AGENT_MODEL", "nemotron-mini:4b")
EMBED_MODEL   = os.getenv("EMBED_MODEL", "nomic-embed-text")  # ollama pull nomic-embed-text

# ── SQL Server ─────────────────────────────────────────────────────────────────
SQL_SERVER   = os.getenv("SQL_SERVER",   "192.168.0.133")
SQL_DATABASE = os.getenv("SQL_DATABASE", "DevAgents")
SQL_USER     = os.getenv("SQL_USER",     "sa")
SQL_PASSWORD = os.getenv("SQL_PASSWORD", "Soyuz@MS22")
SQL_DRIVER   = os.getenv("SQL_DRIVER",   "ODBC Driver 18 for SQL Server")

def get_connection_string() -> str:
    return (
        f"DRIVER={{{SQL_DRIVER}}};"
        f"SERVER={SQL_SERVER};"
        f"DATABASE={SQL_DATABASE};"
        f"UID={SQL_USER};"
        f"PWD={SQL_PASSWORD};"
        "TrustServerCertificate=yes;"
        "Encrypt=yes;"
    )

# ── Memory retrieval weights (Park et al. §3.2) ────────────────────────────────
RECENCY_WEIGHT    = 1.0
IMPORTANCE_WEIGHT = 1.0
RELEVANCE_WEIGHT  = 1.0
RECENCY_DECAY     = 0.99          # per-hour exponential decay

# ── Reflection trigger ─────────────────────────────────────────────────────────
REFLECTION_THRESHOLD = 150        # sum of importance of recent memories before reflecting
REFLECTION_LOOKBACK  = 100        # how many recent memories to scan for reflection trigger

# ── Planning ───────────────────────────────────────────────────────────────────
MAX_SUBTASKS_PER_TASK = 5
MEMORY_RETRIEVE_TOP_K = 10        # memories to inject into each LLM prompt
