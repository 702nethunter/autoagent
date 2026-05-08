"""
memory_store.py — SQL Server persistence layer for the generative agent memory stream.

Implements the three memory operations from Park et al. §3:
  • add_memory()      — insert observation / reflection / plan
  • retrieve()        — score by recency + importance + relevance, return top-k
  • reflect_trigger() — check whether cumulative importance exceeds threshold
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Optional

import pyodbc

import config


# ── Connection pool (one per process) ─────────────────────────────────────────

_conn: Optional[pyodbc.Connection] = None


def get_conn() -> pyodbc.Connection:
    global _conn
    if _conn is None:
        _conn = pyodbc.connect(config.get_connection_string(), autocommit=False)
    return _conn


def _cur():
    return get_conn().cursor()


# ── Agent CRUD ─────────────────────────────────────────────────────────────────

def upsert_agent(name: str, role: str, persona: str) -> int:
    """Return agent_id, creating the row if it doesn't exist."""
    cur = _cur()
    cur.execute("SELECT agent_id FROM Agents WHERE name = ?", name)
    row = cur.fetchone()
    if row:
        return row.agent_id
    cur.execute(
        "INSERT INTO Agents (name, role, persona) OUTPUT INSERTED.agent_id VALUES (?, ?, ?)",
        name, role, persona,
    )
    agent_id = cur.fetchone()[0]
    get_conn().commit()
    return agent_id


def get_agent(agent_id: int) -> dict:
    cur = _cur()
    cur.execute("SELECT agent_id, name, role, persona FROM Agents WHERE agent_id = ?", agent_id)
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Agent {agent_id} not found")
    return {"agent_id": row.agent_id, "name": row.name, "role": row.role, "persona": row.persona}


def list_agents() -> list[dict]:
    cur = _cur()
    cur.execute("SELECT agent_id, name, role FROM Agents")
    return [{"agent_id": r.agent_id, "name": r.name, "role": r.role} for r in cur.fetchall()]


# ── Memory stream ──────────────────────────────────────────────────────────────

def add_memory(
    agent_id: int,
    description: str,
    memory_type: str,           # 'observation' | 'reflection' | 'plan'
    importance: float,
    embedding: Optional[list[float]] = None,
) -> int:
    """Insert a memory object and return its memory_id."""
    emb_json = json.dumps(embedding) if embedding else None
    cur = _cur()
    cur.execute(
        """INSERT INTO MemoryStream
               (agent_id, memory_type, description, importance, embedding_json)
           OUTPUT INSERTED.memory_id
           VALUES (?, ?, ?, ?, ?)""",
        agent_id, memory_type, description, importance, emb_json,
    )
    memory_id = cur.fetchone()[0]
    get_conn().commit()
    return memory_id


def link_reflection_sources(reflection_id: int, source_ids: list[int]) -> None:
    cur = _cur()
    for sid in source_ids:
        cur.execute(
            "INSERT INTO ReflectionSources (reflection_memory_id, source_memory_id) VALUES (?, ?)",
            reflection_id, sid,
        )
    get_conn().commit()


def get_recent_memories(agent_id: int, limit: int = 100) -> list[dict]:
    cur = _cur()
    cur.execute(
        """SELECT memory_id, memory_type, description, importance, embedding_json,
                  created_at, last_accessed
           FROM MemoryStream
           WHERE agent_id = ? AND is_active = 1
           ORDER BY created_at DESC
           OFFSET 0 ROWS FETCH NEXT ? ROWS ONLY""",
        agent_id, limit,
    )
    return [_row_to_dict(r) for r in cur.fetchall()]


def _row_to_dict(r) -> dict:
    return {
        "memory_id":    r.memory_id,
        "memory_type":  r.memory_type,
        "description":  r.description,
        "importance":   r.importance,
        "embedding":    json.loads(r.embedding_json) if r.embedding_json else None,
        "created_at":   r.created_at,
        "last_accessed": r.last_accessed,
    }


# ── Retrieval scoring (Park et al. §3.2) ──────────────────────────────────────

def _recency_score(mem: dict, now: datetime) -> float:
    """Exponential decay: score = decay_factor ^ hours_since_access."""
    last = mem["last_accessed"]
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    hours = max((now - last).total_seconds() / 3600, 0)
    return config.RECENCY_DECAY ** hours


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _relevance_score(mem: dict, query_embedding: Optional[list[float]]) -> float:
    if query_embedding is None or mem["embedding"] is None:
        return 0.5   # neutral when embeddings unavailable
    return max(0.0, _cosine(mem["embedding"], query_embedding))


def _normalize(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def retrieve(
    agent_id: int,
    query_embedding: Optional[list[float]],
    top_k: int = 10,
    candidate_limit: int = 200,
) -> list[dict]:
    """
    Retrieve the top-k most relevant memories using the paper's scoring formula:
        score = α_rec·recency + α_imp·importance + α_rel·relevance
    All three components normalised to [0, 1] across the candidate set.
    Updates last_accessed for returned memories.
    """
    candidates = get_recent_memories(agent_id, limit=candidate_limit)
    if not candidates:
        return []

    now = datetime.now(timezone.utc)

    rec_raw = [_recency_score(m, now)   for m in candidates]
    imp_raw = [m["importance"] / 10.0   for m in candidates]   # already 1–10 → 0–1
    rel_raw = [_relevance_score(m, query_embedding) for m in candidates]

    rec_n = _normalize(rec_raw)
    imp_n = _normalize(imp_raw)
    rel_n = _normalize(rel_raw)

    scored = []
    for i, mem in enumerate(candidates):
        score = (
            config.RECENCY_WEIGHT    * rec_n[i] +
            config.IMPORTANCE_WEIGHT * imp_n[i] +
            config.RELEVANCE_WEIGHT  * rel_n[i]
        )
        scored.append((score, mem))

    scored.sort(key=lambda x: -x[0])
    top = [m for _, m in scored[:top_k]]

    # Update last_accessed
    ids = [m["memory_id"] for m in top]
    if ids:
        cur = _cur()
        placeholders = ",".join("?" * len(ids))
        cur.execute(
            f"UPDATE MemoryStream SET last_accessed = GETUTCDATE() WHERE memory_id IN ({placeholders})",
            *ids,
        )
        get_conn().commit()

    return top


# ── Reflection trigger ─────────────────────────────────────────────────────────

def should_reflect(agent_id: int) -> tuple[bool, list[dict]]:
    """
    Returns (True, recent_memories) if sum of importance of recent non-reflection
    memories exceeds REFLECTION_THRESHOLD.
    """
    recent = get_recent_memories(agent_id, limit=config.REFLECTION_LOOKBACK)
    observations = [m for m in recent if m["memory_type"] != "reflection"]
    total = sum(m["importance"] for m in observations)
    return total >= config.REFLECTION_THRESHOLD, observations


# ── Plans ──────────────────────────────────────────────────────────────────────

def create_plan(
    agent_id: int,
    title: str,
    description: str = "",
    parent_plan_id: Optional[int] = None,
    assigned_to: Optional[int] = None,
    priority: int = 5,
) -> int:
    cur = _cur()
    cur.execute(
        """INSERT INTO Plans (agent_id, title, description, parent_plan_id, assigned_to, priority)
           OUTPUT INSERTED.plan_id VALUES (?, ?, ?, ?, ?, ?)""",
        agent_id, title, description, parent_plan_id, assigned_to, priority,
    )
    plan_id = cur.fetchone()[0]
    get_conn().commit()
    return plan_id


def update_plan_status(plan_id: int, status: str) -> None:
    cur = _cur()
    if status == "in_progress":
        cur.execute(
            "UPDATE Plans SET status = ?, started_at = GETUTCDATE() WHERE plan_id = ?",
            status, plan_id,
        )
    elif status == "done":
        cur.execute(
            "UPDATE Plans SET status = ?, completed_at = GETUTCDATE() WHERE plan_id = ?",
            status, plan_id,
        )
    else:
        cur.execute("UPDATE Plans SET status = ? WHERE plan_id = ?", status, plan_id)
    get_conn().commit()


def get_open_tasks(assigned_to: int) -> list[dict]:
    cur = _cur()
    cur.execute(
        """SELECT plan_id, title, description, priority, status, parent_plan_id
           FROM Plans WHERE assigned_to = ? AND status IN ('pending', 'in_progress')
           ORDER BY priority, plan_id""",
        assigned_to,
    )
    return [
        {"plan_id": r.plan_id, "title": r.title, "description": r.description,
         "priority": r.priority, "status": r.status, "parent_plan_id": r.parent_plan_id}
        for r in cur.fetchall()
    ]


# ── Messages ───────────────────────────────────────────────────────────────────

def send_message(from_id: int, to_id: int, content: str, msg_type: str = "chat") -> int:
    cur = _cur()
    cur.execute(
        """INSERT INTO Messages (from_agent_id, to_agent_id, content, message_type)
           OUTPUT INSERTED.message_id VALUES (?, ?, ?, ?)""",
        from_id, to_id, content, msg_type,
    )
    msg_id = cur.fetchone()[0]
    get_conn().commit()
    return msg_id


def read_unread_messages(agent_id: int) -> list[dict]:
    cur = _cur()
    cur.execute(
        """SELECT m.message_id, m.from_agent_id, a.name as from_name,
                  m.content, m.message_type, m.created_at
           FROM Messages m
           JOIN Agents a ON a.agent_id = m.from_agent_id
           WHERE m.to_agent_id = ? AND m.is_read = 0
           ORDER BY m.created_at""",
        agent_id,
    )
    rows = cur.fetchall()
    if rows:
        ids = [r.message_id for r in rows]
        placeholders = ",".join("?" * len(ids))
        cur.execute(f"UPDATE Messages SET is_read = 1 WHERE message_id IN ({placeholders})", *ids)
        get_conn().commit()
    return [
        {"message_id": r.message_id, "from_agent_id": r.from_agent_id,
         "from_name": r.from_name, "content": r.content,
         "message_type": r.message_type, "created_at": r.created_at}
        for r in rows
    ]


# ── Incidents ──────────────────────────────────────────────────────────────────

def log_incident(
    reported_by: int,
    title: str,
    description: str,
    severity: str = "medium",
    plan_id: Optional[int] = None,
    assigned_to: Optional[int] = None,
) -> int:
    cur = _cur()
    cur.execute(
        """INSERT INTO Incidents (reported_by, title, description, severity, plan_id, assigned_to)
           OUTPUT INSERTED.incident_id VALUES (?, ?, ?, ?, ?, ?)""",
        reported_by, title, description, severity, plan_id, assigned_to,
    )
    incident_id = cur.fetchone()[0]
    get_conn().commit()
    return incident_id


def resolve_incident(incident_id: int, root_cause: str, resolution: str) -> None:
    cur = _cur()
    cur.execute(
        """UPDATE Incidents SET status = 'resolved', root_cause = ?,
           resolution = ?, resolved_at = GETUTCDATE()
           WHERE incident_id = ?""",
        root_cause, resolution, incident_id,
    )
    get_conn().commit()


def get_open_incidents(assigned_to: Optional[int] = None) -> list[dict]:
    cur = _cur()
    if assigned_to:
        cur.execute(
            """SELECT incident_id, title, description, severity, status, created_at
               FROM Incidents WHERE status IN ('open','investigating') AND assigned_to = ?
               ORDER BY severity, created_at""",
            assigned_to,
        )
    else:
        cur.execute(
            """SELECT incident_id, title, description, severity, status, created_at
               FROM Incidents WHERE status IN ('open','investigating')
               ORDER BY severity, created_at""",
        )
    return [
        {"incident_id": r.incident_id, "title": r.title, "description": r.description,
         "severity": r.severity, "status": r.status, "created_at": r.created_at}
        for r in cur.fetchall()
    ]
