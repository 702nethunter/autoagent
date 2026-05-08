"""
base_agent.py — Core GenerativeAgent class.

Implements the three cognitive mechanisms from Park et al. (2023):
  1. Memory stream  — observe() records events with importance scoring
  2. Retrieval      — context() returns top-k memories scored by recency+importance+relevance
  3. Reflection     — reflect() synthesises higher-level insights when threshold exceeded
  4. Planning       — plan() decomposes goals into subtasks stored in the Plans table
"""

from __future__ import annotations

import textwrap
from typing import Optional

import config
import llm
import memory_store as ms


class GenerativeAgent:
    """
    Base class for all agents.  Subclasses override act() to define role behaviour.
    """

    def __init__(self, name: str, role: str, persona: str):
        self.name    = name
        self.role    = role
        self.persona = persona
        self.agent_id: int = ms.upsert_agent(name, role, persona)

    # ── 1. Observe ────────────────────────────────────────────────────────────

    def observe(self, event: str, memory_type: str = "observation") -> int:
        """
        Record an event in the memory stream.
        The LLM rates its importance; an embedding is generated for retrieval.
        """
        importance = llm.rate_importance(event)
        embedding  = llm.embed(event)
        memory_id  = ms.add_memory(
            agent_id    = self.agent_id,
            description = event,
            memory_type = memory_type,
            importance  = importance,
            embedding   = embedding,
        )
        return memory_id

    # ── 2. Retrieve context ───────────────────────────────────────────────────

    def context(self, query: str, top_k: int = config.MEMORY_RETRIEVE_TOP_K) -> list[dict]:
        """Return top-k memories most relevant to query."""
        q_emb = llm.embed(query)
        return ms.retrieve(self.agent_id, q_emb, top_k=top_k)

    def context_text(self, query: str) -> str:
        """Return memories as a formatted string ready to inject into a prompt."""
        mems = self.context(query)
        if not mems:
            return "(no relevant memories)"
        lines = []
        for m in mems:
            tag = m["memory_type"][0].upper()   # O / R / P
            lines.append(f"  [{tag}] {m['description']}")
        return "\n".join(lines)

    # ── 3. Reflect ────────────────────────────────────────────────────────────

    def reflect(self) -> list[int]:
        """
        Synthesise high-level insights from recent memories when the importance
        threshold is exceeded (Park et al. §3.3).
        Returns list of new reflection memory_ids.
        """
        triggered, recent = ms.should_reflect(self.agent_id)
        if not triggered:
            return []

        # Step 1: ask LLM what questions the recent memories raise
        memory_text = "\n".join(
            f"  {i+1}. {m['description']}" for i, m in enumerate(recent[:20])
        )
        question_prompt = (
            f"You are {self.name}, a {self.role} on a software team.\n\n"
            f"Given only the information below, what are the 3 most salient "
            f"high-level questions you can ask about your work situation?\n\n"
            f"Recent observations:\n{memory_text}\n\n"
            f"List exactly 3 questions, one per line, no numbering."
        )
        questions_raw = llm.complete(question_prompt, temperature=0.6, max_tokens=200)
        questions = [q.strip() for q in questions_raw.splitlines() if q.strip()][:3]

        new_ids: list[int] = []
        source_ids = [m["memory_id"] for m in recent[:20]]

        for question in questions:
            # Step 2: synthesise an insight that answers each question
            insight_prompt = (
                f"You are {self.name}, a {self.role}.\n\n"
                f"Given these recent observations:\n{memory_text}\n\n"
                f"Provide a concise, high-level insight that answers:\n"
                f'"{question}"\n\n'
                f"Write 1-2 sentences as a factual statement. No preamble."
            )
            insight = llm.complete(insight_prompt, temperature=0.5, max_tokens=120)
            if not insight:
                continue

            rid = self.observe(insight, memory_type="reflection")
            ms.link_reflection_sources(rid, source_ids)
            new_ids.append(rid)
            print(f"  [REFLECT:{self.name}] {insight[:100]}")

        return new_ids

    # ── 4. Plan ───────────────────────────────────────────────────────────────

    def make_plan(
        self,
        goal: str,
        context_query: str = "",
        parent_plan_id: Optional[int] = None,
        assigned_to: Optional[int] = None,
        priority: int = 5,
    ) -> tuple[int, list[int]]:
        """
        Create a top-level plan and decompose it into subtasks.
        Returns (plan_id, [subtask_plan_ids]).
        """
        ctx = self.context_text(context_query or goal)

        decompose_prompt = (
            f"You are {self.name}, a {self.role} on a software development team.\n\n"
            f"Relevant context from memory:\n{ctx}\n\n"
            f"Goal: {goal}\n\n"
            f"Break this into {config.MAX_SUBTASKS_PER_TASK} or fewer concrete subtasks. "
            f"Each subtask should be a single actionable step.\n"
            f"Format: one subtask per line, no numbering, no bullets."
        )
        raw = llm.complete(decompose_prompt, temperature=0.5, max_tokens=300)
        subtasks = [s.strip() for s in raw.splitlines() if s.strip()]
        subtasks = subtasks[:config.MAX_SUBTASKS_PER_TASK]

        plan_id = ms.create_plan(
            agent_id       = self.agent_id,
            title          = goal,
            description    = raw,
            parent_plan_id = parent_plan_id,
            assigned_to    = assigned_to,
            priority       = priority,
        )

        subtask_ids = []
        for i, st in enumerate(subtasks):
            sid = ms.create_plan(
                agent_id       = self.agent_id,
                title          = st,
                parent_plan_id = plan_id,
                assigned_to    = assigned_to,
                priority       = priority + i,
            )
            subtask_ids.append(sid)

        self.observe(f"Created plan: {goal} with {len(subtasks)} subtasks", memory_type="plan")
        return plan_id, subtask_ids

    # ── 5. Communicate ────────────────────────────────────────────────────────

    def send(self, to_agent_id: int, content: str, msg_type: str = "chat") -> int:
        msg_id = ms.send_message(self.agent_id, to_agent_id, content, msg_type)
        self.observe(f"Sent message to agent {to_agent_id}: {content[:80]}")
        return msg_id

    def read_inbox(self) -> list[dict]:
        messages = ms.read_unread_messages(self.agent_id)
        for msg in messages:
            self.observe(f"Received from {msg['from_name']}: {msg['content'][:120]}")
        return messages

    # ── 6. Report incident ────────────────────────────────────────────────────

    def report_incident(
        self,
        title: str,
        description: str,
        severity: str = "medium",
        plan_id: Optional[int] = None,
    ) -> int:
        incident_id = ms.log_incident(
            reported_by = self.agent_id,
            title       = title,
            description = description,
            severity    = severity,
            plan_id     = plan_id,
        )
        self.observe(f"Reported incident [{severity}]: {title}", memory_type="observation")
        print(f"  [INCIDENT:{self.name}] #{incident_id} [{severity}] {title}")
        return incident_id

    # ── 7. Core action loop ───────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        return textwrap.dedent(f"""
            You are {self.name}.
            Role: {self.role}
            Persona: {self.persona}
            Always respond in character. Be concise and professional.
        """).strip()

    def act(self, situation: str) -> str:
        """
        Main decision point — subclasses override this.
        Default: retrieve context and generate a response.
        """
        ctx = self.context_text(situation)
        prompt = (
            f"{self._build_system_prompt()}\n\n"
            f"Relevant memory:\n{ctx}\n\n"
            f"Current situation: {situation}\n\n"
            f"What do you do? Respond briefly."
        )
        response = llm.complete(prompt, temperature=0.7, max_tokens=300)
        self.observe(f"In response to '{situation[:80]}': {response[:120]}")
        # Check if reflection should be triggered after this action
        self.reflect()
        return response

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} id={self.agent_id}>"
