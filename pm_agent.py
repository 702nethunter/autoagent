"""
pm_agent.py — Project Manager agent.

Responsibilities:
  • Sprint planning: decompose feature requests into tasks and assign to devs
  • Daily standup: query each dev's progress and synthesise a status report
  • Escalation: detect blocked tasks / open incidents and resolve or escalate
  • Reflection: generate sprint retrospective insights
"""

from __future__ import annotations

from typing import Optional

import llm
import memory_store as ms
from base_agent import GenerativeAgent

PM_PERSONA = (
    "You are an experienced software project manager. "
    "You are concise, organised, and focused on unblocking your team. "
    "You track progress rigorously, identify risks early, and communicate "
    "clearly with both technical developers and business stakeholders."
)


class ProjectManagerAgent(GenerativeAgent):

    def __init__(self, name: str = "Project Manager"):
        super().__init__(name=name, role="pm", persona=PM_PERSONA)
        self._dev_agents: list[GenerativeAgent] = []

    def register_dev(self, agent: "GenerativeAgent") -> None:
        """Register a developer agent so the PM can assign and query them."""
        self._dev_agents.append(agent)

    # ── Sprint planning ────────────────────────────────────────────────────────

    def plan_sprint(self, feature_request: str, sprint_name: str = "Sprint") -> int:
        """
        Decompose a feature request into tasks and assign each to an appropriate dev.
        Returns the top-level plan_id.
        """
        print(f"\n[PM] Planning sprint: {feature_request[:80]}")
        self.observe(f"Received feature request: {feature_request}")

        # Step 1: decide which tasks belong to which technology
        routing_prompt = (
            f"{self._build_system_prompt()}\n\n"
            f"Feature request: {feature_request}\n\n"
            f"Available developers:\n"
            + "\n".join(f"  - {d.name} ({d.role})" for d in self._dev_agents)
            + "\n\nDecompose this into concrete tasks. For each task write:\n"
            "  TASK: <title> | ASSIGN: <developer name> | PRIORITY: <1-5>\n"
            "One task per line. No extra text."
        )
        routing_raw = llm.complete(routing_prompt, temperature=0.4, max_tokens=400)

        # Parse task lines
        tasks: list[dict] = []
        for line in routing_raw.splitlines():
            line = line.strip()
            if not line.startswith("TASK:"):
                continue
            parts = {p.split(":")[0].strip(): ":".join(p.split(":")[1:]).strip()
                     for p in line.split("|") if ":" in p}
            title    = parts.get("TASK", line)
            dev_name = parts.get("ASSIGN", "")
            try:
                priority = int(parts.get("PRIORITY", "5"))
            except ValueError:
                priority = 5

            # Find matching dev agent
            assigned_agent = next(
                (d for d in self._dev_agents if dev_name.lower() in d.name.lower()),
                self._dev_agents[0] if self._dev_agents else None,
            )
            tasks.append({"title": title, "agent": assigned_agent, "priority": priority})

        if not tasks:
            # Fallback: one task per dev
            lines = [l.strip() for l in routing_raw.splitlines() if l.strip()]
            for i, d in enumerate(self._dev_agents):
                desc = lines[i] if i < len(lines) else f"{feature_request} ({d.role} part)"
                tasks.append({"title": desc, "agent": d, "priority": i + 1})

        # Create top-level plan
        plan_id = ms.create_plan(
            agent_id    = self.agent_id,
            title       = f"{sprint_name}: {feature_request[:80]}",
            description = routing_raw,
            priority    = 1,
        )

        for task in tasks:
            dev: Optional[GenerativeAgent] = task["agent"]
            assigned_id = dev.agent_id if dev else None
            subtask_id = ms.create_plan(
                agent_id       = self.agent_id,
                title          = task["title"],
                parent_plan_id = plan_id,
                assigned_to    = assigned_id,
                priority       = task["priority"],
            )
            if dev:
                # Notify the dev agent via message
                self.send(
                    to_agent_id = dev.agent_id,
                    content     = f"You have been assigned: {task['title']} (task #{subtask_id})",
                    msg_type    = "assignment",
                )
                print(f"  [PM→{dev.name}] Assigned: {task['title'][:60]}")

        self.observe(f"Created {sprint_name} with {len(tasks)} tasks (plan #{plan_id})")
        return plan_id

    # ── Daily standup ──────────────────────────────────────────────────────────

    def run_standup(self) -> str:
        """
        Ask each dev what they've done, what they're doing, and if they're blocked.
        Synthesise a standup summary and store it as a reflection.
        """
        print(f"\n[PM] Running daily standup …")
        updates: list[str] = []

        for dev in self._dev_agents:
            tasks    = ms.get_open_tasks(dev.agent_id)
            incidents = ms.get_open_incidents(assigned_to=dev.agent_id)

            standup_prompt = (
                f"{dev._build_system_prompt()}\n\n"
                f"It's standup time. Your open tasks:\n"
                + "\n".join(f"  - [{t['status']}] {t['title']}" for t in tasks)
                + f"\n\nOpen incidents assigned to you:\n"
                + ("\n".join(f"  - [{i['severity']}] {i['title']}" for i in incidents)
                   if incidents else "  None")
                + "\n\nGive your standup update (2-3 sentences: done, doing, blocked)."
            )
            update = llm.complete(standup_prompt, temperature=0.6, max_tokens=150)
            dev.observe(f"Gave standup update: {update[:100]}")
            updates.append(f"{dev.name}: {update}")
            print(f"  [{dev.name}] {update[:120]}")

        # PM synthesises the full standup
        summary_prompt = (
            f"{self._build_system_prompt()}\n\n"
            f"Standup updates from the team:\n"
            + "\n\n".join(updates)
            + "\n\nWrite a brief PM summary: overall status, key blockers, "
            "and any actions you will take (2-4 sentences)."
        )
        summary = llm.complete(summary_prompt, temperature=0.5, max_tokens=200)
        self.observe(f"Standup summary: {summary[:200]}", memory_type="reflection")
        print(f"\n  [PM SUMMARY] {summary}")
        return summary

    # ── Escalation ─────────────────────────────────────────────────────────────

    def handle_escalations(self) -> None:
        """Check for open critical/high incidents and decide on action."""
        incidents = ms.get_open_incidents()
        high = [i for i in incidents if i["severity"] in ("critical", "high")]
        if not high:
            print("[PM] No critical incidents to escalate.")
            return

        for inc in high:
            print(f"  [PM] Escalating incident #{inc['incident_id']}: {inc['title']}")
            ctx = self.context_text(inc["description"])
            action_prompt = (
                f"{self._build_system_prompt()}\n\n"
                f"Critical incident: {inc['title']}\n"
                f"Description: {inc['description']}\n\n"
                f"Relevant history:\n{ctx}\n\n"
                f"What immediate action do you take? (2 sentences)"
            )
            action = llm.complete(action_prompt, temperature=0.4, max_tokens=120)
            self.observe(f"Escalation action for incident #{inc['incident_id']}: {action[:100]}")

            # Notify all devs
            for dev in self._dev_agents:
                self.send(
                    to_agent_id = dev.agent_id,
                    content     = f"ESCALATION — Incident #{inc['incident_id']} [{inc['severity']}]: "
                                  f"{inc['title']}. PM action: {action}",
                    msg_type    = "escalation",
                )

    # ── Override act() ─────────────────────────────────────────────────────────

    def act(self, situation: str) -> str:
        # Read inbox first — devs may have sent updates
        messages = self.read_inbox()
        if messages:
            for msg in messages:
                print(f"  [PM inbox] {msg['from_name']}: {msg['content'][:100]}")

        ctx = self.context_text(situation)
        prompt = (
            f"{self._build_system_prompt()}\n\n"
            f"Relevant memory:\n{ctx}\n\n"
            f"Situation: {situation}\n\n"
            f"What do you do as project manager? (2-3 sentences)"
        )
        response = llm.complete(prompt, temperature=0.6, max_tokens=200)
        self.observe(f"PM action: {response[:120]}")
        self.reflect()
        return response
