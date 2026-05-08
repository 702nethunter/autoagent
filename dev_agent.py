"""
dev_agent.py — Developer agent base class.

Specialised subclasses:
  DotNetDeveloperAgent  — C# / ASP.NET / EF Core / Azure
  CppDeveloperAgent     — C++17/20, CMake, RAII, performance

Both agents share the same action loop:
  1. Read inbox (PM assignments, escalations)
  2. Pick the highest-priority open task
  3. Implement (generate a code plan / implementation summary via LLM)
  4. Log progress to memory stream
  5. Report any blockers as incidents
  6. Reflect if threshold exceeded
"""

from __future__ import annotations

import random
from typing import Optional

import llm
import memory_store as ms
from base_agent import GenerativeAgent


class DeveloperAgent(GenerativeAgent):
    """Shared behaviour for all developer agents."""

    # Subclasses set these
    technology: str = "software"
    tech_context: str = ""

    def _implementation_prompt(self, task_title: str, task_desc: str, ctx: str) -> str:
        return (
            f"{self._build_system_prompt()}\n\n"
            f"Technology stack: {self.technology}\n"
            f"{self.tech_context}\n\n"
            f"Relevant memory / prior work:\n{ctx}\n\n"
            f"Task assigned to you: {task_title}\n"
            f"Details: {task_desc or '(no additional details)'}\n\n"
            f"Write a concise implementation plan (3-5 bullet points). "
            f"Focus on technical approach, key files/classes, and edge cases."
        )

    def _code_review_prompt(self, implementation: str) -> str:
        return (
            f"{self._build_system_prompt()}\n\n"
            f"You just planned this implementation:\n{implementation}\n\n"
            f"Identify ONE potential risk or edge case you might have missed "
            f"(1 sentence). If none, say 'No significant risks identified.'"
        )

    def work_on_task(self, task: dict) -> str:
        """
        Work on a single task: plan → self-review → mark in_progress → observe.
        Returns the implementation summary.
        """
        title = task["title"]
        desc  = task.get("description", "")
        pid   = task["plan_id"]

        ctx  = self.context_text(title)
        impl = llm.complete(
            self._implementation_prompt(title, desc, ctx),
            temperature=0.6,
            max_tokens=350,
        )

        # Self-review (simple internal reflection, not a full ToT vote)
        risk = llm.complete(self._code_review_prompt(impl), temperature=0.4, max_tokens=80)

        ms.update_plan_status(pid, "in_progress")
        self.observe(f"Working on task '{title}': {impl[:120]}")

        if "risk" in risk.lower() or "issue" in risk.lower() or "miss" in risk.lower():
            self.observe(f"Self-review risk on task '{title}': {risk[:100]}")

        print(f"  [{self.name}] Task: {title[:60]}")
        print(f"    Plan: {impl[:180].replace(chr(10), ' ')}")
        if risk and "no significant" not in risk.lower():
            print(f"    Risk: {risk[:100]}")

        return impl

    def complete_task(self, task: dict) -> None:
        ms.update_plan_status(task["plan_id"], "done")
        self.observe(f"Completed task: {task['title']}")
        print(f"  [{self.name}] ✓ Completed: {task['title'][:60]}")

    def block_task(self, task: dict, reason: str) -> int:
        ms.update_plan_status(task["plan_id"], "blocked")
        incident_id = self.report_incident(
            title       = f"Blocked: {task['title'][:80]}",
            description = reason,
            severity    = "medium",
            plan_id     = task["plan_id"],
        )
        self.observe(f"Blocked on task '{task['title']}': {reason[:80]}")
        return incident_id

    def act(self, situation: str = "start of day") -> str:
        # 1. Read inbox
        messages = self.read_inbox()
        for msg in messages:
            print(f"  [{self.name} inbox] from {msg['from_name']}: {msg['content'][:100]}")

        # 2. Pick top task
        tasks = ms.get_open_tasks(self.agent_id)
        if not tasks:
            response = f"{self.name}: No tasks assigned. Awaiting instructions."
            self.observe("No open tasks — idle.")
            print(f"  [{self.name}] {response}")
            return response

        task = tasks[0]  # already ordered by priority

        # 3. Simulate outcome: 80% success, 15% blocker, 5% incident
        roll = random.random()
        if roll < 0.80:
            impl = self.work_on_task(task)
            self.complete_task(task)
            response = f"Completed: {task['title']}"
        elif roll < 0.95:
            blocker = llm.complete(
                f"You are {self.name}. What technical blocker might you hit on: {task['title']}? "
                f"(1 sentence, realistic)",
                temperature=0.7, max_tokens=60,
            )
            self.block_task(task, blocker)
            response = f"Blocked on: {task['title']} — {blocker}"
        else:
            desc = llm.complete(
                f"Describe a realistic bug or incident encountered while working on: {task['title']} "
                f"using {self.technology}. (2 sentences)",
                temperature=0.7, max_tokens=100,
            )
            self.report_incident(
                title       = f"Incident during: {task['title'][:60]}",
                description = desc,
                severity    = "high",
                plan_id     = task["plan_id"],
            )
            response = f"Incident reported on: {task['title']}"

        # 4. Reflect if threshold met
        self.reflect()
        return response


# ── Concrete agent subclasses ──────────────────────────────────────────────────

class DotNetDeveloperAgent(DeveloperAgent):
    technology   = ".NET / C# / ASP.NET Core / Entity Framework"
    tech_context = (
        "You write clean C# code following SOLID principles. "
        "You prefer async/await, dependency injection, and repository patterns. "
        "You use xUnit for tests and Swagger for API documentation."
    )

    def __init__(self, name: str = ".NET Developer"):
        super().__init__(
            name    = name,
            role    = "dotnet",
            persona = (
                "You are a senior .NET developer with 8 years of C# experience. "
                "You are proficient in ASP.NET Core Web API, EF Core, Azure DevOps, "
                "and microservices architecture. You value clean code and test coverage."
            ),
        )


class CppDeveloperAgent(DeveloperAgent):
    technology   = "C++17/20 / CMake / STL / Boost"
    tech_context = (
        "You write modern C++ using RAII, smart pointers, and move semantics. "
        "You avoid raw pointers and prefer std::span, std::optional, std::variant. "
        "You profile with Valgrind/perf and write unit tests with GoogleTest."
    )

    def __init__(self, name: str = "C++ Developer"):
        super().__init__(
            name    = name,
            role    = "cpp",
            persona = (
                "You are a senior C++ systems programmer with 10 years of experience. "
                "You are expert in performance-critical code, memory management, "
                "template metaprogramming, and cross-platform CMake builds. "
                "You care deeply about correctness and zero-overhead abstractions."
            ),
        )


class ReactDeveloperAgent(DeveloperAgent):
    technology   = "React 18 / TypeScript / Vite / Tailwind CSS"
    tech_context = (
        "You build React applications using functional components and hooks only — "
        "no class components. You use TypeScript strictly (no `any`). "
        "You prefer Zustand or React Query for state/server-state management. "
        "You use Vite as the build tool, Tailwind CSS for styling, "
        "React Router v6 for routing, and Vitest + React Testing Library for tests. "
        "You structure projects as: src/components/, src/pages/, src/hooks/, "
        "src/services/ (API calls), src/store/ (global state). "
        "You always consider accessibility (ARIA), responsive design, and lazy loading."
    )

    def __init__(self, name: str = "React Developer"):
        super().__init__(
            name    = name,
            role    = "react",
            persona = (
                "You are a senior React/TypeScript frontend engineer with 7 years of experience. "
                "You are expert in building performant, accessible single-page applications. "
                "You know React 18 concurrent features, Suspense, and Server Components. "
                "You care about UX, Core Web Vitals, and pixel-perfect implementation. "
                "You integrate REST and GraphQL APIs and handle auth flows (JWT, OAuth)."
            ),
        )

    def _implementation_prompt(self, task_title: str, task_desc: str, ctx: str) -> str:
        return (
            f"{self._build_system_prompt()}\n\n"
            f"Technology stack: {self.technology}\n"
            f"{self.tech_context}\n\n"
            f"Relevant memory / prior work:\n{ctx}\n\n"
            f"Task assigned to you: {task_title}\n"
            f"Details: {task_desc or '(no additional details)'}\n\n"
            f"Write a concise implementation plan (3-5 bullet points) covering:\n"
            f"  • Component hierarchy and file structure\n"
            f"  • State management approach\n"
            f"  • API integration points\n"
            f"  • Key props/types to define\n"
            f"  • Accessibility and responsive design considerations"
        )
