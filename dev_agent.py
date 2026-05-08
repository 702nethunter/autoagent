"""
dev_agent.py — Developer agent base class.

Each agent now produces actual code files (not just plans).
Generated files are written to output_dir and returned so callers
can post them to GitHub or upload as artifacts.

Action loop:
  1. Read inbox (PM assignments, escalations)
  2. Pick the highest-priority open task
  3. Generate implementation plan
  4. Generate actual code file(s)
  5. Self-review for risks
  6. Mark task status, log to memory stream
  7. Reflect if threshold exceeded
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Optional

import llm
import memory_store as ms
from base_agent import GenerativeAgent

# Where generated files are written — can be overridden by callers
DEFAULT_OUTPUT_DIR = Path("output")


class DeveloperAgent(GenerativeAgent):
    """Shared behaviour for all developer agents."""

    technology: str = "software"
    tech_context: str = ""
    file_extension: str = ".txt"      # subclasses override

    # ── Prompts ───────────────────────────────────────────────────────────────

    def _implementation_prompt(self, task_title: str, task_desc: str, ctx: str) -> str:
        return (
            f"{self._build_system_prompt()}\n\n"
            f"Technology stack: {self.technology}\n"
            f"{self.tech_context}\n\n"
            f"Relevant memory / prior work:\n{ctx}\n\n"
            f"Task: {task_title}\n"
            f"Details: {task_desc or '(no additional details)'}\n\n"
            f"Write a concise implementation plan (3-5 bullet points). "
            f"Focus on technical approach, key files/classes, and edge cases."
        )

    def _code_prompt(self, task_title: str, task_desc: str, plan: str) -> str:
        return (
            f"{self._build_system_prompt()}\n\n"
            f"Technology stack: {self.technology}\n"
            f"{self.tech_context}\n\n"
            f"Task: {task_title}\n"
            f"Details: {task_desc or '(no additional details)'}\n\n"
            f"Implementation plan:\n{plan}\n\n"
            f"Write the FULL implementation code for this task. "
            f"Output ONLY the code — no explanations, no markdown fences. "
            f"Make it production-quality and complete."
        )

    def _code_review_prompt(self, code: str) -> str:
        return (
            f"{self._build_system_prompt()}\n\n"
            f"Review this code:\n{code[:600]}\n\n"
            f"Identify ONE potential risk or edge case (1 sentence). "
            f"If none, say 'No significant risks identified.'"
        )

    # ── Filename derivation ───────────────────────────────────────────────────

    def _derive_filename(self, task_title: str) -> str:
        """Turn a task title into a safe filename."""
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", task_title.lower()).strip("_")[:50]
        return slug + self.file_extension

    # ── Core work method ──────────────────────────────────────────────────────

    def work_on_task(self, task: dict, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
        """
        Work on a task: plan → generate code → self-review → save file.
        Returns {"plan": str, "code": str, "filename": str, "filepath": Path, "risk": str}
        """
        title = task["title"]
        desc  = task.get("description", "")
        pid   = task["plan_id"]

        ctx  = self.context_text(title)

        # Step 1: plan
        plan = llm.complete(
            self._implementation_prompt(title, desc, ctx),
            temperature=0.6,
            max_tokens=350,
        )

        # Step 2: generate actual code
        code = llm.complete(
            self._code_prompt(title, desc, plan),
            temperature=0.5,
            max_tokens=800,
        )

        # Step 3: self-review
        risk = llm.complete(
            self._code_review_prompt(code),
            temperature=0.4,
            max_tokens=80,
        )

        # Step 4: save to disk
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = self._derive_filename(title)
        filepath = output_dir / filename
        filepath.write_text(code, encoding="utf-8")

        ms.update_plan_status(pid, "in_progress")
        self.observe(f"Implemented task '{title}' → {filename}")
        if "risk" in risk.lower() or "issue" in risk.lower() or "miss" in risk.lower():
            self.observe(f"Self-review risk on '{title}': {risk[:100]}")

        print(f"  [{self.name}] Task: {title[:60]}")
        print(f"    → {filepath}  ({len(code)} chars)")
        if risk and "no significant" not in risk.lower():
            print(f"    Risk: {risk[:100]}")

        return {"plan": plan, "code": code, "filename": filename,
                "filepath": filepath, "risk": risk}

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

    def act(self, situation: str = "start of day",
            output_dir: Path = DEFAULT_OUTPUT_DIR) -> tuple[str, list[dict]]:
        """
        Returns (status_string, list_of_artifact_dicts).
        Each artifact dict: {filename, filepath, code, plan, task_title}
        """
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
            return response, []

        task      = tasks[0]
        artifacts = []

        roll = random.random()
        if roll < 0.80:
            result = self.work_on_task(task, output_dir=output_dir)
            self.complete_task(task)
            artifacts.append({**result, "task_title": task["title"]})
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
                f"Describe a realistic bug encountered while working on: {task['title']} "
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

        self.reflect()
        return response, artifacts


# ── Concrete agent subclasses ──────────────────────────────────────────────────

class DotNetDeveloperAgent(DeveloperAgent):
    technology     = ".NET / C# / ASP.NET Core / Entity Framework"
    file_extension = ".cs"
    tech_context   = (
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
    technology     = "C++17/20 / CMake / STL / Boost"
    file_extension = ".cpp"
    tech_context   = (
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
    technology     = "React 18 / TypeScript / Vite / Tailwind CSS"
    file_extension = ".tsx"
    tech_context   = (
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
            f"Task: {task_title}\n"
            f"Details: {task_desc or '(no additional details)'}\n\n"
            f"Write a concise implementation plan (3-5 bullet points) covering:\n"
            f"  • Component hierarchy and file structure\n"
            f"  • State management approach\n"
            f"  • API integration points\n"
            f"  • Key props/types to define\n"
            f"  • Accessibility and responsive design considerations"
        )

    def _code_prompt(self, task_title: str, task_desc: str, plan: str) -> str:
        return (
            f"{self._build_system_prompt()}\n\n"
            f"Technology stack: {self.technology}\n"
            f"{self.tech_context}\n\n"
            f"Task: {task_title}\n"
            f"Details: {task_desc or '(no additional details)'}\n\n"
            f"Implementation plan:\n{plan}\n\n"
            f"Write the FULL React TypeScript component (.tsx) for this task. "
            f"Output ONLY the code — no explanations, no markdown fences. "
            f"Use proper TypeScript types, hooks, and Tailwind CSS classes."
        )
