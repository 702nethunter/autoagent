#!/usr/bin/env python3
"""
main.py — Generative Dev Agents simulation runner.

Simulates a software development team (PM + .NET dev + C++ dev) working through
a sprint, following the architecture from Park et al. (2023) "Generative Agents".

Usage:
    python main.py                         # default feature request, 3 sprint days
    python main.py --days 5
    python main.py --feature "Build a real-time telemetry dashboard"
    python main.py --no-standup            # skip daily standups
"""

from __future__ import annotations

import argparse
import sys

import config
import memory_store as ms
from dev_agent import CppDeveloperAgent, DotNetDeveloperAgent
from pm_agent import ProjectManagerAgent

WIDTH = 70

DEFAULT_FEATURE = (
    "Build a cross-platform telemetry pipeline: "
    "a C++ data collector that captures system metrics and streams them "
    "over gRPC to a .NET REST API that stores and exposes them via "
    "an ASP.NET Core endpoint backed by SQL Server."
)


def separator(title: str = "") -> None:
    if title:
        pad = max(0, WIDTH - len(title) - 4)
        print(f"\n{'─' * 2} {title} {'─' * pad}")
    else:
        print(f"{'─' * WIDTH}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generative Dev Agents simulation")
    p.add_argument("--feature",    default=DEFAULT_FEATURE, help="Feature request for the sprint")
    p.add_argument("--days",       type=int, default=3,     help="Number of sprint days to simulate")
    p.add_argument("--no-standup", action="store_true",     help="Skip daily standups")
    p.add_argument("--model",      default=config.DEFAULT_MODEL, help="Ollama model to use")
    return p.parse_args()


def check_ollama(ollama_url: str) -> None:
    import requests
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as e:
        print(f"Cannot reach Ollama at {ollama_url}: {e}", file=sys.stderr)
        print("Start Ollama with: ollama serve", file=sys.stderr)
        sys.exit(1)


def check_sql_server() -> None:
    try:
        conn = ms.get_conn()
        cur = conn.cursor()
        cur.execute("SELECT 1")
    except Exception as e:
        print(f"Cannot connect to SQL Server: {e}", file=sys.stderr)
        print(
            "Ensure SQL Server is running and update config.py or set env vars:\n"
            "  SQL_SERVER, SQL_DATABASE, SQL_USER, SQL_PASSWORD",
            file=sys.stderr,
        )
        sys.exit(1)


def print_incident_report() -> None:
    separator("INCIDENT REPORT")
    incidents = ms.get_open_incidents()
    if not incidents:
        print("  No open incidents.")
        return
    for inc in incidents:
        print(
            f"  #{inc['incident_id']} [{inc['severity'].upper()}] {inc['title']}\n"
            f"    Status: {inc['status']}  |  Created: {inc['created_at']}\n"
            f"    {inc['description'][:100]}"
        )


def print_plan_summary(pm: ProjectManagerAgent) -> None:
    separator("TASK SUMMARY")
    import pyodbc
    cur = ms.get_conn().cursor()
    cur.execute(
        """SELECT p.plan_id, p.title, p.status, a.name as assigned_name
           FROM Plans p
           LEFT JOIN Agents a ON a.agent_id = p.assigned_to
           WHERE p.parent_plan_id IS NOT NULL
           ORDER BY p.priority, p.plan_id"""
    )
    rows = cur.fetchall()
    done    = sum(1 for r in rows if r.status == "done")
    blocked = sum(1 for r in rows if r.status == "blocked")
    pending = sum(1 for r in rows if r.status == "pending")
    in_prog = sum(1 for r in rows if r.status == "in_progress")
    print(f"  Total tasks: {len(rows)}  |  Done: {done}  |  In-progress: {in_prog}  "
          f"|  Blocked: {blocked}  |  Pending: {pending}")
    for r in rows:
        icon = {"done": "✓", "blocked": "✗", "in_progress": "→", "pending": "·"}.get(r.status, " ")
        name = r.assigned_name or "unassigned"
        print(f"  {icon} [{r.status:<11}] {r.title[:55]:<55} ({name})")


def main() -> None:
    args = parse_args()
    config.DEFAULT_MODEL = args.model   # apply CLI model override

    print(f"\n{'═' * WIDTH}")
    print(f"  Generative Dev Agents — based on Park et al. (2023)")
    print(f"  Model: {args.model}  |  Days: {args.days}")
    print(f"{'═' * WIDTH}")

    # ── Pre-flight checks ──────────────────────────────────────────────────────
    check_ollama(config.OLLAMA_URL)
    check_sql_server()

    # ── Bootstrap agents ───────────────────────────────────────────────────────
    separator("AGENT INIT")
    pm      = ProjectManagerAgent()
    dotnet  = DotNetDeveloperAgent()
    cpp_dev = CppDeveloperAgent()

    pm.register_dev(dotnet)
    pm.register_dev(cpp_dev)

    for agent in (pm, dotnet, cpp_dev):
        print(f"  {agent.name} (id={agent.agent_id})")

    # ── Sprint planning (day 0) ────────────────────────────────────────────────
    separator("SPRINT PLANNING")
    print(f"  Feature: {args.feature[:80]}")
    sprint_plan_id = pm.plan_sprint(args.feature, sprint_name="Sprint 1")
    print(f"  Sprint plan created (id={sprint_plan_id})")

    # ── Daily simulation ───────────────────────────────────────────────────────
    for day in range(1, args.days + 1):
        separator(f"DAY {day}")

        # Dev agents work their tasks
        for dev in (dotnet, cpp_dev):
            print(f"\n── {dev.name} working …")
            dev.act()

        # PM handles escalations mid-day
        pm.handle_escalations()

        # Daily standup
        if not args.no_standup:
            separator(f"DAY {day} — STANDUP")
            pm.run_standup()

    # ── End-of-sprint reports ──────────────────────────────────────────────────
    separator("END OF SPRINT")
    print_plan_summary(pm)
    print_incident_report()

    # Final PM reflection
    separator("PM RETROSPECTIVE")
    retro = pm.act("Sprint has ended. Reflect on what went well and what to improve.")
    print(f"\n  {retro}")

    print(f"\n{'═' * WIDTH}")
    print("  Simulation complete. All data persisted to SQL Server.")
    print(f"{'═' * WIDTH}\n")


if __name__ == "__main__":
    main()
