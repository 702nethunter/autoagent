#!/usr/bin/env python3
"""
run_from_issue.py — GitHub Actions entry point.

Reads a GitHub Issue (via env vars), runs the PM + dev agents,
then posts the sprint results back as an Issue comment.

Environment variables (set by the workflow):
  ISSUE_NUMBER   — GitHub issue number
  ISSUE_TITLE    — Issue title (used as the feature request)
  ISSUE_BODY     — Issue body (optional extra context)
  GITHUB_TOKEN   — GitHub token for posting comments
  REPO           — owner/repo  e.g. nethunter2023-gif/autoagent
  SQL_SERVER     — SQL Server host (192.168.0.133)
  SQL_USER       — SQL Server username
  SQL_PASSWORD   — SQL Server password
  SQL_DATABASE   — Database name (DevAgents)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
from io import StringIO

import requests

# ── Logging to both console and file ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent_run.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ── Read env vars ──────────────────────────────────────────────────────────────
ISSUE_NUMBER  = os.environ.get("ISSUE_NUMBER",  "0")
ISSUE_TITLE   = os.environ.get("ISSUE_TITLE",   "No title")
ISSUE_BODY    = os.environ.get("ISSUE_BODY",    "")
GITHUB_TOKEN  = os.environ.get("GITHUB_TOKEN",  "")
REPO          = os.environ.get("REPO",          "nethunter2023-gif/autoagent")

# Override SQL Server config from env before importing config
for var, key in [
    ("SQL_SERVER",   "SQL_SERVER"),
    ("SQL_USER",     "SQL_USER"),
    ("SQL_PASSWORD", "SQL_PASSWORD"),
    ("SQL_DATABASE", "SQL_DATABASE"),
]:
    if os.environ.get(var):
        os.environ[key] = os.environ[var]   # config.py reads these

# ── Import agents (after env override) ────────────────────────────────────────
import config                                           # noqa: E402
import memory_store as ms                               # noqa: E402
from dev_agent import CppDeveloperAgent, DotNetDeveloperAgent, ReactDeveloperAgent  # noqa: E402
from pm_agent import ProjectManagerAgent               # noqa: E402


# ── GitHub API helpers ─────────────────────────────────────────────────────────

def post_comment(body: str) -> None:
    if not GITHUB_TOKEN or ISSUE_NUMBER == "0":
        log.info("[DRY RUN] Would post comment:\n%s", body[:400])
        return
    url = f"https://api.github.com/repos/{REPO}/issues/{ISSUE_NUMBER}/comments"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {GITHUB_TOKEN}",
                 "Accept": "application/vnd.github+json"},
        json={"body": body},
        timeout=15,
    )
    if resp.status_code == 201:
        log.info("Comment posted to issue #%s", ISSUE_NUMBER)
    else:
        log.error("Failed to post comment: %s %s", resp.status_code, resp.text[:200])


def add_label(label: str) -> None:
    if not GITHUB_TOKEN or ISSUE_NUMBER == "0":
        return
    url = f"https://api.github.com/repos/{REPO}/issues/{ISSUE_NUMBER}/labels"
    requests.post(
        url,
        headers={"Authorization": f"Bearer {GITHUB_TOKEN}",
                 "Accept": "application/vnd.github+json"},
        json={"labels": [label]},
        timeout=10,
    )


# ── Output capture ─────────────────────────────────────────────────────────────

class Tee(StringIO):
    """Write to both a StringIO buffer and stdout."""
    def write(self, s):
        sys.stdout.write(s)
        return super().write(s)


# ── Main ───────────────────────────────────────────────────────────────────────

def build_feature_request() -> str:
    """Combine issue title + body into a single feature description."""
    if ISSUE_BODY and ISSUE_BODY.strip():
        return f"{ISSUE_TITLE}\n\n{ISSUE_BODY.strip()}"
    return ISSUE_TITLE


def format_github_comment(
    feature: str,
    sprint_plan_id: int,
    task_rows: list[dict],
    incidents: list[dict],
    standup: str,
    retro: str,
) -> str:
    sep = "─" * 50

    tasks_md = "\n".join(
        f"| {icon(r['status'])} `{r['status']}` | {r['title'][:60]} | {r['assigned'] or '—'} |"
        for r in task_rows
    )
    done    = sum(1 for r in task_rows if r["status"] == "done")
    blocked = sum(1 for r in task_rows if r["status"] == "blocked")

    inc_md = "\n".join(
        f"- **[{i['severity'].upper()}]** {i['title']}"
        for i in incidents
    ) or "_No incidents_"

    return textwrap.dedent(f"""
    ## 🤖 Agent Sprint Report — Issue #{ISSUE_NUMBER}

    **Feature:** {feature[:120]}
    **Sprint plan ID:** `{sprint_plan_id}`  |  Tasks: **{len(task_rows)}**  |  ✅ Done: **{done}**  |  ❌ Blocked: **{blocked}**

    ### Tasks
    | Status | Title | Assigned to |
    |--------|-------|-------------|
    {tasks_md}

    ### Incidents
    {inc_md}

    ### Daily Standup Summary
    > {standup.strip().replace(chr(10), ' ')[:400]}

    ### PM Retrospective
    > {retro.strip().replace(chr(10), ' ')[:300]}

    {sep}
    _Generated by [Generative Dev Agents](https://arxiv.org/abs/2304.03442) — nemotron-mini:4b via Ollama_
    """).strip()


def icon(status: str) -> str:
    return {"done": "✅", "blocked": "❌", "in_progress": "🔄", "pending": "⏳"}.get(status, "❓")


def get_task_rows() -> list[dict]:
    cur = ms.get_conn().cursor()
    cur.execute(
        """SELECT p.plan_id, p.title, p.status, a.name as assigned
           FROM Plans p
           LEFT JOIN Agents a ON a.agent_id = p.assigned_to
           WHERE p.parent_plan_id IS NOT NULL
           ORDER BY p.priority, p.plan_id"""
    )
    return [{"plan_id": r.plan_id, "title": r.title,
             "status": r.status, "assigned": r.assigned} for r in cur.fetchall()]


def main() -> None:
    feature = build_feature_request()
    log.info("=" * 60)
    log.info("Issue #%s: %s", ISSUE_NUMBER, ISSUE_TITLE)
    log.info("Feature: %s", feature[:120])
    log.info("=" * 60)

    # ── Post "starting" comment immediately ───────────────────────────────────
    post_comment(
        f"## 🤖 Agents picking up Issue #{ISSUE_NUMBER}\n\n"
        f"**Feature:** {feature[:120]}\n\n"
        f"Running sprint planning with `nemotron-mini:4b`… I'll update this issue when done."
    )
    add_label("agent-running")

    # ── Check connections ──────────────────────────────────────────────────────
    try:
        ms.get_conn().cursor().execute("SELECT 1")
        log.info("SQL Server: OK")
    except Exception as e:
        log.error("SQL Server connection failed: %s", e)
        post_comment(f"❌ **Agent failed:** SQL Server connection error — `{e}`")
        sys.exit(1)

    try:
        r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        log.info("Ollama: OK")
    except Exception as e:
        log.error("Ollama connection failed: %s", e)
        post_comment(f"❌ **Agent failed:** Ollama not reachable — `{e}`")
        sys.exit(1)

    # ── Spin up agents ─────────────────────────────────────────────────────────
    pm      = ProjectManagerAgent()
    dotnet  = DotNetDeveloperAgent()
    cpp_dev = CppDeveloperAgent()
    react   = ReactDeveloperAgent()
    pm.register_dev(dotnet)
    pm.register_dev(cpp_dev)
    pm.register_dev(react)

    # ── Sprint planning ────────────────────────────────────────────────────────
    sprint_plan_id = pm.plan_sprint(feature, sprint_name=f"Issue-{ISSUE_NUMBER}")

    # ── One sprint day — collect generated code artifacts ────────────────────
    from pathlib import Path
    output_dir = Path(f"output/issue-{ISSUE_NUMBER}")
    all_artifacts: list[dict] = []

    for dev in (dotnet, cpp_dev, react):
        _, artifacts = dev.act(output_dir=output_dir)
        all_artifacts.extend(artifacts)

    pm.handle_escalations()

    # ── Standup ────────────────────────────────────────────────────────────────
    standup = pm.run_standup()

    # ── Retrospective ──────────────────────────────────────────────────────────
    retro = pm.act("Sprint complete. Summarise what was achieved on this issue.")

    # ── Gather results ─────────────────────────────────────────────────────────
    task_rows = get_task_rows()
    incidents = ms.get_open_incidents()

    # ── Post sprint report comment ─────────────────────────────────────────────
    comment = format_github_comment(
        feature, sprint_plan_id, task_rows, incidents, standup, retro
    )
    post_comment(comment)

    # ── Post generated code as separate comments (one per file) ───────────────
    if all_artifacts:
        for artifact in all_artifacts:
            ext      = Path(artifact["filename"]).suffix.lstrip(".")
            lang_map = {"tsx": "tsx", "cs": "csharp", "cpp": "cpp", "txt": ""}
            lang     = lang_map.get(ext, ext)
            code_comment = (
                f"### 📄 `{artifact['filename']}`\n"
                f"**Task:** {artifact['task_title']}\n\n"
                f"<details>\n<summary>Implementation plan</summary>\n\n"
                f"```\n{artifact['plan'][:800]}\n```\n</details>\n\n"
                f"```{lang}\n{artifact['code']}\n```"
            )
            post_comment(code_comment)
            log.info("Posted code file: %s (%d chars)", artifact["filename"], len(artifact["code"]))

    add_label("agent-done")
    log.info("Done. %d code files generated. Results posted to GitHub Issue #%s.",
             len(all_artifacts), ISSUE_NUMBER)


if __name__ == "__main__":
    main()
