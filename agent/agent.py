"""
Background Coding Agent - runs on GitHub Actions
=================================================

This agent demonstrates every building block from the course:

  L01 - Agent Loop:      The main loop that reads issue → plans → fixes → validates
  L04 - Tool Use:        Claude API as the reasoning engine (function calling pattern)
  L05 - RAG:             Reads codebase files as context (small repo = full context)
                         For a large repo, you'd replace read_codebase() with a
                         vector search over embedded code chunks (hybrid retrieval).
  L06 - Trustworthy:     Two-job separation. This script is read-only.
                         It outputs a diff file. A separate job opens the PR.
  L07 - Planning:        Agent generates a plan before writing code.
  L08 - Multi-agent:     Planner → Coder → Reviewer in a single script
                         (sequential calls with different system prompts).
  L09 - Metacognition:   Agent evaluates its own fix before outputting.
  L10 - Production:      Structured logging so Actions logs are readable.
  L12 - Context Eng:     Careful context assembly: system prompt + issue + code + tests.
  L13 - Memory:          Shared state dict carries context between agent steps.

Usage (runs automatically in GitHub Actions, but you can also run locally):
    export ANTHROPIC_API_KEY=sk-...
    python agent/agent.py --issue-number 1 --repo-path .

The agent outputs:
    - agent-output/plan.md        (the plan it generated)
    - agent-output/fix.diff       (the unified diff to apply)
    - agent-output/summary.md     (self-review summary)
"""

import os
import sys
import json
import glob
import argparse
import subprocess
from pathlib import Path

import anthropic


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 4096

# Where the agent writes its output (picked up by the second Actions job)
OUTPUT_DIR = "agent-output"


# ============================================================================
# LOGGING — structured output so Actions logs are easy to read (L10)
# ============================================================================

def log(stage, message):
    """Print a structured log line. Each stage maps to a building block."""
    print(f"\n{'='*60}")
    print(f"[{stage}] {message}")
    print(f"{'='*60}")


# ============================================================================
# CONTEXT ASSEMBLY — what goes into the LLM's prompt (L05, L12)
# ============================================================================

def read_issue(issue_number, repo_path):
    """
    Read the issue body from GitHub.
    
    In Actions, we get this from the event payload (GITHUB_EVENT_PATH).
    Locally, we fall back to the gh CLI.
    
    This is the TASK — the input to the agent loop (L01).
    """
    event_path = os.environ.get("GITHUB_EVENT_PATH")

    if event_path and os.path.exists(event_path):
        # Running in Actions — read from event payload
        with open(event_path) as f:
            event = json.load(f)
        issue = event.get("issue", {})
        return {
            "number": issue.get("number"),
            "title": issue.get("title", ""),
            "body": issue.get("body", ""),
        }
    else:
        # Running locally — use gh CLI
        try:
            result = subprocess.run(
                ["gh", "issue", "view", str(issue_number), "--json", "number,title,body"],
                cwd=repo_path, capture_output=True, text=True, check=True,
            )
            return json.loads(result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {
                "number": issue_number,
                "title": "Test issue",
                "body": "Fix the bug where apply_discount crashes when discount_code is None",
            }


def read_codebase(repo_path):
    """
    Read all Python source and test files from the repo.
    
    RAG NOTE (L05): This works because our repo is tiny (~100 lines).
    For a large monorepo (thousands of files), you would NOT do this.
    Instead you'd:
      1. Chunk code by function/class using AST parsing
      2. Embed chunks with a code-optimized model
      3. Store in pgvector or similar
      4. At query time: embed the issue text, hybrid search (lexical + semantic)
      5. Return only the top K relevant chunks
      
    The interface stays the same — this function returns a string of code
    that gets stuffed into the prompt. Only the retrieval method changes.
    """
    files = {}
    # Find all Python files in src/ and tests/
    for pattern in ["src/**/*.py", "tests/**/*.py"]:
        for filepath in glob.glob(os.path.join(repo_path, pattern), recursive=True):
            relpath = os.path.relpath(filepath, repo_path)
            with open(filepath) as f:
                files[relpath] = f.read()

    # Format as a string the LLM can read
    context = ""
    for path, content in sorted(files.items()):
        context += f"\n### {path}\n```python\n{content}\n```\n"
    return context


def run_tests(repo_path):
    """
    Run the test suite and capture output.
    
    This gives the agent ground truth about what's broken (L09).
    The agent uses test output to understand the bug, and later
    to validate its fix.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            cwd=repo_path, capture_output=True, text=True, timeout=60,
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": 1, "stdout": "", "stderr": "Tests timed out after 60s"}


# ============================================================================
# AGENT STEPS — each step is a separate LLM call with its own prompt (L08)
# ============================================================================

def call_claude(system_prompt, user_message):
    """
    Make a single Claude API call.
    
    This is the raw function-calling protocol from L04:
    system prompt + user message → response text.
    
    Every agent step below calls this with a different system prompt,
    making each step act like a specialized agent (L08 multi-agent pattern).
    """
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    
    return response.content[0].text


def step_plan(issue, codebase, test_output):
    """
    STEP 1: PLANNER (L07)
    
    Reads the issue + codebase + test results and generates a plan.
    The planner doesn't write code — it analyzes and strategizes.
    """
    log("PLANNER", f"Analyzing issue #{issue['number']}: {issue['title']}")

    system_prompt = """\
You are a senior engineer analyzing a bug report. Your job is to create a plan, NOT write code.

Analyze:
1. What is the bug? (from the issue description)
2. What is the root cause? (from the code)  
3. What tests are failing? (from the test output)
4. What files need to change?
5. What is the minimal fix?

Output a clear, numbered plan. Be specific about which files and functions to change."""

    user_message = f"""\
## Issue #{issue['number']}: {issue['title']}

{issue['body']}

## Current codebase
{codebase}

## Test results
```
{test_output['stdout']}
{test_output['stderr']}
```"""

    plan = call_claude(system_prompt, user_message)
    log("PLANNER", f"Plan generated ({len(plan)} chars)")
    return plan


def step_code(issue, codebase, plan):
    """
    STEP 2: CODER (L08)
    
    Receives the plan from step 1 and generates the fixed file contents.
    
    Instead of generating a diff (which LLMs often malformat), we ask
    for both a diff AND the full corrected files. The workflow tries
    the diff first and falls back to full file replacement.
    """
    log("CODER", "Generating fix based on plan...")

    system_prompt = """\
You are a coding agent. You receive a plan from the planner and write the fix.

You MUST output TWO sections:

1. A unified diff (for `git apply`). Use exact --- a/ and +++ b/ headers.
2. The FULL corrected file contents, wrapped like this:
   === FILE: src/cart.py ===
   (entire file content here)
   === END FILE ===

Rules:
- Make the minimal change needed
- Don't refactor unrelated code
- Output BOTH the diff AND the full file"""

    user_message = f"""\
## Plan to follow
{plan}

## Current codebase
{codebase}"""

    response = call_claude(system_prompt, user_message)
    log("CODER", f"Response generated ({len(response.splitlines())} lines)")
    return response


def step_review(issue, plan, diff, test_output):
    """
    STEP 3: REVIEWER / SELF-EVALUATION (L09)
    
    The agent reviews its own work before outputting.
    This is the metacognition step — evaluating the fix quality,
    not just whether it compiles.
    
    The reviewer checks:
    - Does the diff match the plan?
    - Does it fix the right bug?
    - Are there any obvious issues?
    - Is it minimal (no unnecessary changes)?
    """
    log("REVIEWER", "Self-reviewing the fix...")

    system_prompt = """\
You are a code reviewer evaluating a fix generated by an AI coding agent.

Check:
1. Does the diff match the plan?
2. Does it address the issue described?
3. Are there any obvious bugs in the fix itself?
4. Is it minimal (no unnecessary changes)?
5. Would the failing tests pass after this fix?

Output a short review summary. End with one of:
- VERDICT: APPROVE — if the fix looks good
- VERDICT: REJECT — if there's a problem (explain what)"""

    user_message = f"""\
## Issue #{issue['number']}: {issue['title']}
{issue['body']}

## Plan
{plan}

## Generated diff
```diff
{diff}
```

## Current test results (before fix)
```
{test_output['stdout'][:2000]}
```"""

    review = call_claude(system_prompt, user_message)
    log("REVIEWER", f"Review complete")
    return review


# ============================================================================
# MAIN AGENT LOOP (L01)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Background coding agent")
    parser.add_argument("--issue-number", type=int, default=1)
    parser.add_argument("--repo-path", type=str, default=".")
    args = parser.parse_args()

    log("AGENT", f"Starting agent for issue #{args.issue_number}")
    log("AGENT", f"Repo path: {os.path.abspath(args.repo_path)}")
    log("AGENT", f"Model: {MODEL}")

    # ── SHARED STATE (L13 Memory) ──────────────────────────────────
    # This dict carries context between agent steps.
    # Each step reads what it needs and writes its output.
    # In a multi-agent system, this would be the shared artifact store.
    state = {}

    # ── STEP 0: GATHER CONTEXT (L05 RAG, L12 Context Engineering) ──
    log("CONTEXT", "Reading issue...")
    state["issue"] = read_issue(args.issue_number, args.repo_path)
    print(f"  Issue: #{state['issue']['number']} - {state['issue']['title']}")

    log("CONTEXT", "Reading codebase...")
    state["codebase"] = read_codebase(args.repo_path)
    print(f"  Files loaded: {state['codebase'].count('###')} files")
    # RAG NOTE: For a large repo, this would be a vector search call instead.

    log("CONTEXT", "Running tests to understand current state...")
    state["test_output"] = run_tests(args.repo_path)
    print(f"  Test exit code: {state['test_output']['exit_code']}")

    # ── STEP 1: PLAN (L07 Planning) ─────────────────────────────────
    state["plan"] = step_plan(
        state["issue"], state["codebase"], state["test_output"]
    )

    # ── STEP 2: CODE (L08 Multi-agent — coder role) ─────────────────
    state["diff"] = step_code(
        state["issue"], state["codebase"], state["plan"]
    )

    # ── STEP 3: REVIEW (L09 Metacognition — self-evaluation) ────────
    state["review"] = step_review(
        state["issue"], state["plan"], state["diff"], state["test_output"]
    )

    # ── OUTPUT ARTIFACTS ─────────────────────────────────────────────
    # These files are picked up by the second Actions job (L06 capability separation).
    # The agent (this script) has NO write access to the repo.
    # The second job reads these artifacts and opens a PR.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(OUTPUT_DIR, "plan.md"), "w") as f:
        f.write(f"# Plan for Issue #{state['issue']['number']}\n\n{state['plan']}")

    # Parse the coder's response into diff + full files
    coder_output = state["diff"]
    
    # Extract the diff portion (everything before === FILE:)
    diff_part = coder_output.split("=== FILE:")[0].strip()
    # Clean up markdown code fences if present
    if "```diff" in diff_part:
        diff_part = diff_part.split("```diff")[-1].split("```")[0].strip()
    elif "```" in diff_part:
        diff_part = diff_part.split("```")[1].split("```")[0].strip()
    
    with open(os.path.join(OUTPUT_DIR, "fix.diff"), "w") as f:
        f.write(diff_part)

    # Extract full file replacements as a fallback
    files_dir = os.path.join(OUTPUT_DIR, "files")
    if "=== FILE:" in coder_output:
        os.makedirs(files_dir, exist_ok=True)
        parts = coder_output.split("=== FILE:")
        for part in parts[1:]:  # skip everything before first === FILE:
            if "=== END FILE ===" in part:
                header, content = part.split("\n", 1)
                filepath = header.strip().rstrip("=").strip()
                content = content.split("=== END FILE ===")[0].strip()
                # Clean markdown fences if present
                if content.startswith("```"):
                    content = "\n".join(content.split("\n")[1:])
                if content.endswith("```"):
                    content = "\n".join(content.split("\n")[:-1])
                full_path = os.path.join(files_dir, filepath)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w") as f:
                    f.write(content + "\n")
                log("OUTPUT", f"Wrote full file: {filepath}")

    with open(os.path.join(OUTPUT_DIR, "summary.md"), "w") as f:
        f.write(f"# Review Summary\n\n{state['review']}")

    # ── FINAL STATUS ─────────────────────────────────────────────────
    approved = "APPROVE" in state["review"].upper()
    log("AGENT", f"Review verdict: {'APPROVED ✓' if approved else 'REJECTED ✗'}")
    log("AGENT", f"Artifacts written to {OUTPUT_DIR}/")

    if not approved:
        log("AGENT", "Fix was rejected by self-review. Exiting with error.")
        sys.exit(1)

    log("AGENT", "Done. The apply job will create the PR.")


if __name__ == "__main__":
    main()
