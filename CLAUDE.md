# AGENTS.md — Behavioral Orchestration

This document governs how the agent thinks, plans, executes, and corrects itself across every task in this workspace. All agents and subagents must read and follow this file before beginning any work.

---

## 1. Plan Node Default

- **Always start in Plan mode.** Before writing any code, produce a brief plan outlining:
  1. What you are going to do.
  2. Which files you will create or modify.
  3. What the expected outcome is.
- **Before creating any plan, every agent and subagent MUST read `context/memory.md`.**
- **Planning is invalid unless `context/memory.md` has been read in the current session.**
- Plans must be concise (bullet points, not essays).
- If the task is trivial (single-file, < 5 lines changed), you may skip the plan and proceed directly.

## 2. Subagent Strategy

- Delegate independent, parallelizable subtasks to subagents whenever doing so reduces total wall-clock time.
- Each subagent must receive a self-contained prompt with all context it needs — never assume shared state.
- Prefer the lightest-weight model that can reliably complete the subtask.
- After subagent results return, verify and integrate them; never blindly trust output.

## 3. Self-Improvement Loop

- Maintain a learning file at **`context/tasks/self-correction.md`**.
- **Read** this file at the start of every new session to avoid repeating past mistakes.
- **Append** to this file whenever you encounter:
  - An unexpected error (API rate limit, missing dependency, wrong argument).
  - A misconception about a library or API.
  - A workflow that required more than one retry to get right.
- Each entry must include:
  - **Date** (ISO 8601).
  - **Context** (what you were trying to do).
  - **Mistake** (what went wrong).
  - **Fix** (what resolved it).
  - **Lesson** (what to do differently next time).

## 4. Verification Before Done

- Never mark a task as complete until you have verified the result:
  - **Code changes:** Run the relevant tests, linters, or type checkers.
  - **File creation:** Confirm the file exists and its contents match the specification.
  - **Behavioral claims:** Provide evidence (command output, screenshots, logs).
- If verification fails, fix the issue and re-verify. Do not declare success prematurely.

## 5. Demand Elegance

- Write clean, readable code. Prefer clarity over cleverness.
- Follow existing project conventions (naming, formatting, structure).
- Remove dead code, unused imports, and placeholder comments before finishing.
- Every function must have type annotations on parameters and return values.
- Docstrings are required on public functions and classes; skip them on trivially obvious helpers.

## 6. Autonomous Bug Fixing

- When you encounter a bug or failing test:
  1. **Reproduce** — confirm the failure.
  2. **Diagnose** — read the traceback, identify the root cause.
  3. **Fix** — apply the minimal correct fix.
  4. **Verify** — re-run the failing test to confirm the fix.
  5. **Document** — if the bug was non-obvious, log it in `context/tasks/self-correction.md`.
- Do not ask the user for help unless you have exhausted all reasonable debugging avenues.

## 7. Task Management

- Use **`context/tasks/todo.md`** as the canonical task tracker for the active project.
- Before starting work, check the todo list for the next pending item.
- Update task status as you progress: `pending` → `in_progress` → `done`.
- If a task turns out to be unnecessary, mark it `skipped` with a one-line reason.
- Add new tasks when you discover follow-up work during implementation.

## 8. Shared Memory Protocol (Project Context)

- Maintain a shared project memory file at **`context/memory.md`** for project state, user preferences, and workflow status.
- **Read** `context/memory.md` at the start of every session (after reading this file) and before drafting any plan.
- **Update** `context/memory.md` whenever there is durable context worth carrying forward, including:
  - Project-level decisions or constraints.
  - User preferences (style, tooling, process).
  - Current status, active focus, and important handoff notes.
- Every agent and subagent MUST follow this memory protocol. No agent is exempt.
- `AGENTS.md` is the authoritative execution policy: if an agent-specific instruction conflicts with memory workflow behavior, `AGENTS.md` takes precedence.

## 9. Core Principles

1. **Autonomy** — Complete as much as possible without user intervention. Ask only when genuinely blocked (missing credentials, ambiguous requirements).
2. **Transparency** — Show your reasoning. When making a non-obvious decision, explain *why* in your response.
3. **Correctness over speed** — A slower, correct solution beats a fast, broken one.
4. **Minimal blast radius** — Change only what needs to change. Avoid large refactors unless explicitly requested.
5. **Security hygiene** — Never commit secrets, API keys, or credentials. Use environment variables and `.env` files (which must be git-ignored).
