# Self-Correction Log

> This file is the project's learning database. Agents read it at the start of every session
> to avoid repeating past mistakes, and append to it whenever they encounter non-obvious issues.
>
> **Format:** Each entry must include Date, Context, Mistake, Fix, and Lesson.

---

<!-- Entries will be appended below by agents as they encounter issues. -->

## 2026-03-31T00:08:00Z
- **Context:** Verifying extracted Python app using sequential shell commands on Windows PowerShell.
- **Mistake:** Used `&&` as a command separator, which failed in this PowerShell environment.
- **Fix:** Switched to `;` for sequential execution in PowerShell.
- **Lesson:** Prefer PowerShell-compatible separators for multi-command verification steps.

## 2026-03-31T00:08:20Z
- **Context:** Runtime smoke test for extracted notebook app.
- **Mistake:** Ran the app before installing notebook dependencies.
- **Fix:** Installed `requirements.txt` and re-ran verification commands.
- **Lesson:** Install runtime dependencies before startup checks when extracting notebook code into scripts.

## 2026-03-31T13:42:00Z
- **Context:** Pushing files to a newly created GitHub repo via MCP `push_files`.
- **Mistake:** Called `push_files` on an empty repo (no commits yet), which returned `Conflict: Git Repository is empty`.
- **Fix:** Used `create_or_update_file` to create the initial commit (README.md), then used `push_files` for remaining files.
- **Lesson:** Always initialize an empty GitHub repo with at least one file via `create_or_update_file` before using `push_files`.

## 2026-03-31T17:50:00Z
- **Context:** LinkedIn post creator prints LLM-generated drafts containing emoji characters on Windows.
- **Mistake:** `print(draft)` raises `UnicodeEncodeError` because Windows console uses `cp1252` encoding which cannot represent emoji. `UnicodeEncodeError` inherits from `ValueError`, so it was caught by the generic `except ValueError` in `main()` instead of the intended `except UnicodeEncodeError` in `run_linkedin_post`.
- **Fix:** Added `try/except UnicodeEncodeError` around the print call with a fallback that encodes using `sys.stdout.encoding` with `errors="replace"`. For reliable testing, set `PYTHONUTF8=1` env var.
- **Lesson:** On Windows, always guard `print()` of LLM output against `UnicodeEncodeError`. Remember that `UnicodeEncodeError` is a subclass of `ValueError` — catch it before broader handlers.
