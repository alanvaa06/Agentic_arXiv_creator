# Project Memory

> Durable context carried across sessions. Read by every agent before planning.

---

## User Preferences

- **LLM Provider:** Anthropic (Claude). Do NOT use OpenAI/GPT models. All LLM calls, agents, and configurations must target Anthropic APIs and `langchain-anthropic` (`ChatAnthropic`).
- **Default model:** `claude-sonnet-4-20250514` (configurable via `ANTHROPIC_MODEL` env var).

## Project Status

- Multi-agent AGI research pipeline extracted from notebook into `research_multi_agent_system.py`.
- Multi-agent LinkedIn post creator extracted from notebook into `linkedin_post_creator.py` (2026-03-31).
  - Pipeline: Supervisor → Researcher (Tavily) → Writer → Critic + Groundedness evaluator.
  - Migrated from OpenAI/GPT to Anthropic Claude; secrets via `.env` (not `config.json`).
- Multi-domain evaluation support added to research pipeline (2026-03-31): `RUBRIC_REGISTRY` with AGI, ML, Finance, Economics rubrics; `--domain` CLI flag; domain-aware reports.
- Research-to-LinkedIn pipeline integrated (2026-03-31):
  - `linkedin_post_creator.py` now supports `--from-report` flag to seed posts from evaluation reports.
  - `run_pipeline.py` orchestrates end-to-end: research → evaluate → LinkedIn post.
  - Report parser (`parse_evaluation_report`) extracts top paper data from `.md` reports.
  - Paper context flows through `LinkedInPostState.paper_context` to enrich researcher + writer prompts.
- Stack: LangChain + LangGraph + `langchain-anthropic` + arXiv API + Tavily search.
- GitHub repo created: https://github.com/alanvaa06/Agentic_arXiv_creator
- README enriched with architecture diagrams, Anthropic Claude integration docs, AGI evaluation framework, and multi-agent design explanation (2026-03-31).
- Gradio web app (`app.py`) added (2026-03-31):
  - Three modes: Full Pipeline, Research Only, LinkedIn Post from Report.
  - Config bypass: builds `AppConfig` objects from form inputs; clears `@lru_cache` for research module.
  - Stdout capture via `contextlib.redirect_stdout` feeds agent logs to UI textbox.
  - `gradio>=5.0.0` added to `requirements.txt`.
- README updates (2026-04-05): clarified **query vs domain** (topic vs rubric), documented all `RUBRIC_REGISTRY` domains in a table, retitled evaluation diagram copy to “Rubric Scoring Agent,” added Gradio usage subsection, noted legacy `agi_*` field names in reports/UI.
- Minor source normalization (2026-04-05): real em dash characters and f-string spacing in `run_pipeline.py`, `linkedin_post_creator.py`, `research_multi_agent_system.py`; self-correction log entry for Windows `UnicodeEncodeError` on printing drafts.
