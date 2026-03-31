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
- Stack: LangChain + LangGraph + `langchain-anthropic` + arXiv API + Tavily search.
- GitHub repo created: https://github.com/alanvaa06/Agentic_arXiv_creator
- README enriched with architecture diagrams, Anthropic Claude integration docs, AGI evaluation framework, and multi-agent design explanation (2026-03-31).
