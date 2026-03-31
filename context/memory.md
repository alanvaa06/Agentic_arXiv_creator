# Project Memory

> Durable context carried across sessions. Read by every agent before planning.

---

## User Preferences

- **LLM Provider:** Anthropic (Claude). Do NOT use OpenAI/GPT models. All LLM calls, agents, and configurations must target Anthropic APIs and `langchain-anthropic` (`ChatAnthropic`).
- **Default model:** `claude-sonnet-4-20250514` (configurable via `ANTHROPIC_MODEL` env var).

## Project Status

- Multi-agent AGI research pipeline extracted from notebook into `research_multi_agent_system.py`.
- Migrated from OpenAI/GPT to Anthropic Claude (2026-03-31).
- Stack: LangChain + LangGraph + `langchain-anthropic` + arXiv API.
- GitHub repo created: https://github.com/alanvaa06/Agentic_arXiv_creator
