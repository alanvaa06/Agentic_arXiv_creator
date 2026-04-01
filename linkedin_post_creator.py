"""Multi-agent LinkedIn post creator extracted from notebook.

Pipeline: Supervisor → Researcher → Writer → Critic (+ Groundedness evaluator).
Uses LangGraph for orchestration, Tavily for web search, and Anthropic Claude for LLM.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Annotated

import operator
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("linkedin_creator")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration loaded from environment variables."""

    anthropic_api_key: str
    anthropic_model: str
    max_tokens: int
    tavily_api_key: str
    max_revisions: int
    log_level: str

    @staticmethod
    def from_env() -> AppConfig:
        """Load and validate configuration from environment / .env file."""
        load_dotenv(override=False)

        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not anthropic_key:
            raise ValueError(
                "Missing ANTHROPIC_API_KEY. Set it in your environment or .env file."
            )

        tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not tavily_key:
            raise ValueError(
                "Missing TAVILY_API_KEY. Set it in your environment or .env file."
            )

        return AppConfig(
            anthropic_api_key=anthropic_key,
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514").strip(),
            max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096")),
            tavily_api_key=tavily_key,
            max_revisions=int(os.getenv("MAX_REVISIONS", "5")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper().strip(),
        )


def _build_llm(config: AppConfig, temperature: float = 0.0) -> ChatAnthropic:
    """Create a configured Anthropic LLM client."""
    return ChatAnthropic(
        model=config.anthropic_model,
        temperature=temperature,
        anthropic_api_key=config.anthropic_api_key,
        max_tokens=config.max_tokens,
    )


def _build_tavily(config: AppConfig) -> TavilySearch:
    """Create a configured Tavily search tool."""
    os.environ["TAVILY_API_KEY"] = config.tavily_api_key
    return TavilySearch(
        max_results=5,
        topic="general",
        include_answer=False,
        include_raw_content=False,
        search_depth="basic",
    )


# ---------------------------------------------------------------------------
# Report Parser
# ---------------------------------------------------------------------------

def parse_evaluation_report(path: Path, top_n: int = 1) -> Dict[str, Any]:
    """Parse a detailed evaluation report and extract the Nth-ranked paper.

    The report format is produced by ``_generate_detailed_report`` in
    ``research_multi_agent_system.py``.  Papers appear highest-score-first
    across the HIGH / MEDIUM / LOW sections, so document order equals rank.

    Args:
        path: Path to the ``.md`` evaluation report.
        top_n: Which paper to extract (1 = highest scored).

    Returns:
        Dict with keys: title, authors, score, classification, assessment,
        innovations, domain.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If no papers can be parsed from the file.
    """
    text = path.read_text(encoding="utf-8")

    domain_match = re.search(r"#\s+Detailed\s+(\w+)\s+Evaluation\s+Report", text)
    domain = domain_match.group(1).lower() if domain_match else "unknown"

    paper_pattern = re.compile(r"####\s+\d+\.\s+")
    splits = paper_pattern.split(text)

    if len(splits) < 2:
        raise ValueError(f"No papers found in report: {path}")

    papers: List[Dict[str, Any]] = []
    for block in splits[1:]:
        title = block.strip().split("\n", 1)[0].strip()

        authors_m = re.search(r"Authors?:\s*(.+)", block)
        authors = authors_m.group(1).strip() if authors_m else ""

        score_m = re.search(
            r"(?:Relevance|AGI)\s+Score:\s*([\d.]+)/100\s*\(([^)]+)\)", block
        )
        score = float(score_m.group(1)) if score_m else 0.0
        classification = score_m.group(2).strip() if score_m else "Unknown"

        assess_m = re.search(
            r"Assessment:\s*(.+?)(?:\n\n|\nKey Innovations)", block, re.DOTALL
        )
        assessment = assess_m.group(1).strip() if assess_m else ""

        innov_m = re.search(r"Key Innovations:\n((?:- .+\n?)+)", block)
        innovations: List[str] = []
        if innov_m:
            innovations = [
                line.lstrip("- ").strip()
                for line in innov_m.group(1).strip().split("\n")
                if line.startswith("- ")
            ]

        papers.append({
            "title": title,
            "authors": authors,
            "score": score,
            "classification": classification,
            "assessment": assessment,
            "innovations": innovations,
            "domain": domain,
        })

    if not papers:
        raise ValueError(f"No papers found in report: {path}")

    idx = max(0, min(top_n - 1, len(papers) - 1))
    return papers[idx]


def format_paper_context(paper: Dict[str, Any]) -> str:
    """Format parsed paper data into a context string for downstream agents."""
    innovations_text = "\n".join(f"- {i}" for i in paper.get("innovations", []))
    return (
        f"Title: {paper['title']}\n"
        f"Authors: {paper['authors']}\n"
        f"Score: {paper['score']}/100 ({paper['classification']})\n"
        f"Domain: {paper['domain']}\n"
        f"Assessment: {paper['assessment']}\n"
        f"Key Innovations:\n{innovations_text}"
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class LinkedInPostState(TypedDict):
    """Shared state passed through the LangGraph pipeline."""

    main_task: str
    research_findings: Annotated[List[str], operator.add]
    draft: str
    critique_notes: str
    revision_number: int
    next_step: str
    current_sub_task: str
    paper_context: str


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = """You are a content project supervisor managing a blog post creation workflow.

Current Task: {main_task}

Current State:
- Research Insights: {research_findings}
- Blog Draft: {draft}
- Reviewer Feedback: {critique_notes}
- Revision Number: {revision_number}

Your goal is to ensure a clear, engaging, and valuable blog post for a LinkedIn audience that includes both technical and semi-technical readers.

Decide the next step and respond ONLY with a JSON object (no extra text):

{{
  "next_step": "researcher" or "writer" or "END",
  "task_description": "Brief description of what needs to be done next"
}}

Decision Rules:
- If no research exists, choose "researcher"
- If research exists but no draft, choose "writer"
- If draft exists and reviewer says "APPROVED", choose "END"
- If draft needs revision, choose "writer"
- If revision_number >= {max_revisions}, choose "END"
"""

RESEARCHER_PROMPT = """You are an insights researcher for a LinkedIn tech blog.

Research Topic: {task}

Your goal is to find relevant, up-to-date, and actionable insights for professionals in tech or semi-tech roles. Focus on:
- Key trends, challenges, or innovations
- Real-world use cases or success stories
- Supporting data or quotes from credible sources
- Simple explanations for semi-technical readers

Summarize your findings concisely, avoiding jargon. Include short citations or links to credible sources where applicable.
"""

RESEARCHER_WITH_CONTEXT_PROMPT = """You are an insights researcher for a LinkedIn tech blog.

A specific research paper has already been identified. Your job is NOT to research the topic from scratch.
Instead, find **supplemental context** around this paper:

{paper_context}

Search for:
- Industry reactions or commentary on this paper or its techniques
- Related news, applications, or real-world deployments of similar approaches
- Practical implications for professionals
- Supporting data or quotes from credible sources

Summarize your findings concisely. Include citations or links where applicable.
"""

WRITER_PROMPT = """
You are a professional Linkedin writer.

Main Task: {main_task}

Research Insights:
{research_findings}

Previous Draft (if any):
{draft}

Reviewer Feedback (if any):
{critique_notes}

Instructions:
- If there is no draft, create a new one based on research.
- If there is a draft AND reviewer feedback, revise the draft accordingly.
- Write a professional LinkedIn post that is clear, engaging, and skimmable.
"""

WRITER_WITH_CONTEXT_PROMPT = """You are a professional LinkedIn writer.

Main Task: {main_task}

Paper Context:
{paper_context}

Research Insights:
{research_findings}

Previous Draft (if any):
{draft}

Reviewer Feedback (if any):
{critique_notes}

Instructions:
- Center the post around this specific paper's findings and innovations.
- If there is no draft, create a new one based on the paper context and supplemental research.
- If there is a draft AND reviewer feedback, revise the draft accordingly.
- Write a professional LinkedIn post that is clear, engaging, and skimmable.
- Weave the paper's key innovations and assessment into the narrative naturally.
- Make the content accessible to both technical and semi-technical readers.
"""

CRITIQUE_PROMPT = """
You are a critical reviewer evaluating a content for Linkedin post.

Main Task: {main_task}

Draft to Review:
{draft}

Evaluate the draft based on:
1. Hook Strength - Does the opening grab attention?
2. Clarity - Is the message easy to understand?
3. Value - Does the post offer real insights, lessons, or actionable takeaways?
4. Structure - Are paragraphs short and skimmable for LinkedIn?
5. Engagement Potential - Will readers feel motivated to comment, react, or share?
6. Tone - Is it authentic, professional, and appropriate for LinkedIn?

Provide your evaluation:
- If the draft is satisfactory (minor issues are okay), respond with: "APPROVED - [brief positive comment]"
- If the draft needs improvement, provide specific, actionable feedback for revision

Your response:
"""

GROUNDEDNESS_PROMPT = """You are a Groundedness Checker AI.

Your job is to evaluate whether the draft is fully supported by the given research findings.

Given:
- Research Findings: {research_findings}
- Draft: {draft}

Instructions:
1. Identify each factual claim made in the draft.
2. For each claim, verify if it is directly supported by at least one research finding.
3. List:
   - Fully / Partially supported claims
   - Unsupported or hallucinated claims
4. Provide a Groundedness Score from 0 to 5:
   - 5 = All claims fully supported
   - 4 = Minor ungrounded phrasing
   - 3 = Several claims need verification
   - 2 = Mostly unsupported
   - 1 = Major unsupported statements
   - 0 = Completely ungrounded
5. Suggest specific corrections only for unsupported claims.

Return JSON like:
{{
  "supported": [...],
  "unsupported": [...],
  "score": X,
  "notes": "..."
}}
"""


# ---------------------------------------------------------------------------
# Agent Factories
# ---------------------------------------------------------------------------

def _create_supervisor_chain(
    llm: ChatAnthropic, max_revisions: int
) -> Any:
    """Create the supervisor decision function."""

    def supervisor_invoke(state: LinkedInPostState) -> Dict[str, str]:
        research = state.get("research_findings", [])
        research_text = "\n---\n".join(research) if research else "No research yet."

        revision = state.get("revision_number", 0)
        has_research = len(research) > 0
        has_draft = bool(state.get("draft", "").strip())
        critique = state.get("critique_notes", "")

        if "APPROVED" in critique.upper() and has_draft:
            logger.info("Supervisor: Draft approved, ending workflow")
            return {"next_step": "END", "task_description": "Report approved and complete"}

        if not has_research:
            logger.info("Supervisor: No research yet, directing to researcher")
            return {
                "next_step": "researcher",
                "task_description": f"Research the topic: {state.get('main_task', '')}",
            }

        if has_research and not has_draft:
            logger.info("Supervisor: Have research, creating first draft")
            return {"next_step": "writer", "task_description": "Write the first draft based on research findings"}

        if has_draft and not critique:
            logger.info("Supervisor: Have draft, sending to critiquer")
            return {"next_step": "writer", "task_description": "Prepare draft for critique"}

        if critique and "APPROVED" not in critique.upper() and revision < max_revisions:
            logger.info("Supervisor: Revision %d, sending back to writer", revision)
            return {"next_step": "writer", "task_description": "Revise the draft based on critique feedback"}

        if revision >= max_revisions:
            logger.info("Supervisor: Max revisions reached, ending")
            return {"next_step": "END", "task_description": "Maximum revisions reached, finalizing report"}

        prompt = SUPERVISOR_PROMPT.format(
            main_task=state.get("main_task", ""),
            research_findings=research_text,
            draft=state.get("draft", "No draft yet."),
            critique_notes=critique if critique else "No critique yet.",
            revision_number=revision,
            max_revisions=max_revisions,
        )

        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            text = content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(l for l in lines if not l.strip().startswith("```"))
                text = text.strip()
            decision = json.loads(text)
            if "next_step" in decision:
                return decision
        except Exception as exc:
            logger.warning("LLM parsing error: %s", exc)

        logger.info("Supervisor: Using final fallback - continuing with writer")
        return {"next_step": "writer", "task_description": "Continue with draft creation"}

    return supervisor_invoke


def _create_researcher_agent(
    llm: ChatAnthropic, tavily_tool: TavilySearch
) -> Any:
    """Create the researcher agent function."""

    def researcher_invoke(input_dict: Dict[str, str]) -> Dict[str, str]:
        query = input_dict.get("input", "")
        try:
            search_response = tavily_tool.invoke({"query": query})
            results = search_response if isinstance(search_response, list) else search_response.get("results", [])

            formatted_results: List[str] = []
            if results:
                for result in results[:3]:
                    if isinstance(result, dict):
                        title = result.get("title", "Untitled")
                        url = result.get("url", "N/A")
                        content = result.get("content", "")
                    else:
                        title = "Result"
                        url = "N/A"
                        content = str(result)[:300]
                    formatted_results.append(f"**{title}**\nSource: {url}\n{content[:300]}...\n")

            raw_output = "\n---\n".join(formatted_results) if formatted_results else "No results found"

            summary_prompt = (
                f"Based on these search results about '{query}', "
                "provide a concise summary of key findings (5-7 bullet points):\n"
                f"{raw_output}\n"
                "Format as clear bullet points with the most important information."
            )
            summary_response = llm.invoke(summary_prompt)
            summary = summary_response.content if hasattr(summary_response, "content") else str(summary_response)
            return {"output": summary if summary else raw_output, "input": query}

        except Exception as exc:
            logger.error("Research error: %s", exc)
            return {
                "output": f"Research completed on: {query}. Key information has been gathered from web sources.",
                "input": query,
            }

    return researcher_invoke


def _create_writer_chain(llm: ChatAnthropic) -> Any:
    """Create the writer chain function."""

    def writer_invoke(state: LinkedInPostState) -> str:
        research = state.get("research_findings", [])
        research_text = "\n\n".join(research) if research else "No research available."
        paper_ctx = state.get("paper_context", "")

        if paper_ctx:
            prompt = WRITER_WITH_CONTEXT_PROMPT.format(
                main_task=state.get("main_task", ""),
                paper_context=paper_ctx,
                research_findings=research_text,
                draft=state.get("draft", ""),
                critique_notes=state.get("critique_notes", ""),
            )
        else:
            prompt = WRITER_PROMPT.format(
                main_task=state.get("main_task", ""),
                research_findings=research_text,
                draft=state.get("draft", ""),
                critique_notes=state.get("critique_notes", ""),
            )

        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            return content if content else "Draft in progress..."
        except Exception as exc:
            logger.error("Writer error: %s", exc)
            return "Error generating draft. Please try again."

    return writer_invoke


def _create_critique_chain(llm: ChatAnthropic, max_revisions: int) -> Any:
    """Create the critique chain function."""

    def critique_invoke(state: LinkedInPostState) -> str:
        draft = state.get("draft", "")
        revision_num = state.get("revision_number", 0)

        if len(draft.strip()) < 100:
            return "APPROVED - Draft is minimal but acceptable."
        if revision_num >= max_revisions:
            return "APPROVED - Maximum revisions reached. The report is satisfactory."

        prompt = CRITIQUE_PROMPT.format(
            main_task=state.get("main_task", ""),
            draft=draft,
        )
        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            return content if content else "APPROVED"
        except Exception as exc:
            logger.error("Critique error: %s", exc)
            return "APPROVED - Error in critique, proceeding with current draft."

    return critique_invoke


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

def _build_supervisor_node(
    supervisor_chain: Any,
) -> Any:
    """Build the supervisor graph node."""

    def supervisor_node(state: LinkedInPostState) -> Dict[str, str]:
        print("\n=== SUPERVISOR ===")
        decision = supervisor_chain(state)
        next_step = decision.get("next_step", "researcher")
        task_desc = decision.get("task_description", "Continue work")
        print(f"Decision: {next_step}")
        print(f"Task: {task_desc}")
        return {"next_step": next_step, "current_sub_task": task_desc}

    return supervisor_node


def _build_research_node(
    researcher_agent: Any,
) -> Any:
    """Build the researcher graph node.

    When ``paper_context`` is present in the state the node pre-seeds
    ``research_findings`` with it and constructs a targeted Tavily query
    from the paper title + first innovation instead of the raw main_task.
    """

    def _extract_tavily_query(paper_ctx: str) -> str:
        """Derive a web-search query from the structured paper context."""
        title_m = re.search(r"Title:\s*(.+)", paper_ctx)
        innov_m = re.search(r"Key Innovations:\n- (.+)", paper_ctx)
        parts: List[str] = []
        if title_m:
            parts.append(title_m.group(1).strip())
        if innov_m:
            parts.append(innov_m.group(1).strip())
        return " ".join(parts) if parts else paper_ctx[:200]

    def research_node(state: LinkedInPostState) -> Dict[str, Any]:
        print("\n=== RESEARCHER ===")
        paper_ctx = state.get("paper_context", "")

        if paper_ctx:
            query = _extract_tavily_query(paper_ctx)
            print(f"Researching supplemental context: {query[:80]}...")
            try:
                result = researcher_agent({"input": query})
                supplemental = result.get("output", "Research completed")
                print(f"Found: {str(supplemental)[:100]}...")
            except Exception as exc:
                logger.error("Research error: %s", exc)
                supplemental = f"Supplemental research on paper - information gathered"
            return {"research_findings": [paper_ctx, supplemental]}

        sub_task = state.get("current_sub_task", state.get("main_task"))
        print(f"Researching: {sub_task}")
        try:
            result = researcher_agent({"input": sub_task})
            findings = result.get("output", "Research completed")
            print(f"Found: {str(findings)[:100]}...")
        except Exception as exc:
            logger.error("Research error: %s", exc)
            findings = f"Research on {sub_task} - information gathered"
        return {"research_findings": [findings]}

    return research_node


def _build_write_node(writer_chain: Any) -> Any:
    """Build the writer graph node."""

    def write_node(state: LinkedInPostState) -> Dict[str, Any]:
        print("\n=== WRITER ===")
        draft = writer_chain(state)
        print(f"Draft created: {len(draft)} characters")
        return {"draft": draft, "revision_number": state.get("revision_number", 0) + 1}

    return write_node


def _build_critique_node(critique_chain: Any) -> Any:
    """Build the critique graph node."""

    def critique_node(state: LinkedInPostState) -> Dict[str, str]:
        print("\n=== CRITIQUER ===")
        critique = critique_chain(state)
        print(f"Critique: {critique[:100]}...")
        is_approved = "APPROVED" in critique.upper()
        if is_approved:
            print(">>> Draft APPROVED")
            return {"critique_notes": "APPROVED", "next_step": "END"}
        print(">>> Revisions needed")
        return {"critique_notes": critique, "next_step": "writer"}

    return critique_node


# ---------------------------------------------------------------------------
# Workflow Builder
# ---------------------------------------------------------------------------

def build_linkedin_graph(config: AppConfig) -> Any:
    """Build and compile the LangGraph LinkedIn post creation pipeline."""
    llm = _build_llm(config)
    tavily_tool = _build_tavily(config)

    supervisor_chain = _create_supervisor_chain(llm, config.max_revisions)
    researcher_agent = _create_researcher_agent(llm, tavily_tool)
    writer_chain = _create_writer_chain(llm)
    critique_chain = _create_critique_chain(llm, config.max_revisions)

    workflow = StateGraph(LinkedInPostState)
    workflow.add_node("supervisor", _build_supervisor_node(supervisor_chain))
    workflow.add_node("researcher", _build_research_node(researcher_agent))
    workflow.add_node("writer", _build_write_node(writer_chain))
    workflow.add_node("critiquer", _build_critique_node(critique_chain))

    workflow.set_entry_point("supervisor")
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("writer", "critiquer")
    workflow.add_edge("critiquer", "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next_step", "researcher"),
        {"researcher": "researcher", "writer": "writer", "END": END},
    )

    return workflow.compile()


# ---------------------------------------------------------------------------
# Groundedness Evaluation
# ---------------------------------------------------------------------------

def evaluate_groundedness(
    config: AppConfig,
    research_findings: List[str],
    draft: str,
) -> Dict[str, Any]:
    """Evaluate factual groundedness of the draft against research findings."""
    eval_llm = _build_llm(config, temperature=0.0)
    prompt = GROUNDEDNESS_PROMPT.format(
        research_findings=research_findings,
        draft=draft,
    )
    response = eval_llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)

    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.strip().startswith("```"))
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            try:
                return json.loads(text[json_start:json_end])
            except json.JSONDecodeError:
                pass
        return {"raw_evaluation": content, "score": -1, "notes": "Failed to parse structured evaluation"}


# ---------------------------------------------------------------------------
# Report Writer
# ---------------------------------------------------------------------------

_REPORTS_DIR = Path("reports")


def _save_post_report(
    topic: str,
    final_state: Dict[str, Any],
    output: Dict[str, Any],
) -> Path:
    """Write the final LinkedIn post and metadata to a markdown file.

    Returns:
        The path to the written report.
    """
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    report_path = _REPORTS_DIR / f"linkedin_post_{timestamp}.md"

    groundedness = output.get("groundedness", {})
    g_score = groundedness.get("score", "N/A")
    paper_ctx = final_state.get("paper_context", "")

    lines = [
        "# LinkedIn Post Report",
        "",
        f"**Topic:** {topic}",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Revisions:** {final_state.get('revision_number', 0)}",
        f"**Processing time:** {output.get('processing_time', 0):.1f}s",
    ]

    if groundedness:
        lines.append(f"**Groundedness score:** {g_score}/5")

    lines += ["", "---", ""]

    if paper_ctx:
        lines += ["## Paper Context", "", paper_ctx, "", "---", ""]

    lines += ["## Final Post", "", final_state.get("draft", ""), ""]

    if groundedness:
        lines += ["---", "", "## Groundedness Evaluation", ""]
        supported = groundedness.get("supported", [])
        unsupported = groundedness.get("unsupported", [])
        if supported:
            lines.append("**Supported claims:**")
            for claim in supported:
                lines.append(f"- {claim}")
            lines.append("")
        if unsupported:
            lines.append("**Unsupported claims:**")
            for claim in unsupported:
                lines.append(f"- {claim}")
            lines.append("")
        if groundedness.get("notes"):
            lines.append(f"**Notes:** {groundedness['notes']}")
            lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Post saved to: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_linkedin_post(
    topic: str,
    *,
    config: Optional[AppConfig] = None,
    stream: bool = True,
    run_groundedness: bool = True,
    paper_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full LinkedIn post creation pipeline.

    Args:
        topic: The subject / prompt for the LinkedIn post.
        config: Optional pre-built config; loaded from env if ``None``.
        stream: If ``True``, print incremental state updates.
        run_groundedness: If ``True``, evaluate factual groundedness after completion.
        paper_context: Pre-formatted paper data from an evaluation report.
            When provided the researcher uses it as a seed and performs
            supplemental web search around the specific paper.

    Returns:
        Final state dict including ``draft``, ``research_findings``, and optionally ``groundedness``.
    """
    if config is None:
        config = AppConfig.from_env()
    logging.getLogger().setLevel(getattr(logging, config.log_level, logging.INFO))

    app = build_linkedin_graph(config)

    initial_state: LinkedInPostState = {
        "main_task": topic,
        "research_findings": [],
        "draft": "",
        "critique_notes": "",
        "revision_number": 0,
        "next_step": "",
        "current_sub_task": "",
        "paper_context": paper_context or "",
    }

    print("=" * 70)
    print("Multi-Agent LinkedIn Post Creator")
    print("=" * 70)
    print(f"Topic: {topic}")
    if paper_context:
        print("Mode: Paper-seeded (from evaluation report)")
    print(f"Model: {config.anthropic_model}")
    print(f"Max revisions: {config.max_revisions}")
    print("-" * 70)

    start = time.time()

    if stream:
        final_state: Dict[str, Any] = dict(initial_state)
        for step in app.stream(initial_state):
            for node_name, node_output in step.items():
                print(f"\n--- {node_name} ---")
                if isinstance(node_output, dict):
                    final_state.update(node_output)
    else:
        final_state = app.invoke(initial_state)

    duration = time.time() - start

    print("\n" + "=" * 70)
    print(f"Completed in {duration:.1f}s")
    print(f"Revisions: {final_state.get('revision_number', 0)}")
    print("=" * 70)

    if final_state.get("draft"):
        print("\n--- FINAL DRAFT ---")
        try:
            print(final_state["draft"])
        except UnicodeEncodeError:
            import sys
            enc = sys.stdout.encoding or "utf-8"
            print(final_state["draft"].encode(enc, errors="replace").decode(enc))
        print("--- END DRAFT ---\n")

    output: Dict[str, Any] = dict(final_state)
    output["processing_time"] = duration

    if run_groundedness and final_state.get("draft") and final_state.get("research_findings"):
        print("\nRunning groundedness evaluation...")
        groundedness = evaluate_groundedness(
            config,
            final_state["research_findings"],
            final_state["draft"],
        )
        output["groundedness"] = groundedness
        score = groundedness.get("score", "N/A")
        print(f"Groundedness score: {score}/5")
        if isinstance(groundedness.get("unsupported"), list) and groundedness["unsupported"]:
            print(f"Unsupported claims: {len(groundedness['unsupported'])}")

    if final_state.get("draft"):
        report_path = _save_post_report(topic, final_state, output)
        output["report_path"] = str(report_path)

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent LinkedIn Post Creator \u2014 generates, critiques, and refines LinkedIn posts.",
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="Topic / prompt for the LinkedIn post (auto-derived when --from-report is used)",
    )
    parser.add_argument(
        "--from-report",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to an evaluation report .md file; seeds the post with the top-ranked paper",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=1,
        help="Which paper from the report to use (1 = highest scored, default: 1)",
    )
    parser.add_argument(
        "--no-groundedness",
        action="store_true",
        help="Skip the groundedness evaluation step",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming and run the graph with invoke only",
    )

    args = parser.parse_args()
    if args.topic is None and args.from_report is None:
        parser.error("Either --topic or --from-report is required")
    return args


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()

    paper_ctx: Optional[str] = None
    topic = args.topic

    if args.from_report:
        report_path = Path(args.from_report)
        if not report_path.exists():
            print(f"Report not found: {report_path}")
            raise SystemExit(1)
        paper_data = parse_evaluation_report(report_path, top_n=args.top_n)
        paper_ctx = format_paper_context(paper_data)
        if topic is None:
            topic = f"LinkedIn post about: {paper_data['title']}"
        logger.info("Seeded from report \u2014 paper: %s", paper_data["title"])

    try:
        run_linkedin_post(
            topic=topic,
            stream=not args.no_stream,
            run_groundedness=not args.no_groundedness,
            paper_context=paper_ctx,
        )
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
