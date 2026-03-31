"""Multi-agent AGI research pipeline extracted from notebook.

This script keeps the original notebook purpose:
1) Plan research queries
2) Discover papers from arXiv
3) Evaluate AGI potential
4) Generate reports
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict

import arxiv
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from anthropic import APIError, APITimeoutError, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration loaded from environment variables."""

    anthropic_api_key: str
    anthropic_model: str
    max_tokens: int
    log_level: str
    report_output_dir: Path

    @staticmethod
    def from_env() -> "AppConfig":
        """Load and validate config from environment variables."""
        load_dotenv(override=False)

        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "Missing ANTHROPIC_API_KEY. Set it in your environment or .env file."
            )

        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514").strip()
        max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096"))
        log_level = os.getenv("LOG_LEVEL", "INFO").upper().strip()
        output_dir = Path(os.getenv("REPORT_OUTPUT_DIR", "reports")).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        return AppConfig(
            anthropic_api_key=api_key,
            anthropic_model=model,
            max_tokens=max_tokens,
            log_level=log_level,
            report_output_dir=output_dir,
        )


def build_llm(config: AppConfig, temperature: float = 0.0) -> ChatAnthropic:
    """Create a configured LLM client."""
    return ChatAnthropic(
        model=config.anthropic_model,
        temperature=temperature,
        anthropic_api_key=config.anthropic_api_key,
        max_tokens=config.max_tokens,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("research_system")


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Lazy-load config so importing the module does not require secrets."""
    config = AppConfig.from_env()
    logging.getLogger().setLevel(getattr(logging, config.log_level, logging.INFO))
    return config


class ResearchPhase(Enum):
    """Phases of the research workflow."""

    INITIALIZATION = "initialization"
    PLANNING = "planning"
    DISCOVERY = "discovery"
    EVALUATION = "evaluation"
    COMPLETION = "completion"


class ResearchSystemState(TypedDict):
    """Shared state passed through all nodes in the LangGraph pipeline."""

    request_id: str
    research_objective: str
    current_phase: ResearchPhase
    phase_history: List[Dict[str, Any]]
    discovered_papers: List[Dict[str, Any]]
    evaluation_results: List[Dict[str, Any]]
    synthesis_data: Optional[Dict[str, Any]]
    final_report: Optional[str]
    messages: List[BaseMessage]
    errors: List[Dict[str, Any]]
    total_processing_time: float
    max_papers: int


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((APIError, RateLimitError, APITimeoutError)),
)
def call_llm(llm: ChatAnthropic, messages: Sequence[BaseMessage]) -> BaseMessage:
    """Call LLM with automatic retry on transient failures."""
    return llm.invoke(messages)


def search_arxiv(
    query: str,
    max_papers: int = 10,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search arXiv for AI/ML research papers."""
    logger.info(
        "ArXiv search: query='%s', max=%s, from=%s, to=%s",
        query,
        max_papers,
        from_date,
        to_date,
    )

    try:
        categories = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE"]
        cat_filter = " OR ".join(f"cat:{c}" for c in categories)
        full_query = f"({query}) AND ({cat_filter})"

        if from_date or to_date:
            fd = (
                datetime.strptime(from_date, "%Y-%m-%d").strftime("%Y%m%d0000")
                if from_date
                else "202001010000"
            )
            td = (
                datetime.strptime(to_date, "%Y-%m-%d").strftime("%Y%m%d2359")
                if to_date
                else datetime.now().strftime("%Y%m%d2359")
            )
            full_query += f" AND submittedDate:[{fd} TO {td}]"

        logger.info("Full arXiv query: %s", full_query)

        client = arxiv.Client()
        search = arxiv.Search(
            query=full_query,
            max_results=max_papers,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        papers: List[Dict[str, Any]] = []
        for result in client.results(search):
            papers.append(
                {
                    "id": result.get_short_id(),
                    "title": result.title,
                    "link": result.entry_id,
                    "metadata": {
                        "authors": [
                            {"name": author.name, "affiliation": ""}
                            for author in result.authors
                        ],
                        "abstract": result.summary.replace("\n", " ").strip(),
                        "published_date": result.published.isoformat(),
                        "updated_date": result.updated.isoformat(),
                        "categories": result.categories,
                        "source": "arxiv",
                        "doi": getattr(result, "doi", None),
                        "journal_ref": getattr(result, "journal_ref", None),
                    },
                }
            )

        logger.info("ArXiv returned %s papers", len(papers))
        return papers
    except Exception as exc:
        logger.error("ArXiv search failed: %s", exc)
        return []


@tool
def discover_and_process_papers(
    query: str,
    max_papers: int = 10,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Search arXiv, deduplicate papers, and validate minimum content quality."""
    start_time = time.time()
    papers = search_arxiv(query, max_papers, from_date, to_date)

    seen_titles: set[str] = set()
    unique_papers: List[Dict[str, Any]] = []
    for paper in papers:
        normalized_title = re.sub(r"\s+", " ", paper["title"].lower().strip())
        if normalized_title not in seen_titles:
            seen_titles.add(normalized_title)
            unique_papers.append(paper)

    valid_papers = [
        paper
        for paper in unique_papers
        if paper.get("title")
        and len(paper.get("metadata", {}).get("abstract", "")) >= 50
    ]

    processing_time = time.time() - start_time
    return {
        "processed_papers": valid_papers,
        "statistics": {
            "initial_count": len(papers),
            "after_deduplication": len(unique_papers),
            "final_count": len(valid_papers),
            "duplicates_removed": len(papers) - len(unique_papers),
            "invalid_removed": len(unique_papers) - len(valid_papers),
            "processing_time": f"{processing_time:.2f}s",
        },
        "source_counts": {"arxiv": len(valid_papers)},
        "search_metadata": {
            "query_used": query,
            "date_range": (
                f"{from_date} to {to_date}" if from_date or to_date else "all_time"
            ),
            "sources_searched": ["arxiv"],
            "processing_timestamp": datetime.now().isoformat(),
        },
    }


DISCOVERY_TOOLS = [discover_and_process_papers]


AGI_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "novel_problem_solving": {"weight": 0.15, "desc": "Solving new, unseen problems"},
    "few_shot_learning": {"weight": 0.15, "desc": "Learning from minimal examples"},
    "task_transfer": {"weight": 0.15, "desc": "Applying skills across domains"},
    "abstract_reasoning": {
        "weight": 0.12,
        "desc": "Logical thinking and pattern recognition",
    },
    "contextual_adaptation": {"weight": 0.10, "desc": "Adapting behavior to context"},
    "multi_rule_integration": {
        "weight": 0.10,
        "desc": "Following multiple complex rules",
    },
    "generalization_efficiency": {
        "weight": 0.08,
        "desc": "Generalizing from small data",
    },
    "meta_learning": {"weight": 0.08, "desc": "Learning how to learn"},
    "world_modeling": {"weight": 0.04, "desc": "Modeling complex environments"},
    "autonomous_goal_setting": {
        "weight": 0.03,
        "desc": "Setting and pursuing own objectives",
    },
}


def calculate_agi_score(parameter_scores: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
    """Calculate a weighted AGI score (0-100) from individual parameter scores (1-10)."""
    total_weighted = 0.0
    total_weight = 0.0
    contributions: Dict[str, Dict[str, Any]] = {}

    for name, cfg in AGI_PARAMETERS.items():
        if name in parameter_scores:
            score = float(parameter_scores[name])
            weight = float(cfg["weight"])
            contribution = score * weight
            total_weighted += contribution
            total_weight += weight
            contributions[name] = {
                "score": score,
                "weight": weight,
                "contribution": round(contribution, 1),
            }

    final_score = (total_weighted / total_weight) * 10 if total_weight > 0 else 0.0
    final_score = round(final_score, 1)

    if final_score >= 70:
        classification = "High AGI Potential"
    elif final_score >= 40:
        classification = "Medium AGI Potential"
    else:
        classification = "Low AGI Potential"

    return final_score, {
        "final_score": final_score,
        "classification": classification,
        "parameter_contributions": contributions,
        "total_weight_used": total_weight,
    }


def get_agi_evaluation_prompt(title: str, abstract: str, authors: List[str]) -> str:
    """Build the structured AGI evaluation prompt for one paper."""
    authors_str = ", ".join(authors[:5])
    return f"""EVALUATE THIS RESEARCH PAPER FOR AGI POTENTIAL

## PAPER DETAILS
Title: {title}
Authors: {authors_str}
Abstract: {abstract}

## EVALUATION TASK
Rate each AGI parameter on a 1-10 scale.

## AGI PARAMETERS
1. Novel Problem Solving
2. Few-Shot Learning
3. Task Transfer
4. Abstract Reasoning
5. Contextual Adaptation
6. Multi-Rule Integration
7. Generalization Efficiency
8. Meta-Learning
9. World Modeling
10. Autonomous Goal Setting

## REQUIRED OUTPUT FORMAT
Return ONLY valid JSON with:
{{
  "parameter_scores": {{
    "novel_problem_solving": {{"score": X, "reasoning": "..."}},
    "few_shot_learning": {{"score": X, "reasoning": "..."}},
    "task_transfer": {{"score": X, "reasoning": "..."}},
    "abstract_reasoning": {{"score": X, "reasoning": "..."}},
    "contextual_adaptation": {{"score": X, "reasoning": "..."}},
    "multi_rule_integration": {{"score": X, "reasoning": "..."}},
    "generalization_efficiency": {{"score": X, "reasoning": "..."}},
    "meta_learning": {{"score": X, "reasoning": "..."}},
    "world_modeling": {{"score": X, "reasoning": "..."}},
    "autonomous_goal_setting": {{"score": X, "reasoning": "..."}}
  }},
  "overall_agi_assessment": "2-3 sentence summary",
  "key_innovations": ["a", "b", "c"],
  "limitations": ["l1", "l2"],
  "confidence_level": "High/Medium/Low",
  "confidence_level_reason": "reason"
}}"""


def planner_node(state: ResearchSystemState) -> ResearchSystemState:
    """LangGraph node: create an execution plan from the research objective."""
    logger.info("=== PLANNER NODE ===")
    print("\n[PLANNER] Creating execution plan...")

    llm = build_llm(get_config(), temperature=0.2)
    objective = state.get("research_objective", "")
    desired_max_papers = int(state.get("max_papers", 10))
    today = datetime.now().strftime("%Y-%m-%d")

    system_prompt = """You are a Research Planning Specialist for an AGI research system.
Create a JSON execution plan for the given research objective.

You MUST derive date range from the query:
- past week / 1 week => 7 days back
- 2 weeks => 14 days back
- past month => 30 days back
- explicit dates => preserve them
- no period => default 7 days

Return ONLY JSON:
{
  "search_keywords": ["k1", "k2"],
  "search_strategy": {
    "primary_sources": ["arxiv"],
    "categories": ["cs.AI", "cs.LG", "cs.CL"],
    "date_range": "YYYY-MM-DD to YYYY-MM-DD",
    "max_papers_per_source": {desired_max_papers}
  },
  "focus_areas": ["a1", "a2"],
  "exclusions": []
}"""
    system_prompt = system_prompt.replace(
        "{desired_max_papers}", str(desired_max_papers)
    )

    user_prompt = f"""Research Objective: {objective}
Today's date is: {today}
Reference dates:
- 7 days ago:  {(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")}
- 14 days ago: {(datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")}
- 30 days ago: {(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")}"""

    try:
        response = call_llm(
            llm,
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
        )
        plan_text = str(response.content)

        if "```json" in plan_text:
            start = plan_text.find("```json") + 7
            end = plan_text.find("```", start)
            plan_text = plan_text[start:end]
        elif "```" in plan_text:
            start = plan_text.find("```") + 3
            end = plan_text.find("```", start)
            plan_text = plan_text[start:end]

        plan = json.loads(plan_text.strip())

        if not plan.get("search_strategy", {}).get("date_range"):
            fallback_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            plan.setdefault("search_strategy", {})["date_range"] = (
                f"{fallback_start} to {today}"
            )
        plan.setdefault("search_strategy", {})["max_papers_per_source"] = desired_max_papers
    except Exception as exc:
        logger.error("Planning failed (%s), using default plan", exc)
        fallback_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        plan = {
            "search_keywords": ["AGI", "artificial general intelligence", "general AI"],
            "search_strategy": {
                "primary_sources": ["arxiv"],
                "categories": ["cs.AI", "cs.LG"],
                "date_range": f"{fallback_start} to {today}",
                "max_papers_per_source": desired_max_papers,
            },
            "focus_areas": ["artificial general intelligence"],
            "exclusions": [],
        }

    if state.get("synthesis_data") is None:
        state["synthesis_data"] = {}
    state["synthesis_data"]["execution_plan"] = plan
    state["synthesis_data"]["plan_created_at"] = datetime.now().isoformat()
    return state


def discovery_node(state: ResearchSystemState) -> ResearchSystemState:
    """LangGraph node: discover papers using an LLM agent plus arXiv tool."""
    logger.info("=== DISCOVERY NODE ===")
    print("\n[DISCOVERY] Searching for papers on arXiv...")

    plan = state.get("synthesis_data", {}).get("execution_plan", {})
    if not plan:
        state["discovered_papers"] = []
        return state

    llm = build_llm(get_config())
    agent = create_agent(
        model=llm,
        tools=DISCOVERY_TOOLS,
        system_prompt=(
            "You are the Discovery Agent. Call discover_and_process_papers exactly once."
        ),
    )

    date_range = plan.get("search_strategy", {}).get("date_range", "")
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    if " to " in date_range:
        parts = date_range.split(" to ")
        from_date = parts[0].strip()
        to_date = parts[1].strip()

    max_papers = int(plan.get("search_strategy", {}).get("max_papers_per_source", 10))
    input_message = (
        f"EXECUTION PLAN\n"
        f"Keywords: {plan.get('search_keywords', [])}\n"
        f"Date Range: {from_date} to {to_date}\n"
        f"Max Papers: {max_papers}\n\n"
        f'Call discover_and_process_papers with from_date="{from_date}", '
        f'to_date="{to_date}", max_papers={max_papers}.'
    )

    discovered_papers: List[Dict[str, Any]] = []
    try:
        result = agent.invoke({"messages": [HumanMessage(content=input_message)]})
        for msg in result.get("messages", []):
            if type(msg).__name__ != "ToolMessage":
                continue
            try:
                content = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                if isinstance(content, dict) and "processed_papers" in content:
                    discovered_papers = content["processed_papers"]
                    if state.get("synthesis_data") is None:
                        state["synthesis_data"] = {}
                    state["synthesis_data"]["discovery_metadata"] = {
                        "statistics": content.get("statistics", {}),
                        "search_metadata": content.get("search_metadata", {}),
                    }
                    break
            except (json.JSONDecodeError, TypeError):
                continue
    except Exception as exc:
        logger.error("Discovery failed: %s", exc)
        state.setdefault("errors", []).append(
            {
                "phase": "discovery",
                "error": str(exc),
                "timestamp": datetime.now().isoformat(),
            }
        )

    state["discovered_papers"] = discovered_papers
    return state


def _clean_llm_json(text: str) -> str:
    """Extract and clean JSON from LLM response text."""
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        text = text[start:end] if end > start else text[start:]
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        text = text[start:end] if end > start else text[start:]

    json_start = text.find("{")
    json_end = text.rfind("}") + 1
    if json_start < 0 or json_end <= json_start:
        return ""
    text = text[json_start:json_end]
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)


def _parse_eval_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse evaluation JSON with multiple fallback strategies."""
    cleaned = _clean_llm_json(text)
    if not cleaned:
        return None

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    try:
        scores: Dict[str, Dict[str, Any]] = {}
        for param in AGI_PARAMETERS:
            pattern = rf"\"{param}\"\s*:\s*\{{\s*\"score\"\s*:\s*(\d+(?:\.\d+)?)"
            match = re.search(pattern, cleaned)
            if match:
                scores[param] = {
                    "score": float(match.group(1)),
                    "reasoning": "extracted via fallback",
                }
        if scores:
            assessment_match = re.search(
                r"\"overall_agi_assessment\"\s*:\s*\"([^\"]*)\"",
                cleaned,
            )
            innovations_match = re.search(
                r"\"key_innovations\"\s*:\s*\[(.*?)\]",
                cleaned,
                re.DOTALL,
            )
            return {
                "parameter_scores": scores,
                "overall_agi_assessment": (
                    assessment_match.group(1) if assessment_match else ""
                ),
                "key_innovations": (
                    re.findall(r"\"([^\"]+)\"", innovations_match.group(1))
                    if innovations_match
                    else []
                ),
                "limitations": [],
                "confidence_level": "Medium",
            }
    except Exception:
        return None

    return None


def _generate_detailed_report(
    results: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    request_id: str,
) -> str:
    """Generate a detailed markdown evaluation report."""
    if not results:
        return (
            "# Detailed AGI Evaluation Report\n\n"
            f"Request ID: {request_id}\n"
            f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n"
            "No papers were successfully evaluated.\n"
        )

    sorted_papers = sorted(results, key=lambda item: item.get("agi_score", 0), reverse=True)
    total = len(results)
    high = [paper for paper in sorted_papers if paper["agi_score"] >= 70]
    medium = [paper for paper in sorted_papers if 40 <= paper["agi_score"] < 70]
    low = [paper for paper in sorted_papers if paper["agi_score"] < 40]

    report = (
        "# Detailed AGI Evaluation Report\n\n"
        f"Request ID: {request_id}\n"
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"Total Papers Evaluated: {total}\n\n"
        "---\n\n"
        "## Executive Summary\n"
        f"- Average AGI Score: {metadata.get('avg_agi_score', 0):.1f}/100\n"
        f"- High AGI Potential (>=70): {len(high)} papers ({len(high)/total*100:.0f}%)\n"
        f"- Medium AGI Potential (40-69): {len(medium)} papers ({len(medium)/total*100:.0f}%)\n"
        f"- Low AGI Potential (<40): {len(low)} papers ({len(low)/total*100:.0f}%)\n\n"
        "---\n\n"
        "## Paper Analysis\n\n"
    )

    for label, papers_list in [
        ("HIGH AGI POTENTIAL (>=70)", high),
        ("MEDIUM AGI POTENTIAL (40-69)", medium),
        ("LOW AGI POTENTIAL (<40)", low),
    ]:
        if not papers_list:
            continue
        report += f"### {label}\n\n"
        for index, paper in enumerate(papers_list, start=1):
            authors_str = ", ".join(paper.get("paper_authors", [])[:3])
            report += (
                f"#### {index}. {paper['paper_title']}\n"
                f"Authors: {authors_str}\n"
                f"AGI Score: {paper['agi_score']:.1f}/100 ({paper['agi_classification']})\n"
                f"Assessment: {paper.get('overall_assessment', 'N/A')}\n\n"
            )
            innovations = paper.get("key_innovations", [])
            if innovations:
                report += "Key Innovations:\n"
                for innovation in innovations[:3]:
                    report += f"- {innovation}\n"
                report += "\n"
            report += "---\n\n"

    report += (
        "\n## Scoring Methodology\n\n"
        "10 weighted AGI parameters:\n"
        "- Novel Problem Solving 15%\n"
        "- Few-Shot Learning 15%\n"
        "- Task Transfer 15%\n"
        "- Abstract Reasoning 12%\n"
        "- Contextual Adaptation 10%\n"
        "- Multi-Rule Integration 10%\n"
        "- Generalization Efficiency 8%\n"
        "- Meta-Learning 8%\n"
        "- World Modeling 4%\n"
        "- Autonomous Goal Setting 3%\n\n"
        "*Report generated by Multi-Agent Research System*\n"
    )
    return report


def evaluation_node(state: ResearchSystemState) -> ResearchSystemState:
    """LangGraph node: evaluate each paper for AGI potential."""
    logger.info("=== EVALUATION NODE ===")
    papers = state.get("discovered_papers", [])
    if not papers:
        state["evaluation_results"] = []
        return state

    llm = build_llm(get_config())
    system_msg = SystemMessage(
        content=(
            "You are an expert AGI evaluator. "
            "Return only valid JSON in the requested format."
        )
    )

    results: List[Dict[str, Any]] = []
    total_score = 0.0
    for index, paper in enumerate(papers, start=1):
        title = paper.get("title", "Unknown")
        metadata = paper.get("metadata", {})
        abstract = metadata.get("abstract", "")
        authors_raw = metadata.get("authors", [])

        author_names: List[str] = []
        for author in authors_raw:
            if isinstance(author, dict):
                author_names.append(str(author.get("name", "Unknown")))
            elif isinstance(author, str):
                author_names.append(author)

        if len(abstract) < 50:
            logger.warning("Skipping paper %s due to short abstract", index)
            continue

        try:
            response = call_llm(
                llm,
                [
                    system_msg,
                    HumanMessage(content=get_agi_evaluation_prompt(title, abstract, author_names)),
                ],
            )
            eval_data = _parse_eval_json(str(response.content))
            if eval_data is None:
                continue

            parameter_scores = {
                name: value["score"]
                for name, value in eval_data.get("parameter_scores", {}).items()
                if isinstance(value, dict) and "score" in value
            }
            weighted_score, breakdown = calculate_agi_score(parameter_scores)

            results.append(
                {
                    "paper_id": paper.get("id", f"paper_{index}"),
                    "paper_title": title,
                    "paper_authors": author_names,
                    "paper_source": metadata.get("source", "unknown"),
                    "paper_url": paper.get("link", ""),
                    "evaluation_timestamp": datetime.now().isoformat(),
                    "agi_score": weighted_score,
                    "agi_classification": breakdown["classification"],
                    "parameter_scores": eval_data.get("parameter_scores", {}),
                    "overall_assessment": eval_data.get("overall_agi_assessment", ""),
                    "key_innovations": eval_data.get("key_innovations", []),
                    "limitations": eval_data.get("limitations", []),
                    "confidence_level": eval_data.get("confidence_level", "Medium"),
                    "score_breakdown": breakdown,
                }
            )
            total_score += weighted_score
        except Exception as exc:
            logger.error("Error evaluating paper %s: %s", index, exc)

    state["evaluation_results"] = results
    avg_score = round(total_score / len(results), 1) if results else 0
    eval_metadata = {
        "total_papers": len(papers),
        "successful_evaluations": len(results),
        "failed_evaluations": len(papers) - len(results),
        "avg_agi_score": avg_score,
        "score_distribution": {
            "high": len([r for r in results if r["agi_score"] >= 70]),
            "medium": len([r for r in results if 40 <= r["agi_score"] < 70]),
            "low": len([r for r in results if r["agi_score"] < 40]),
        },
        "processing_time": datetime.now().isoformat(),
    }

    if state.get("synthesis_data") is None:
        state["synthesis_data"] = {}
    state["synthesis_data"]["evaluation_metadata"] = eval_metadata

    request_id = state.get("request_id", "unknown")
    detailed_report_name = f"evaluation_detailed_report_{request_id}.md"
    detailed_report_path = get_config().report_output_dir / detailed_report_name
    detailed_report_path.write_text(
        _generate_detailed_report(results, eval_metadata, request_id),
        encoding="utf-8",
    )
    logger.info("Detailed report written to %s", detailed_report_path)
    return state


def _generate_final_report(state: ResearchSystemState) -> str:
    """Generate the executive summary report from evaluation results."""
    results = state.get("evaluation_results", [])
    papers = state.get("discovered_papers", [])
    metadata = state.get("synthesis_data", {}).get("evaluation_metadata", {})

    if not results:
        return (
            "# AGI Research Analysis Report\n\n"
            f"Objective: {state.get('research_objective', 'N/A')}\n\n"
            "No papers were successfully evaluated.\n"
        )

    sorted_results = sorted(results, key=lambda item: item.get("agi_score", 0), reverse=True)
    avg = metadata.get("avg_agi_score", 0)
    dist = metadata.get("score_distribution", {})

    report = (
        "# AGI Research Analysis Report\n\n"
        "## Executive Summary\n\n"
        f"Research Objective: {state.get('research_objective', 'N/A')}\n\n"
        "Discovery Overview:\n"
        f"- Total papers discovered: {len(papers)}\n"
        f"- Papers successfully evaluated: {len(results)}\n"
        f"- Average AGI score: {avg:.1f}/100\n\n"
        "AGI Potential Distribution:\n"
        f"- High AGI Potential: {dist.get('high', 0)} papers\n"
        f"- Medium AGI Potential: {dist.get('medium', 0)} papers\n"
        f"- Low AGI Potential: {dist.get('low', 0)} papers\n\n"
        "## Key Findings - Top Papers\n\n"
    )

    for index, paper in enumerate(sorted_results[:5], start=1):
        authors = paper.get("paper_authors", [])
        authors_str = ", ".join(authors[:3]) + ("..." if len(authors) > 3 else "")
        innovations = ", ".join(paper.get("key_innovations", [])[:3])
        report += (
            f"### {index}. {paper['paper_title']}\n"
            f"Authors: {authors_str}\n"
            f"AGI Score: {paper['agi_score']}/100 ({paper['agi_classification']})\n"
            f"Key Innovations: {innovations}\n"
            f"Assessment: {paper.get('overall_assessment', 'N/A')}\n\n"
        )

    high_count = dist.get("high", 0)
    report += "## Insights and Recommendations\n\n"
    if avg >= 70:
        report += "- Strong AGI advancement observed in top papers.\n"
    elif avg >= 40:
        report += "- Moderate AGI progress with room for breakthrough innovation.\n"
    else:
        report += "- Most papers remain in narrow-AI territory.\n"

    if high_count > 0:
        report += (
            f"- Breakthrough potential: {high_count} papers show high AGI potential.\n\n"
            "Recommended Actions:\n"
            "1. Deep analysis of high-potential papers.\n"
            "2. Track top research groups.\n"
            "3. Explore practical applications.\n"
        )
    else:
        report += (
            "\nRecommended Actions:\n"
            "1. Broaden search criteria.\n"
            "2. Extend search timeframe.\n"
            "3. Refine AGI-focused keywords.\n"
        )

    report += f"\n---\nReport generated on {datetime.now():%Y-%m-%d %H:%M:%S}\n"
    return report


def supervisor_node(state: ResearchSystemState) -> ResearchSystemState:
    """LangGraph node: coordinate phase transitions."""
    phase = state.get("current_phase", ResearchPhase.INITIALIZATION)
    logger.info("=== SUPERVISOR NODE === phase=%s", phase)

    if phase == ResearchPhase.INITIALIZATION:
        if not state.get("request_id"):
            state["request_id"] = str(uuid.uuid4())
        state["current_phase"] = ResearchPhase.PLANNING
    elif phase == ResearchPhase.PLANNING:
        if state.get("synthesis_data", {}).get("execution_plan"):
            state["current_phase"] = ResearchPhase.DISCOVERY
        else:
            state["current_phase"] = ResearchPhase.COMPLETION
            state["final_report"] = "Failed to create execution plan."
    elif phase == ResearchPhase.DISCOVERY:
        if len(state.get("discovered_papers", [])) > 0:
            state["current_phase"] = ResearchPhase.EVALUATION
        else:
            state["current_phase"] = ResearchPhase.COMPLETION
            state["final_report"] = "No papers discovered."
    elif phase == ResearchPhase.EVALUATION:
        state["final_report"] = _generate_final_report(state)
        state["current_phase"] = ResearchPhase.COMPLETION
        final_report_path = get_config().report_output_dir / "final_report.md"
        final_report_path.write_text(state["final_report"], encoding="utf-8")
        logger.info("Final report saved to %s", final_report_path)

    return state


def route_next_phase(
    state: ResearchSystemState,
) -> Literal["planner", "discovery", "evaluation", "complete"]:
    """Route from supervisor to the appropriate phase node."""
    phase = state.get("current_phase", ResearchPhase.INITIALIZATION)
    mapping: Dict[ResearchPhase, Literal["planner", "discovery", "evaluation"]] = {
        ResearchPhase.PLANNING: "planner",
        ResearchPhase.DISCOVERY: "discovery",
        ResearchPhase.EVALUATION: "evaluation",
    }
    return mapping.get(phase, "complete")


def build_research_graph() -> Any:
    """Build and compile the LangGraph research pipeline."""
    graph = StateGraph(ResearchSystemState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("planner", planner_node)
    graph.add_node("discovery", discovery_node)
    graph.add_node("evaluation", evaluation_node)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_next_phase,
        {
            "planner": "planner",
            "discovery": "discovery",
            "evaluation": "evaluation",
            "complete": END,
        },
    )
    for node_name in ["planner", "discovery", "evaluation"]:
        graph.add_edge(node_name, "supervisor")
    return graph.compile()


def run_research(query: str, max_papers: int = 10) -> Dict[str, Any]:
    """Run the full research pipeline for a query."""
    _ = get_config()
    logger.info("Starting research run for query: %s", query)
    print("=" * 70)
    print("Multi-Agent Research System")
    print("=" * 70)
    print(f"Research Query: {query}")
    print(f"Max Papers: {max_papers}")
    print("-" * 70)

    system = build_research_graph()
    initial_state: ResearchSystemState = {
        "messages": [HumanMessage(content=query)],
        "request_id": "",
        "research_objective": query,
        "current_phase": ResearchPhase.INITIALIZATION,
        "phase_history": [],
        "discovered_papers": [],
        "evaluation_results": [],
        "synthesis_data": {
            "execution_plan": {
                "search_strategy": {"max_papers_per_source": max_papers}
            }
        },
        "final_report": None,
        "errors": [],
        "total_processing_time": 0.0,
        "max_papers": max_papers,
    }

    start_time = time.time()
    result = system.invoke(initial_state)
    duration = time.time() - start_time
    result["total_processing_time"] = duration
    logger.info("Run completed in %.1fs", duration)

    print(f"\nCompleted in {duration:.1f}s")
    print(f"Papers discovered: {len(result.get('discovered_papers', []))}")
    print(f"Papers evaluated: {len(result.get('evaluation_results', []))}")
    print(f"Reports written to: {get_config().report_output_dir}")
    print("=" * 70)
    return result


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run AGI research multi-agent workflow extracted from notebook."
    )
    parser.add_argument("--query", required=True, help="Research query/objective")
    parser.add_argument("--max-papers", type=int, default=10, help="Maximum papers")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    try:
        run_research(query=args.query, max_papers=args.max_papers)
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
