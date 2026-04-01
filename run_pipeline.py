"""End-to-end orchestrator: research → LinkedIn post.

Chains ``run_research()`` from the research pipeline into
``run_linkedin_post()`` from the LinkedIn creator, passing the
top-ranked paper as seeded context.

Usage::

    python run_pipeline.py --query "deep learning portfolio optimization" \\
                           --domain finance --max-papers 5
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Any, List, Optional

from research_multi_agent_system import RUBRIC_REGISTRY, run_research
from linkedin_post_creator import format_paper_context, run_linkedin_post


def _top_paper_context(
    evaluation_results: List[Dict[str, Any]], top_n: int = 1
) -> tuple[str, Dict[str, Any]]:
    """Extract the Nth-ranked paper from evaluation results and format it."""
    if not evaluation_results:
        raise ValueError("Research pipeline returned no evaluated papers.")

    sorted_results = sorted(
        evaluation_results,
        key=lambda r: r.get("agi_score", 0),
        reverse=True,
    )
    idx = max(0, min(top_n - 1, len(sorted_results) - 1))
    paper = sorted_results[idx]

    paper_data: Dict[str, Any] = {
        "title": paper.get("paper_title", "Unknown"),
        "authors": ", ".join(paper.get("paper_authors", [])[:5]),
        "score": paper.get("agi_score", 0),
        "classification": paper.get("agi_classification", "Unknown"),
        "assessment": paper.get("overall_assessment", ""),
        "innovations": paper.get("key_innovations", []),
        "domain": "research",
    }
    return format_paper_context(paper_data), paper_data


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the combined pipeline."""
    parser = argparse.ArgumentParser(
        description="Research-to-LinkedIn pipeline: discover papers, then generate a LinkedIn post.",
    )
    parser.add_argument("--query", required=True, help="Research query / objective")
    parser.add_argument(
        "--domain",
        choices=list(RUBRIC_REGISTRY.keys()),
        default="agi",
        help="Evaluation domain (default: agi)",
    )
    parser.add_argument(
        "--max-papers", type=int, default=5, help="Max papers to discover (default: 5)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=1,
        help="Which ranked paper to feature (1 = best, default: 1)",
    )
    parser.add_argument(
        "--no-groundedness",
        action="store_true",
        help="Skip the groundedness evaluation on the LinkedIn post",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full research → LinkedIn post pipeline."""
    args = parse_args()

    print("=" * 70)
    print("STAGE 1 \u2014 Research Pipeline")
    print("=" * 70)
    try:
        research_result = run_research(
            query=args.query,
            max_papers=args.max_papers,
            domain=args.domain,
        )
    except ValueError as exc:
        print(f"Research configuration error: {exc}")
        sys.exit(2)

    evaluation_results = research_result.get("evaluation_results", [])
    if not evaluation_results:
        print("No papers were evaluated \u2014 cannot generate LinkedIn post.")
        sys.exit(1)

    paper_ctx, paper_data = _top_paper_context(evaluation_results, top_n=args.top_n)
    topic = f"LinkedIn post about: {paper_data['title']}"

    print()
    print("=" * 70)
    print("STAGE 2 \u2014 LinkedIn Post Creator")
    print("=" * 70)
    print(f"Featuring paper: {paper_data['title']}")
    print(f"Score: {paper_data['score']}/100 ({paper_data['classification']})")
    print("-" * 70)

    try:
        result = run_linkedin_post(
            topic=topic,
            paper_context=paper_ctx,
            run_groundedness=not args.no_groundedness,
        )
    except ValueError as exc:
        print(f"LinkedIn creator configuration error: {exc}")
        sys.exit(2)

    if result.get("groundedness"):
        score = result["groundedness"].get("score", "N/A")
        print(f"\nGroundedness score: {score}/5")

    if result.get("report_path"):
        print(f"Post report: {result['report_path']}")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
