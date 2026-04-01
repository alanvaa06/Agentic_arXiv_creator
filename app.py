"""Gradio web application for the Agentic ArXiv Creator pipelines.

Provides three modes:
1. Full Pipeline — research papers then generate a LinkedIn post.
2. Research Only — discover and evaluate papers.
3. LinkedIn Post from Report — upload an evaluation report to generate a post.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from gradio.themes import Base, GoogleFont
from gradio.themes.utils import colors, fonts, sizes

from linkedin_post_creator import (
    AppConfig as LinkedInConfig,
    format_paper_context,
    parse_evaluation_report,
    run_linkedin_post,
)
from research_multi_agent_system import (
    RUBRIC_REGISTRY,
    get_config as get_research_config,
    run_research,
)
from run_pipeline import _top_paper_context


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _build_linkedin_config(
    anthropic_key: str,
    tavily_key: str,
    model: str,
    max_tokens: int,
    max_revisions: int,
) -> LinkedInConfig:
    """Construct a LinkedInConfig directly from form values."""
    return LinkedInConfig(
        anthropic_api_key=anthropic_key,
        anthropic_model=model,
        max_tokens=max_tokens,
        tavily_api_key=tavily_key,
        max_revisions=max_revisions,
        log_level="INFO",
    )


def _setup_research_env(
    anthropic_key: str,
    model: str,
    max_tokens: int = 4096,
) -> None:
    """Inject API credentials into env and clear the cached research config.

    The research module uses ``@lru_cache`` on ``get_config()`` which reads
    from ``os.environ``.  We set the vars, clear the cache, then trigger a
    reload so subsequent calls inside ``run_research`` pick up the new values.
    Safe because concurrency_limit=1 in the Gradio app.
    """
    os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    os.environ["ANTHROPIC_MODEL"] = model
    os.environ["ANTHROPIC_MAX_TOKENS"] = str(max_tokens)
    get_research_config.cache_clear()


# ---------------------------------------------------------------------------
# Stdout capture
# ---------------------------------------------------------------------------

class _CaptureStdout:
    """Context manager that captures everything printed to stdout."""

    def __init__(self) -> None:
        self._buffer = io.StringIO()
        self._redirect: Optional[contextlib.redirect_stdout[io.StringIO]] = None

    def __enter__(self) -> "_CaptureStdout":
        self._redirect = contextlib.redirect_stdout(self._buffer)
        self._redirect.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._redirect is not None:
            self._redirect.__exit__(*exc)

    @property
    def text(self) -> str:
        return self._buffer.getvalue()


def _safe_capture(fn: Any, *args: Any, **kwargs: Any) -> Tuple[Any, str]:
    """Run *fn* while capturing stdout; returns ``(result, captured_log)``."""
    cap = _CaptureStdout()
    with cap:
        result = fn(*args, **kwargs)
    return result, cap.text


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _validate_keys(anthropic_key: str, tavily_key: str | None = None) -> Optional[str]:
    """Return an error message if required keys are empty, else None."""
    if not anthropic_key.strip():
        return "Anthropic API key is required."
    if tavily_key is not None and not tavily_key.strip():
        return "Tavily API key is required for this mode."
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _papers_to_table(evaluation_results: List[Dict[str, Any]]) -> str:
    """Format evaluated papers as a markdown table."""
    if not evaluation_results:
        return "*No papers were evaluated.*"

    sorted_papers = sorted(
        evaluation_results, key=lambda r: r.get("agi_score", 0), reverse=True
    )
    rows = ["| # | Title | Score | Classification | Assessment |"]
    rows.append("|---|-------|-------|----------------|------------|")
    for i, p in enumerate(sorted_papers, 1):
        title = p.get("paper_title", "\u2014")
        score = p.get("agi_score", 0)
        cls_ = p.get("agi_classification", "\u2014")
        assess = (p.get("overall_assessment") or "\u2014")[:120]
        rows.append(f"| {i} | {title} | {score:.1f} | {cls_} | {assess} |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Runner functions
# ---------------------------------------------------------------------------

def run_full_pipeline(
    anthropic_key: str,
    tavily_key: str,
    model: str,
    max_revisions: int,
    groundedness: bool,
    query: str,
    domain: str,
    max_papers: int,
    top_n: int,
) -> Tuple[str, str, str, Optional[str]]:
    """Full pipeline: research \u2192 LinkedIn post.

    Returns (agent_log, papers_table, final_post, report_filepath).
    """
    err = _validate_keys(anthropic_key, tavily_key)
    if err:
        return err, "", "", None

    if not query.strip():
        return "Please enter a research query.", "", "", None

    try:
        _setup_research_env(anthropic_key, model)

        research_result, research_log = _safe_capture(
            run_research, query=query, max_papers=int(max_papers), domain=domain
        )

        evaluation_results = research_result.get("evaluation_results", [])
        papers_table = _papers_to_table(evaluation_results)

        if not evaluation_results:
            return (
                research_log + "\n\nNo papers were evaluated \u2014 cannot generate post.",
                papers_table,
                "",
                None,
            )

        paper_ctx, paper_data = _top_paper_context(evaluation_results, top_n=int(top_n))
        topic = f"LinkedIn post about: {paper_data['title']}"

        li_config = _build_linkedin_config(
            anthropic_key, tavily_key, model, 4096, int(max_revisions)
        )
        li_result, li_log = _safe_capture(
            run_linkedin_post,
            topic=topic,
            config=li_config,
            paper_context=paper_ctx,
            run_groundedness=groundedness,
        )

        full_log = research_log + "\n" + li_log
        draft = li_result.get("draft", "")
        report_path = li_result.get("report_path")
        return full_log, papers_table, draft, report_path

    except Exception as exc:
        return f"Error: {exc}", "", "", None


def run_research_only(
    anthropic_key: str,
    model: str,
    query: str,
    domain: str,
    max_papers: int,
) -> Tuple[str, str]:
    """Research only mode.

    Returns (agent_log, papers_table).
    """
    err = _validate_keys(anthropic_key)
    if err:
        return err, ""

    if not query.strip():
        return "Please enter a research query.", ""

    try:
        _setup_research_env(anthropic_key, model)
        result, log = _safe_capture(
            run_research, query=query, max_papers=int(max_papers), domain=domain
        )
        papers_table = _papers_to_table(result.get("evaluation_results", []))
        return log, papers_table

    except Exception as exc:
        return f"Error: {exc}", ""


def run_linkedin_from_report(
    anthropic_key: str,
    tavily_key: str,
    model: str,
    max_revisions: int,
    groundedness: bool,
    report_file: Optional[str],
    topic_override: str,
    top_n: int,
) -> Tuple[str, str, Optional[str]]:
    """Generate a LinkedIn post from an uploaded evaluation report.

    Returns (agent_log, final_post, report_filepath).
    """
    err = _validate_keys(anthropic_key, tavily_key)
    if err:
        return err, "", None

    if report_file is None:
        return "Please upload an evaluation report (.md file).", "", None

    report_path = Path(report_file)
    if not report_path.exists():
        return f"Report file not found: {report_path}", "", None

    try:
        paper_data = parse_evaluation_report(report_path, top_n=int(top_n))
        paper_ctx = format_paper_context(paper_data)

        topic = topic_override.strip() if topic_override.strip() else f"LinkedIn post about: {paper_data['title']}"

        li_config = _build_linkedin_config(
            anthropic_key, tavily_key, model, 4096, int(max_revisions)
        )
        li_result, log = _safe_capture(
            run_linkedin_post,
            topic=topic,
            config=li_config,
            paper_context=paper_ctx,
            run_groundedness=groundedness,
        )
        draft = li_result.get("draft", "")
        result_report = li_result.get("report_path")
        return log, draft, result_report

    except Exception as exc:
        return f"Error: {exc}", "", None


# ---------------------------------------------------------------------------
# Theme & Styling
# ---------------------------------------------------------------------------

ARXIV_RED = colors.Color(
    name="arxiv_red",
    c50="#FEF2F2",
    c100="#FEE2E2",
    c200="#FECACA",
    c300="#FCA5A5",
    c400="#F87171",
    c500="#DC2626",
    c600="#B31B1B",
    c700="#991B1B",
    c800="#7F1D1D",
    c900="#641414",
    c950="#450A0A",
)

ARXIV_GRAY = colors.Color(
    name="arxiv_gray",
    c50="#FAFAF9",
    c100="#F5F5F4",
    c200="#E7E5E4",
    c300="#D6D3D1",
    c400="#A8A29E",
    c500="#78716C",
    c600="#57534E",
    c700="#44403C",
    c800="#292524",
    c900="#1C1917",
    c950="#0C0A09",
)


def _build_theme() -> Base:
    """Build a light arXiv-inspired theme with Cornell Red accents."""
    theme = Base(
        primary_hue=ARXIV_RED,
        secondary_hue=ARXIV_GRAY,
        neutral_hue=ARXIV_GRAY,
        font=[
            GoogleFont("Source Sans 3"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ],
        font_mono=[
            GoogleFont("JetBrains Mono"),
            "ui-monospace",
            "monospace",
        ],
    )

    theme.set(
        body_background_fill="#FFFFFF",
        body_background_fill_dark="#FFFFFF",
        body_text_color="#1C1917",
        body_text_color_dark="#1C1917",
        body_text_color_subdued="#57534E",
        body_text_color_subdued_dark="#57534E",

        background_fill_primary="#FFFFFF",
        background_fill_primary_dark="#FFFFFF",
        background_fill_secondary="#FAFAF9",
        background_fill_secondary_dark="#FAFAF9",

        block_background_fill="#FAFAF9",
        block_background_fill_dark="#FAFAF9",
        block_border_color="#E7E5E4",
        block_border_color_dark="#E7E5E4",
        block_label_background_fill="#F5F5F4",
        block_label_background_fill_dark="#F5F5F4",
        block_label_text_color="#44403C",
        block_label_text_color_dark="#44403C",
        block_title_text_color="#1C1917",
        block_title_text_color_dark="#1C1917",
        block_shadow="0 1px 3px rgba(0, 0, 0, 0.06)",
        block_shadow_dark="0 1px 3px rgba(0, 0, 0, 0.06)",

        border_color_accent="#B31B1B",
        border_color_accent_dark="#B31B1B",
        border_color_primary="#E7E5E4",
        border_color_primary_dark="#E7E5E4",

        button_primary_background_fill="#B31B1B",
        button_primary_background_fill_dark="#B31B1B",
        button_primary_background_fill_hover="#991B1B",
        button_primary_background_fill_hover_dark="#991B1B",
        button_primary_text_color="#FFFFFF",
        button_primary_text_color_dark="#FFFFFF",
        button_primary_border_color="#B31B1B",
        button_primary_border_color_dark="#B31B1B",

        button_secondary_background_fill="#F5F5F4",
        button_secondary_background_fill_dark="#F5F5F4",
        button_secondary_background_fill_hover="#E7E5E4",
        button_secondary_background_fill_hover_dark="#E7E5E4",
        button_secondary_text_color="#1C1917",
        button_secondary_text_color_dark="#1C1917",

        input_background_fill="#FFFFFF",
        input_background_fill_dark="#FFFFFF",
        input_border_color="#D6D3D1",
        input_border_color_dark="#D6D3D1",
        input_border_color_focus="#B31B1B",
        input_border_color_focus_dark="#B31B1B",
        input_placeholder_color="#A8A29E",
        input_placeholder_color_dark="#A8A29E",

        panel_background_fill="#FAFAF9",
        panel_background_fill_dark="#FAFAF9",
        panel_border_color="#E7E5E4",
        panel_border_color_dark="#E7E5E4",

        checkbox_background_color="#FFFFFF",
        checkbox_background_color_dark="#FFFFFF",
        checkbox_background_color_selected="#B31B1B",
        checkbox_background_color_selected_dark="#B31B1B",
        checkbox_border_color="#D6D3D1",
        checkbox_border_color_dark="#D6D3D1",
        checkbox_label_text_color="#1C1917",
        checkbox_label_text_color_dark="#1C1917",

        slider_color="#B31B1B",
        slider_color_dark="#B31B1B",

        table_even_background_fill="#FAFAF9",
        table_even_background_fill_dark="#FAFAF9",
        table_odd_background_fill="#FFFFFF",
        table_odd_background_fill_dark="#FFFFFF",
        table_border_color="#E7E5E4",
        table_border_color_dark="#E7E5E4",

        shadow_drop="0 1px 4px rgba(0, 0, 0, 0.06)",
        shadow_drop_lg="0 4px 12px rgba(0, 0, 0, 0.08)",
    )
    return theme


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&display=swap');

.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
}

h1 {
    font-family: 'Libre Baskerville', 'Georgia', serif !important;
    font-weight: 700 !important;
    color: #B31B1B !important;
    letter-spacing: -0.01em !important;
}

h1 + p, .md p:first-child {
    color: #57534E !important;
}

button[role="tab"] {
    color: #78716C !important;
    border-bottom: 2px solid transparent !important;
    transition: color 200ms ease-out, border-color 200ms ease-out !important;
}
button[role="tab"]:hover {
    color: #1C1917 !important;
}
button[role="tab"][aria-selected="true"] {
    color: #B31B1B !important;
    border-bottom-color: #B31B1B !important;
}

button.primary {
    transition: box-shadow 200ms ease-out, background 200ms ease-out !important;
}
button.primary:hover {
    box-shadow: 0 2px 8px rgba(179, 27, 27, 0.25) !important;
}

.label-wrap {
    color: #44403C !important;
}

.upload-text {
    color: #78716C !important;
}

.prose, .md {
    color: #1C1917 !important;
}
.prose table th {
    background: #F5F5F4 !important;
    color: #44403C !important;
    border-bottom: 2px solid #B31B1B !important;
}
.prose table td {
    border-color: #E7E5E4 !important;
}

textarea[aria-label="Agent Log"] {
    font-family: 'JetBrains Mono', 'Consolas', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.55 !important;
    color: #44403C !important;
    background: #FAFAF9 !important;
}

*:focus-visible {
    outline: 2px solid #B31B1B !important;
    outline-offset: 2px !important;
}
"""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

DOMAIN_CHOICES = list(RUBRIC_REGISTRY.keys())

def build_app() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""
    with gr.Blocks(title="Agentic ArXiv Creator") as app:
        gr.Markdown(
            "# Agentic ArXiv Creator\n"
            "Multi-agent research pipeline and LinkedIn post generation, "
            "powered by LangGraph and Anthropic Claude."
        )

        # --- API Keys (always visible) ---
        with gr.Row():
            anthropic_key = gr.Textbox(
                label="Anthropic API Key",
                type="password",
                placeholder="sk-ant-\u2026",
                scale=1,
            )
            tavily_key = gr.Textbox(
                label="Tavily API Key",
                type="password",
                placeholder="tvly-\u2026",
                scale=1,
            )

        # --- Advanced Settings ---
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                model_dd = gr.Dropdown(
                    label="Model",
                    choices=[
                        "claude-sonnet-4-20250514",
                        "claude-haiku-4-20250414",
                    ],
                    value="claude-sonnet-4-20250514",
                )
                max_rev_slider = gr.Slider(
                    label="Max Revisions",
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=3,
                )
                groundedness_cb = gr.Checkbox(
                    label="Run Groundedness Check",
                    value=True,
                )

        # --- Tabs ---
        with gr.Tabs():

            # ---- Tab 1: Full Pipeline ----
            with gr.TabItem("Full Pipeline"):
                with gr.Row():
                    fp_query = gr.Textbox(
                        label="Research Query",
                        placeholder="e.g. recent advances in AGI\u2026",
                        scale=3,
                    )
                    fp_domain = gr.Dropdown(
                        label="Domain",
                        choices=DOMAIN_CHOICES,
                        value="agi",
                        scale=1,
                    )
                with gr.Row():
                    fp_max_papers = gr.Slider(
                        label="Max Papers",
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=5,
                    )
                    fp_top_n = gr.Slider(
                        label="Top N (paper to feature)",
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=1,
                    )
                fp_run_btn = gr.Button("Run Full Pipeline", variant="primary")

                fp_log = gr.Textbox(label="Agent Log", lines=15, interactive=False)
                fp_papers = gr.Markdown(label="Evaluated Papers")
                fp_post = gr.Markdown(label="Final LinkedIn Post")
                fp_file = gr.File(label="Download Report")

                fp_run_btn.click(
                    fn=run_full_pipeline,
                    inputs=[
                        anthropic_key,
                        tavily_key,
                        model_dd,
                        max_rev_slider,
                        groundedness_cb,
                        fp_query,
                        fp_domain,
                        fp_max_papers,
                        fp_top_n,
                    ],
                    outputs=[fp_log, fp_papers, fp_post, fp_file],
                    concurrency_limit=1,
                )

            # ---- Tab 2: Research Only ----
            with gr.TabItem("Research Only"):
                with gr.Row():
                    ro_query = gr.Textbox(
                        label="Research Query",
                        placeholder="e.g. transformer architectures for time series\u2026",
                        scale=3,
                    )
                    ro_domain = gr.Dropdown(
                        label="Domain",
                        choices=DOMAIN_CHOICES,
                        value="agi",
                        scale=1,
                    )
                ro_max_papers = gr.Slider(
                    label="Max Papers",
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=5,
                )
                ro_run_btn = gr.Button("Run Research", variant="primary")

                ro_log = gr.Textbox(label="Agent Log", lines=15, interactive=False)
                ro_papers = gr.Markdown(label="Evaluated Papers")

                ro_run_btn.click(
                    fn=run_research_only,
                    inputs=[
                        anthropic_key,
                        model_dd,
                        ro_query,
                        ro_domain,
                        ro_max_papers,
                    ],
                    outputs=[ro_log, ro_papers],
                    concurrency_limit=1,
                )

            # ---- Tab 3: LinkedIn Post from Report ----
            with gr.TabItem("LinkedIn Post (from Report)"):
                lr_file = gr.File(
                    label="Upload Evaluation Report (.md)",
                    file_types=[".md"],
                )
                lr_topic = gr.Textbox(
                    label="Topic Override (optional)",
                    placeholder="Leave blank to auto-derive from report",
                )
                lr_top_n = gr.Slider(
                    label="Top N (which paper to use)",
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=1,
                )
                lr_run_btn = gr.Button("Generate LinkedIn Post", variant="primary")

                lr_log = gr.Textbox(label="Agent Log", lines=15, interactive=False)
                lr_post = gr.Markdown(label="Final LinkedIn Post")
                lr_report_file = gr.File(label="Download Report")

                lr_run_btn.click(
                    fn=run_linkedin_from_report,
                    inputs=[
                        anthropic_key,
                        tavily_key,
                        model_dd,
                        max_rev_slider,
                        groundedness_cb,
                        lr_file,
                        lr_topic,
                        lr_top_n,
                    ],
                    outputs=[lr_log, lr_post, lr_report_file],
                    concurrency_limit=1,
                )

    return app


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_app()
    demo.launch(theme=_build_theme(), css=CUSTOM_CSS)
