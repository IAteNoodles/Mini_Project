"""Streamlit UI for running bias detection with LIME or SHAP explanations."""
from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components
from bias_detector import BiasDetector, get_detector

LLM_MODEL_NAME = "granite4:3b"
TOP_FACTOR_LIMIT = 5

st.set_page_config(page_title="Bias Detector", page_icon="📰", layout="wide")


def _inject_global_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --primary-color: #f72585;
                --secondary-color: #4361ee;
                --card-bg: rgba(255, 255, 255, 0.85);
                --text-color: #151515;
                --border-color: rgba(255, 255, 255, 0.45);
                font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }
            body {
                background: radial-gradient(circle at top left, #ffe5f1, #f0f4ff 55%, #fafafa);
            }
            [data-testid="block-container"] {
                padding-top: 2rem;
                padding-bottom: 3rem;
            }
            .hero-card {
                background: linear-gradient(120deg, rgba(247, 37, 133, 0.9), rgba(67, 97, 238, 0.9));
                padding: 1.5rem;
                border-radius: 18px;
                color: #fff;
                box-shadow: 0 20px 35px rgba(15, 23, 42, 0.25);
                margin-bottom: 1.5rem;
            }
            .hero-card h1 {
                font-size: 2.2rem;
                margin-bottom: 0.3rem;
            }
            .hero-card p {
                margin: 0;
                font-size: 1.05rem;
            }
            .section-label {
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.15em;
                color: #6b7280;
                margin-bottom: 0.3rem;
            }
            .glass-card {
                background: var(--card-bg);
                border-radius: 16px;
                padding: 1.25rem 1.4rem;
                border: 1px solid var(--border-color);
                box-shadow: 0 25px 40px rgba(15, 23, 42, 0.1);
                margin-bottom: 1.2rem;
            }
            .metric-container {
                display: flex;
                gap: 0.75rem;
                flex-wrap: wrap;
            }
            .metric-box {
                flex: 0 0 auto;
                min-width: 200px;
                max-width: 320px;
                background: rgba(255,255,255,0.92);
                border-radius: 10px;
                border: 1px solid rgba(67, 97, 238, 0.12);
                padding: 0.85rem 1rem;
            }
            .metric-box h3 {
                margin: 0;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.18em;
                color: #7c8db0;
            }
            .metric-box p {
                margin: 0.2rem 0 0;
                font-size: 1.1rem;
                color: #111;
            }
            .metric-box small {
                display: block;
                margin-top: 0.1rem;
                font-size: 0.8rem;
                color: #5b6375;
            }
            .granite-card {
                border-left: 4px solid var(--secondary-color);
            }
            .shap-heatmap-container {
                background: #fff;
                border-radius: 18px;
                padding: 1.25rem;
                border: 1px solid rgba(0,0,0,0.05);
                box-shadow: 0 18px 45px rgba(15,23,42,0.08);
                margin-bottom: 1rem;
            }
            .status-callout {
                margin-top: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_global_styles()

st.markdown(
    """
    <div class="hero-card">
        <h1>📰 Bias Detector with Explainable Highlights</h1>
        <p>Paste an article snippet, compare LIME vs. SHAP, and let Granite4 narrate the rationale.
        GPU-accelerated Hugging Face models keep latency low while token-level scores stay transparent.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def _format_probability_table(detector: BiasDetector, probabilities: List[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Bias Label": [detector.display_label(label) for label in detector.class_names],
            "Probability": [round(prob, 4) for prob in probabilities],
        }
    ).sort_values("Probability", ascending=False)


def _render_lime_explanation(exp) -> pd.DataFrame:
    contributions = exp.as_list()
    df = pd.DataFrame(contributions, columns=["Token", "Weight"])
    return df


def _wrap_shap_html(html: str) -> str:
    return """<div class='shap-heatmap-container'>""" + html + "</div>"


def _build_doc_style_bar(explanation_slice: shap.Explanation, max_display: int) -> Figure:
    fig = plt.figure(figsize=(7.5, max(3.8, 0.6 * max_display + 1)))
    shap.plots.bar(explanation_slice, show=False, max_display=max_display)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def _top_factors_with_other(
    df: pd.DataFrame,
    value_column: str,
    top_k: int = TOP_FACTOR_LIMIT,
) -> Tuple[pd.DataFrame, float]:
    if df.empty:
        return df.copy(), 0.0
    working = df.copy()
    working["abs"] = working[value_column].abs()
    working = working.sort_values("abs", ascending=False)
    top_slice = working.head(top_k).drop(columns=["abs"])
    remainder = working.iloc[top_k:]
    remainder_sum = float(remainder[value_column].sum()) if not remainder.empty else 0.0
    if not remainder.empty:
        other_row = pd.DataFrame({"Token": ["Other tokens"], value_column: [remainder_sum]})
        top_slice = pd.concat([top_slice, other_row], ignore_index=True)
    return top_slice, remainder_sum


def _format_contributions_for_prompt(df: pd.DataFrame, value_column: str, limit: int = 15) -> str:
    working = df.copy()
    working["abs"] = working[value_column].abs()
    trimmed = working.sort_values("abs", ascending=False).head(limit)
    return "\n".join(
        f"- {row['Token']}: {row[value_column]:.4f}" for _, row in trimmed.drop(columns=["abs"]).iterrows()
    )


def _summarize_with_granite(
    *,
    text: str,
    friendly_label: str,
    method: str,
    contributions: pd.DataFrame,
    contribution_column: str,
) -> str:
    try:
        import ollama  # type: ignore
    except ImportError:
        return "Install the `ollama` package and ensure Granite4 is pulled to enable narrative summaries."

    contribution_block = _format_contributions_for_prompt(contributions, contribution_column)
    prompt = f"""
You are a media literacy analyst. Explain why the model predicted {friendly_label} bias.

ARTICLE SNIPPET (verbatim, do not add facts beyond this):
\"\"\"{text}\"\"\"

EXPLANATION SOURCE (only evidence you may rely on):
Method: {method}
Token contributions (higher magnitude = stronger influence):
{contribution_block}

STRICT INSTRUCTIONS:
- Ground every statement in BOTH the snippet and the listed token contributions.
- Reference the contributing tokens verbatim. If a factor is "Other tokens", describe it generically without guessing content.
- Do not invent new phrases, events, or motivations.
- Output Markdown with these sections only:
  ### Tone Snapshot — one sentence tying the overall tone to {friendly_label}.
  ### Evidence From Tokens — bullet list linking the top tokens to their impact direction.
  ### Neutral Rewrite Tips — numbered list of up to 3 changes derived strictly from the provided evidence.
Keep the total response under 90 words and never discuss topics outside the supplied snippet.
    """

    try:
        response = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You turn SHAP or LIME contributions into friendly narratives."},
                {"role": "user", "content": prompt},
            ],
        )
        message = response.get("message", {})
        return message.get("content", "Granite4 did not return any content.").strip()
    except Exception as exc:  # pragma: no cover - best effort safeguard
        return f"Unable to query Granite4: {exc}"


def main() -> None:
    detector = get_detector()

    st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
    user_text = st.text_area(
        "Paste article text",
        height=220,
        placeholder="Write or paste at least a couple of sentences...",
    )

    num_features = st.slider(
        "Explanation detail (tokens)",
        min_value=10,
        max_value=50,
        value=20,
        step=2,
    )
    explanation_mode = st.radio(
        "Choose an explanation method",
        options=["SHAP", "LIME"],
        index=0,
        horizontal=True,
    )
    show_explanations = st.toggle(
        "Show explanation visuals",
        value=True,
        help="Disable to focus on the verdict and narrative while skipping charts.",
    )

    if st.button("Detect Bias", type="primary"):
        if not user_text.strip():
            st.warning("Please provide some text before running the detector.")
            return

        explanation_table: pd.DataFrame | None = None
        explanation_viz_df: pd.DataFrame | None = None
        shap_html: str | None = None
        shap_bar_fig: Figure | None = None
        narrative: str | None = None
        other_sum = 0.0
        friendly_label = ""
        prob_df = pd.DataFrame()
        contribution_column = "Contribution"

        with st.status("Starting analysis...", expanded=False) as status:
            status.update(label="Running detector", state="running")
            prediction = detector.predict(user_text)
            probabilities = detector.predict_proba([user_text])[0]
            friendly_label = detector.display_label(prediction.label)
            prob_df = _format_probability_table(detector, list(probabilities))

            status.update(label="Building explanation", state="running")
            if explanation_mode == "LIME":
                explanation = detector.explain_lime(user_text, num_features=num_features)
                lime_df = _render_lime_explanation(explanation)
                explanation_table = lime_df.rename(columns={"Weight": contribution_column})
                explanation_viz_df, other_sum = _top_factors_with_other(
                    explanation_table,
                    contribution_column,
                )
            else:
                shap_explanation = detector.shap_explain(user_text)
                target_label = prediction.label
                shap_df = detector.shap_dataframe(shap_explanation, target_label)
                explanation_table = shap_df.rename(columns={"SHAP Value": contribution_column})
                explanation_viz_df, other_sum = _top_factors_with_other(
                    explanation_table,
                    contribution_column,
                )
                shap_html = detector.shap_text_html(shap_explanation, target_label)
                label_slice = detector.shap_label_slice(shap_explanation, target_label)
                shap_bar_fig = _build_doc_style_bar(label_slice, max_display=TOP_FACTOR_LIMIT)

            if explanation_table is not None:
                status.update(label="Summarizing via Granite4", state="running")
                narrative = _summarize_with_granite(
                    text=user_text,
                    friendly_label=friendly_label,
                    method=explanation_mode,
                    contributions=explanation_table,
                    contribution_column=contribution_column,
                )

            status.update(label="Analysis complete", state="complete")

        left_col, right_col = st.columns([1.2, 0.8])

        with left_col:
            st.markdown('<div class="section-label">Model verdict</div>', unsafe_allow_html=True)
            st.markdown(
                f"<div class='glass-card metric-container'>\n"
                f"  <div class='metric-box'><h3>Top label</h3><p>{friendly_label}</p>"
                f"  <small>confidence {prediction.score:.2%}</small></div>\n"
                f"</div>",
                unsafe_allow_html=True,
            )

        if explanation_table is not None and explanation_viz_df is not None:
            if show_explanations:
                if explanation_mode == "LIME":
                    with left_col:
                        st.markdown("#### LIME Highlights")
                        st.caption(
                            "Top contributors (absolute weight) with the remainder collapsed into 'Other tokens'."
                        )
                        st.bar_chart(explanation_viz_df.set_index("Token"))
                        st.caption(f"Combined contribution from other tokens: {other_sum:+.4f}")
                        with st.expander("Full contribution table (LIME)"):
                            st.dataframe(
                                explanation_table,
                                width="stretch",
                                use_container_width=True,
                                height=220,
                            )
                else:
                    with left_col:
                        st.markdown("#### SHAP Highlights")
                        st.caption(
                            f"Tokens most responsible for **{friendly_label}** with the rest merged into 'Other tokens'."
                        )
                        if shap_html:
                            components.html(_wrap_shap_html(shap_html), height=430, scrolling=False)
                        if shap_bar_fig is not None:
                            st.markdown(
                                "**Bars show the top impacts; the 'Other tokens' bar captures the remainder of the text.**"
                            )
                            st.pyplot(shap_bar_fig, clear_figure=True)
                        st.caption(f"Combined contribution from other tokens: {other_sum:+.4f}")
                        with st.expander("Full contribution table (SHAP)"):
                            st.dataframe(
                                explanation_table,
                                width="stretch",
                                use_container_width=True,
                                height=240,
                            )
            else:
                with left_col:
                    st.info(
                        "Explanations are hidden. Toggle 'Show explanation visuals' back on to view charts and tables."
                    )

        with right_col:
            if explanation_table is not None and narrative:
                st.markdown('<div class="section-label">Narrative</div>', unsafe_allow_html=True)
                with st.expander("Explain these token contributions", expanded=False):
                    st.markdown(
                        f"<div class='glass-card granite-card'>{narrative}</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown('<div class="section-label">Class probabilities</div>', unsafe_allow_html=True)
            st.dataframe(
                prob_df,
                use_container_width=True,
                height=210,
            )
            if explanation_table is not None:
                with st.expander("Top contributing tokens"):
                    st.dataframe(
                        explanation_table.rename(columns={contribution_column: "Weight"}),
                        use_container_width=True,
                        height=250,
                    )


        st.success("Analysis complete")


if __name__ == "__main__":
    main()
