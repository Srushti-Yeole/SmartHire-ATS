"""
Interactive Plotly Components and UI Visualizations for Streamlit Dashboard.
"""

from typing import Any, Dict, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_gauge_chart(score: float, title: str = "ATS Score") -> go.Figure:
    """Creates an interactive ATS score gauge indicator."""
    color = "#10B981" if score >= 75 else ("#F59E0B" if score >= 50 else "#EF4444")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"<b>{title}</b>", "font": {"size": 20, "color": "#1E3A8A"}},
            number={"suffix": "/100", "font": {"size": 36, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#E2E8F0",
                "steps": [
                    {"range": [0, 50], "color": "rgba(239, 68, 68, 0.12)"},
                    {"range": [50, 75], "color": "rgba(245, 158, 11, 0.12)"},
                    {"range": [75, 100], "color": "rgba(16, 185, 129, 0.12)"},
                ],
                "threshold": {
                    "line": {"color": "#1E3A8A", "width": 3},
                    "thickness": 0.8,
                    "value": 75,
                },
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=25, r=25, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"},
    )
    return fig


def create_radar_chart(breakdown: Dict[str, float]) -> go.Figure:
    """Creates a 5-pillar radar / spider chart for ATS component weights."""
    categories = [
        "Keywords (40%)",
        "Semantic Sim (30%)",
        "Experience (15%)",
        "Education (5%)",
        "Certs (10%)",
    ]

    values = [
        breakdown.get("keyword_matching", 0.0),
        breakdown.get("semantic_similarity", 0.0),
        breakdown.get("experience_matching", 0.0),
        breakdown.get("education_matching", 0.0),
        breakdown.get("certification_matching", 0.0),
    ]

    # Close the radar loop
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(30, 58, 138, 0.25)",
            line=dict(color="#1E3A8A", width=2.5),
            marker=dict(size=7, color="#2563EB"),
            name="Candidate Score",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10, color="#64748B"),
                gridcolor="#E2E8F0",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#1F2937", family="Inter, sans-serif"),
                gridcolor="#E2E8F0",
            ),
        ),
        showlegend=False,
        height=280,
        margin=dict(l=35, r=35, t=25, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_ranking_bar_chart(candidates: List[Dict[str, Any]]) -> go.Figure:
    """Creates a horizontal ranked leaderboard bar chart."""
    if not candidates:
        return go.Figure()

    names = []
    scores = []
    colors = []

    # Sort descending for display (bottom to top in horizontal bar)
    for c in reversed(candidates):
        name = c.get("candidate", {}).get("name", "Unknown")
        score = c.get("score_breakdown", {}).get("ats_score", 0.0)
        names.append(name)
        scores.append(score)
        colors.append("#10B981" if score >= 75 else ("#F59E0B" if score >= 50 else "#EF4444"))

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=names,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#1E293B", width=0.5)),
            text=[f"{s:.1f}" for s in scores],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white", size=12, family="Inter, sans-serif"),
        )
    )

    fig.update_layout(
        title="<b>Candidate ATS Score Comparison</b>",
        title_font=dict(size=15, color="#1E3A8A"),
        xaxis=dict(range=[0, 105], title="Composite ATS Score (0 - 100)", gridcolor="#F1F5F9"),
        yaxis=dict(title=""),
        height=max(260, len(candidates) * 45),
        margin=dict(l=10, r=20, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_skill_donut_chart(matched_count: int, missing_count: int) -> go.Figure:
    """Creates a donut chart displaying skill fulfillment."""
    labels = ["Matched Skills", "Missing Skills"]
    values = [matched_count, missing_count]
    colors = ["#10B981", "#EF4444"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors),
                textinfo="label+value",
                textfont=dict(size=11, family="Inter, sans-serif"),
            )
        ]
    )
    fig.update_layout(
        showlegend=False,
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_multi_candidate_radar_chart(candidates: List[Dict[str, Any]]) -> go.Figure:
    """Creates an overlaid radar chart comparing multiple candidates across the 5 pillars."""
    categories = [
        "Keywords (40%)",
        "Semantic Sim (30%)",
        "Experience (15%)",
        "Education (5%)",
        "Certs (10%)",
    ]
    categories_closed = categories + [categories[0]]

    palette = ["#2563EB", "#10B981", "#8B5CF6", "#F59E0B", "#EC4899", "#06B6D4"]
    fig = go.Figure()

    for idx, c in enumerate(candidates):
        color = palette[idx % len(palette)]
        name = c.get("name") or c.get("candidate", {}).get("name", f"Candidate {idx+1}")
        
        # Check where breakdown is stored (direct or score_breakdown)
        bd = c.get("score_breakdown") or c
        values = [
            bd.get("keyword_matching") or bd.get("keyword_score", 0.0),
            bd.get("semantic_similarity") or bd.get("semantic_score", 0.0),
            bd.get("experience_matching") or bd.get("experience_score", 0.0),
            bd.get("education_matching") or bd.get("education_score", 0.0),
            bd.get("certification_matching") or bd.get("certification_score", 0.0),
        ]
        values_closed = values + [values[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)",
                line=dict(color=color, width=2.5),
                marker=dict(size=6, color=color),
                name=f"{name} ({c.get('ats_score', 0):.1f})",
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10, color="#64748B"),
                gridcolor="#E2E8F0",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#1F2937", family="Inter, sans-serif"),
                gridcolor="#E2E8F0",
            ),
        ),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        height=360,
        margin=dict(l=35, r=35, t=25, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_pillar_progress_bars(breakdown: Dict[str, float]) -> str:
    """Generates clean HTML progress bars for the 5 ATS pillars."""
    kw = breakdown.get("keyword_matching", breakdown.get("keyword_score", 0.0))
    sem = breakdown.get("semantic_similarity", breakdown.get("semantic_score", 0.0))
    exp = breakdown.get("experience_matching", breakdown.get("experience_score", 0.0))
    edu = breakdown.get("education_matching", breakdown.get("education_score", 0.0))
    cert = breakdown.get("certification_matching", breakdown.get("certification_score", 0.0))

    pillars = [
        {"name": "Keyword Matching", "weight": "40%", "score": kw, "color": "#2563EB"},
        {"name": "Semantic Relevance", "weight": "30%", "score": sem, "color": "#0D9488"},
        {"name": "Experience Alignment", "weight": "15%", "score": exp, "color": "#7C3AED"},
        {"name": "Education Match", "weight": "5%", "score": edu, "color": "#D97706"},
        {"name": "Certification Match", "weight": "10%", "score": cert, "color": "#059669"},
    ]

    html = '<div style="margin-top: 0.5rem; margin-bottom: 1rem;">'
    for p in pillars:
        pct = max(0, min(100, p["score"]))
        html += f"""
        <div style="margin-bottom: 0.85rem;">
            <div style="display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 0.25rem;">
                <span style="font-weight: 600; color: #1E293B;">{p['name']} <span style="font-weight: 400; color: #64748B;">({p['weight']} weight)</span></span>
                <span style="font-weight: 700; color: {p['color']};">{p['score']:.1f}%</span>
            </div>
            <div style="background-color: #E2E8F0; border-radius: 9999px; height: 10px; overflow: hidden; width: 100%;">
                <div style="background-color: {p['color']}; width: {pct}%; height: 100%; border-radius: 9999px; transition: width 0.5s ease-in-out;"></div>
            </div>
        </div>
        """
    html += "</div>"
    return html


def apply_custom_css():
    """Injects custom CSS styles for enterprise HR SaaS look and feel."""
    import streamlit as st

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            html, body, [class*="css"] {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }
            .saas-header {
                background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
                color: white;
                padding: 1.6rem 2rem;
                border-radius: 14px;
                margin-bottom: 1.5rem;
                box-shadow: 0 10px 15px -3px rgba(30, 58, 138, 0.15);
            }
            .saas-card {
                background: #FFFFFF;
                border-radius: 12px;
                padding: 1.4rem;
                border: 1px solid #E2E8F0;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
                margin-bottom: 1rem;
            }
            .saas-metric-box {
                background: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 10px;
                padding: 1rem 1.2rem;
                text-align: center;
            }
            .badge-matched {
                display: inline-block;
                background-color: #ECFDF5;
                color: #065F46;
                padding: 0.3rem 0.75rem;
                border-radius: 9999px;
                font-size: 0.82rem;
                font-weight: 600;
                margin: 0.2rem;
                border: 1px solid #A7F3D0;
            }
            .badge-missing {
                display: inline-block;
                background-color: #FEF2F2;
                color: #991B1B;
                padding: 0.3rem 0.75rem;
                border-radius: 9999px;
                font-size: 0.82rem;
                font-weight: 600;
                margin: 0.2rem;
                border: 1px solid #FECACA;
            }
            .badge-recommend {
                display: inline-block;
                background-color: #EFF6FF;
                color: #1E40AF;
                padding: 0.3rem 0.75rem;
                border-radius: 9999px;
                font-size: 0.82rem;
                font-weight: 600;
                margin: 0.2rem;
                border: 1px solid #BFDBFE;
            }
            .badge-boost {
                display: inline-block;
                background-color: #F0FDF4;
                color: #166534;
                font-weight: 700;
                padding: 0.25rem 0.6rem;
                border-radius: 6px;
                border: 1px solid #86EFAC;
                font-size: 0.82rem;
                margin-left: 0.3rem;
            }
            .role-badge-recruiter {
                background-color: #EEF2FF;
                color: #4338CA;
                padding: 0.2rem 0.6rem;
                border-radius: 6px;
                font-size: 0.75rem;
                font-weight: 700;
                border: 1px solid #C7D2FE;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            .role-badge-candidate {
                background-color: #ECFDF5;
                color: #047857;
                padding: 0.2rem 0.6rem;
                border-radius: 6px;
                font-size: 0.75rem;
                font-weight: 700;
                border: 1px solid #A7F3D0;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }

            /* ==============================================================================
               STREAMLIT TOOLBAR & HEADER CUSTOMIZATION
               - Remove Deploy button completely
               - Remove footer ("Made with Streamlit")
               - Keep ONLY Light Mode and Dark Mode toggle (in Settings)
               - Remove Print, Record Screen, Rerun, Clear Cache, About
               - Hide left sidebar completely
               - Maximize main content to full width
               ============================================================================== */
            /* Hide Streamlit Deploy button completely */
            .stDeployButton,
            [data-testid="stDeployButton"],
            [data-testid="stAppDeployButton"],
            [data-testid="manage-app-button"],
            button[kind="header"] {
                display: none !important;
                visibility: hidden !important;
                opacity: 0 !important;
                pointer-events: none !important;
            }

            /* Hide Streamlit footer ("Made with Streamlit") */
            footer,
            [data-testid="stFooter"],
            .reportview-container .main footer {
                display: none !important;
                visibility: hidden !important;
                height: 0 !important;
            }

            /* Hide left sidebar completely so content uses full available width */
            [data-testid="stSidebar"],
            [data-testid="collapsedControl"] {
                display: none !important;
                width: 0 !important;
                min-width: 0 !important;
                max-width: 0 !important;
            }

            /* Maximize main content to full available width */
            .main .block-container {
                max-width: 100% !important;
                padding-top: 1rem !important;
                padding-bottom: 2rem !important;
                padding-left: 2.2rem !important;
                padding-right: 2.2rem !important;
            }

            /* In the Streamlit options menu (hamburger menu), keep only Settings (Light/Dark mode) */
            div[data-baseweb="popover"] ul[role="menu"] > li:nth-child(n+2),
            div[data-baseweb="popover"] ul[role="menu"] > hr,
            div[data-baseweb="popover"] ul[role="menu"] > div[role="separator"] {
                display: none !important;
            }

            /* Profile Avatar Popover Button Styling */
            div[data-testid="stPopover"] > button {
                border-radius: 9999px !important;
                border: 1.5px solid #CBD5E1 !important;
                background: #FFFFFF !important;
                color: #1E3A8A !important;
                font-weight: 600 !important;
                padding: 0.4rem 1rem !important;
                font-size: 0.88rem !important;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06) !important;
                transition: all 0.2s ease !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
            }
            div[data-testid="stPopover"] > button:hover {
                border-color: #2563EB !important;
                background: #F8FAFC !important;
                box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.12) !important;
            }

            /* Profile Dropdown Menu Styling */
            div[data-testid="stPopoverBody"] {
                border-radius: 12px !important;
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.12), 0 8px 10px -6px rgba(0, 0, 0, 0.06) !important;
                border: 1px solid #E2E8F0 !important;
                padding: 0.75rem !important;
                min-width: 250px !important;
            }
            div[data-testid="stPopoverBody"] button {
                text-align: left !important;
                justify-content: flex-start !important;
                border-radius: 8px !important;
                margin-bottom: 2px !important;
                font-size: 0.88rem !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

