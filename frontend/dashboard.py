"""
Streamlit Dashboard for AI Resume Screening & ATS Score Predictor.
Provides Login Authentication, Single Resume Screening, Multi-Resume Batch Ranking,
Interactive Plotly Charts, PDF & Excel Exports, Resume Optimization & Candidate Ranking.
"""

import io
import os
import sys
import pandas as pd
import streamlit as st
import inspect

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frontend.api_client import api_client
from frontend.components import (
    apply_custom_css,
    create_gauge_chart,
    create_radar_chart,
    create_ranking_bar_chart,
    create_skill_donut_chart,
)
from backend.report_generator import report_generator
from backend.similarity_engine import similarity_engine

# Streamlit width compatibility helper
def _width_kwarg(val="stretch"):
    try:
        sig = inspect.signature(st.button)
        if "width" in sig.parameters:
            return {"width": val}
    except Exception:
        pass
    return {"use_container_width": True if val == "stretch" else False}

ST_FULL_WIDTH = _width_kwarg("stretch")

# Page Configuration
st.set_page_config(
    page_title="AI Resume Screening & ATS Predictor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

apply_custom_css()

# Session State Initialization
if "user" not in st.session_state:
    st.session_state["user"] = None

# ==============================================================================
# AUTHENTICATION PAGE (SHOWN WHEN NOT LOGGED IN)
# ==============================================================================
if st.session_state.get("user") is None:
    st.markdown(
        """
        <div style="text-align: center; margin-top: 1.5rem; margin-bottom: 1.5rem;">
            <div style="display: inline-block; background: #EEF2FF; padding: 0.45rem 1.1rem; border-radius: 9999px; margin-bottom: 0.6rem; border: 1px solid #C7D2FE;">
                <span style="color: #3730A3; font-weight: 600; font-size: 0.88rem;">✨ Enterprise AI Resume Screening Platform</span>
            </div>
            <h1 style="color: #1E3A8A; font-size: 2.4rem; font-weight: 700; margin-bottom: 0.3rem;">
                AI Resume Screening & ATS Score Predictor
            </h1>
            <p style="color: #475569; font-size: 1.05rem; max-width: 600px; margin: 0 auto;">
                Please select your portal to sign in to the ATS screening dashboard.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    auth_col1, auth_col2, auth_col3 = st.columns([1, 2, 1])

    with auth_col2:
        # Separate Candidate and Recruiter Login
        auth_portal = st.radio(
            "Select Login Portal:",
            ["👤 Candidate Login", "💼 Recruiter Login"],
            horizontal=True,
            key="login_portal_choice",
        )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # 1. CANDIDATE LOGIN SECTION
        # ------------------------------------------------------------------
        if "Candidate" in auth_portal:
            st.markdown(
                """
                <div class="saas-card" style="border-top: 4px solid #10B981; margin-bottom: 1rem;">
                    <h3 style="margin-top: 0; margin-bottom: 0.3rem; color: #065F46;">Candidate Login</h3>
                    <p style="color: #64748B; font-size: 0.88rem; margin: 0;">Sign in to screen your resume, analyze match scores, and discover missing skills.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            cand_tab_login, cand_tab_reg = st.tabs(["🔑 Candidate Sign In", "📝 Register New Candidate"])

            with cand_tab_login:
                c_email = st.text_input("Candidate Email", value="", placeholder="e.g. candidate@example.com", key="cand_login_email")
                c_pw = st.text_input("Password", type="password", value="", placeholder="Enter your password", key="cand_login_pw")

                if st.button("Sign In as Candidate", type="primary", **ST_FULL_WIDTH, key="btn_cand_login"):
                    if not c_email.strip() or not c_pw:
                        st.error("Please enter your candidate email and password.")
                    else:
                        with st.spinner("Authenticating candidate..."):
                            auth_res = api_client.login(c_email.strip(), c_pw)
                            if auth_res.get("success"):
                                user = auth_res["user"]
                                if user.get("role") != "candidate":
                                    st.error("This account is registered as a Recruiter. Please switch to the Recruiter Login portal.")
                                else:
                                    st.session_state["user"] = user
                                    st.toast("Welcome back, Candidate!", icon="👤")
                                    st.rerun()
                            else:
                                st.error("Account not found or invalid credentials. You cannot sign in without an existing account. Please switch to the 'Register New Candidate' tab to create your account first.")

            with cand_tab_reg:
                st.markdown("<p style='font-size: 0.9rem; color: #475569;'>Create a new Candidate account to start evaluating your resumes.</p>", unsafe_allow_html=True)
                cr_email = st.text_input("Candidate Email", placeholder="e.g. candidate@example.com", key="cand_reg_email")
                cr_pw = st.text_input("Password (min 4 characters)", type="password", key="cand_reg_pw")
                cr_pw2 = st.text_input("Confirm Password", type="password", key="cand_reg_pw2")

                if st.button("Create Candidate Account", type="primary", **ST_FULL_WIDTH, key="btn_cand_register"):
                    if not cr_email.strip() or not cr_pw:
                        st.error("Please fill in all required fields.")
                    elif len(cr_pw) < 4:
                        st.error("Password must be at least 4 characters long.")
                    elif cr_pw != cr_pw2:
                        st.error("Passwords do not match.")
                    else:
                        with st.spinner("Registering candidate..."):
                            reg_res = api_client.register(cr_email.strip(), cr_pw, role="candidate")
                            if reg_res.get("success"):
                                st.session_state["user"] = reg_res["user"]
                                st.success("Account created successfully! Loading your candidate dashboard...")
                                st.rerun()
                            else:
                                st.error(reg_res.get("error", "Registration error."))

        # ------------------------------------------------------------------
        # 2. RECRUITER LOGIN SECTION
        # ------------------------------------------------------------------
        else:
            st.markdown(
                """
                <div class="saas-card" style="border-top: 4px solid #2563EB; margin-bottom: 1rem;">
                    <h3 style="margin-top: 0; margin-bottom: 0.3rem; color: #1E3A8A;">Recruiter Login</h3>
                    <p style="color: #64748B; font-size: 0.88rem; margin: 0;">Sign in to evaluate candidates, perform batch screening, and review rankings.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            rec_tab_login, rec_tab_reg = st.tabs(["🔑 Recruiter Sign In", "🏢 Register Organization"])

            with rec_tab_login:
                r_email = st.text_input("Recruiter Work Email", value="", placeholder="e.g. recruiter@company.com", key="rec_login_email")
                r_pw = st.text_input("Password", type="password", value="", placeholder="Enter your password", key="rec_login_pw")

                if st.button("Sign In as Recruiter", type="primary", **ST_FULL_WIDTH, key="btn_rec_login"):
                    if not r_email.strip() or not r_pw:
                        st.error("Please enter your recruiter email and password.")
                    else:
                        with st.spinner("Authenticating recruiter..."):
                            auth_res = api_client.login(r_email.strip(), r_pw)
                            if auth_res.get("success"):
                                user = auth_res["user"]
                                if user.get("role") != "recruiter":
                                    st.error("This account is registered as a Candidate. Please switch to the Candidate Login portal.")
                                else:
                                    st.session_state["user"] = user
                                    st.toast("Welcome, Recruiter!", icon="💼")
                                    st.rerun()
                            else:
                                st.error("Account not found or invalid credentials. You cannot sign in without an existing account. Please switch to the 'Register Organization' tab to create your account first.")

            with rec_tab_reg:
                st.markdown("<p style='font-size: 0.9rem; color: #475569;'>Register your company or organization to start batch screening candidates.</p>", unsafe_allow_html=True)
                rr_comp = st.text_input("Company / Organization Name", placeholder="e.g. Acme Corp", key="rec_reg_company")
                rr_email = st.text_input("Corporate Email", placeholder="e.g. recruiter@company.com", key="rec_reg_email")
                rr_pw = st.text_input("Password (min 4 characters)", type="password", key="rec_reg_pw")
                rr_pw2 = st.text_input("Confirm Password", type="password", key="rec_reg_pw2")

                if st.button("Register Recruiter Account", type="primary", **ST_FULL_WIDTH, key="btn_rec_register"):
                    if not rr_comp.strip() or not rr_email.strip() or not rr_pw:
                        st.error("Please provide company name, email, and password.")
                    elif len(rr_pw) < 4:
                        st.error("Password must be at least 4 characters long.")
                    elif rr_pw != rr_pw2:
                        st.error("Passwords do not match.")
                    else:
                        with st.spinner("Registering recruiter organization..."):
                            reg_res = api_client.register(rr_email.strip(), rr_pw, role="recruiter", company_name=rr_comp.strip())
                            if reg_res.get("success"):
                                st.session_state["user"] = reg_res["user"]
                                st.success("Organization account registered! Loading recruiter dashboard...")
                                st.rerun()
                            else:
                                st.error(reg_res.get("error", "Registration error."))

    # Stop rendering the rest of the dashboard until logged in
    st.stop()



# ==============================================================================
# MAIN DASHBOARD (SHOWN WHEN LOGGED IN)
# ==============================================================================
current_user = st.session_state.get("user") or {}
user_email = current_user.get("email", "User")


# Role Check
is_candidate = (current_user.get("role") == "candidate")
role_label = current_user.get("role", "user").capitalize()
role_badge = (
    '<span style="background: #ECFDF5; color: #047857; padding: 2px 8px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;">Candidate</span>'
    if is_candidate
    else '<span style="background: #EFF6FF; color: #1E40AF; padding: 2px 8px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;">Recruiter</span>'
)
company_html = (
    f'<div style="font-size: 0.78rem; color: #64748B; margin-top: 3px;">🏢 {current_user.get("company_name")}</div>'
    if current_user.get("company_name")
    else ''
)

# Top Application Header with Profile Avatar Dropdown
header_left, header_right = st.columns([5, 1.2], vertical_alignment="center")

with header_left:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 0.8rem;">
            <div style="background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%); color: white; border-radius: 10px; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; font-size: 1.35rem; font-weight: 700; box-shadow: 0 2px 8px rgba(37,99,235,0.25);">🎯</div>
            <div>
                <div style="font-size: 1.2rem; font-weight: 800; color: #1E3A8A; letter-spacing: -0.02em; line-height: 1.2;">SmartHire ATS</div>
                <div style="font-size: 0.8rem; color: #64748B; font-weight: 500;">AI Resume Screening &amp; Ranking Platform</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    avatar_initial = user_email[0].upper() if user_email else "U"
    display_name = user_email.split("@")[0]
    popover_label = f"👤 {display_name} ▾"

    with st.popover(popover_label, use_container_width=True):
        # Dropdown Header: Account Details
        st.markdown(
            f"""
            <div style="padding-bottom: 0.75rem; margin-bottom: 0.75rem; border-bottom: 1px solid #E2E8F0;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.35rem;">
                    <div style="background: #1E3A8A; color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.95rem;">
                        {avatar_initial}
                    </div>
                    <div style="overflow: hidden;">
                        <div style="font-weight: 700; color: #1E293B; font-size: 0.9rem; line-height: 1.2; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">{user_email}</div>
                        <div style="margin-top: 2px;">{role_badge}</div>
                    </div>
                </div>
                {company_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Role-based menu items
        if is_candidate:
            if st.button("👤 My Profile", key="cand_menu_profile", use_container_width=True):
                st.toast(f"Profile: {user_email} (Candidate)")
            if st.button("📄 My Resume", key="cand_menu_resume", use_container_width=True):
                st.toast("Viewing resume screening section")
            if st.button("📊 ATS Analysis History", key="cand_menu_history", use_container_width=True):
                st.toast("ATS analysis history loaded")
            if st.button("💾 Saved Reports", key="cand_menu_reports", use_container_width=True):
                st.toast("Saved reports accessed")
            if st.button("⚙️ Settings", key="cand_menu_settings", use_container_width=True):
                st.toast("Account settings & preferences")
        else:
            if st.button("👤 My Profile", key="rec_menu_profile", use_container_width=True):
                st.toast(f"Profile: {user_email} (Recruiter)")
            if st.button("💼 Dashboard", key="rec_menu_dashboard", use_container_width=True):
                st.toast("Recruiter dashboard active")
            if st.button("🏆 Candidate Rankings", key="rec_menu_rankings", use_container_width=True):
                st.toast("Navigating to candidate rankings")
            if st.button("⭐ Shortlisted Candidates", key="rec_menu_shortlist", use_container_width=True):
                st.toast("Navigating to shortlisted candidates")
            if st.button("⚙️ Settings", key="rec_menu_settings", use_container_width=True):
                st.toast("Organization settings & preferences")

        st.markdown('<div style="margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid #E2E8F0;"></div>', unsafe_allow_html=True)
        if st.button("🚪 Logout", key="dropdown_btn_logout", use_container_width=True, type="secondary"):
            st.session_state["user"] = None
            st.session_state.pop("single_eval_result", None)
            st.session_state.pop("batch_eval_result", None)
            st.rerun()

st.markdown('<hr style="margin: 0.3rem 0 1.2rem 0; border: none; border-top: 1px solid #E2E8F0;" />', unsafe_allow_html=True)

if "shortlist_status" not in st.session_state:
    st.session_state["shortlist_status"] = {}

# ==============================================================================
# CANDIDATE DASHBOARD
# Features:
# - Resume Screening & Scores
# - ATS Score
# - ATS Score Breakdown
# - Job Match Score
# - Missing Keywords
# - Missing Skills
# - Skill Gap Analysis
# - Resume Improvement Suggestions
# - Resume Optimizer
# - Optimized Resume Download
# ==============================================================================
if is_candidate:
    st.markdown(
        """
        <div style='text-align: left; margin-bottom: 1.5rem;'>
            <h1 style='color: #1E3A8A; margin-bottom: 0.2rem;'>🎯 Candidate ATS Dashboard</h1>
            <p style='color: #4B5563; font-size: 1.05rem;'>
                Analyze your resume against job requirements, check your ATS compatibility score, discover missing keywords, and get tailored recommendations to improve your resume.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cand_tab1, cand_tab2 = st.tabs([
        "📄 Resume Screening & Scores",
        "🚀 Resume Optimizer & Suggestions",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: RESUME SCREENING & SCORES
    # --------------------------------------------------------------------------
    with cand_tab1:
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.subheader("Target Job Description")
            c_jd_title = st.text_input(
                "Target Job Position",
                value="",
                placeholder="e.g. Software Engineer, Data Scientist...",
                key="c_jd_title",
            )
            c_jd_text = st.text_area(
                "Job Description Requirements",
                value="",
                placeholder="Paste the target job description requirements here...",
                height=220,
                key="c_jd_text",
            )
            c_exp_override = st.number_input(
                "Required Experience (Years)",
                min_value=0.0,
                max_value=20.0,
                value=0.0,
                step=0.5,
                key="c_exp",
            )

        with col_right:
            st.subheader("Your Resume")
            c_upload_mode = st.radio(
                "Resume Format",
                ["Upload Document (PDF, DOCX, TXT)", "Paste Resume Text"],
                horizontal=True,
                key="c_up_mode",
            )

            c_uploaded_file = None
            c_pasted_text = ""

            if c_upload_mode == "Upload Document (PDF, DOCX, TXT)":
                c_uploaded_file = st.file_uploader(
                    "Upload Resume Document",
                    type=["pdf", "docx", "txt"],
                    help="Accepts PDF, DOCX, or plain text formats.",
                    key="c_file_upload",
                )
            else:
                c_pasted_text = st.text_area(
                    "Paste Resume Content",
                    value="",
                    placeholder="Paste your complete resume text here...",
                    height=220,
                    key="c_paste_area",
                )

            c_screen_btn = st.button(
                "🚀 Analyze Resume & Calculate Score",
                type="primary",
                **ST_FULL_WIDTH,
                key="c_btn_screen",
            )

        if c_screen_btn:
            has_resume = (c_uploaded_file is not None) or bool(c_pasted_text and c_pasted_text.strip())
            has_jd = bool(c_jd_text and c_jd_text.strip())

            if not has_resume or not has_jd:
                st.error("Please upload a resume and enter a job description.")
            else:
                with st.spinner("Analyzing resume against job requirements..."):
                    eval_result = api_client.evaluate_single(
                        resume_file=c_uploaded_file,
                        resume_text=c_pasted_text,
                        jd_text=c_jd_text,
                        jd_title=c_jd_title or "Target Role",
                        experience_required=c_exp_override,
                    )
                    st.session_state["single_eval_result"] = eval_result

        # Display Candidate Results
        if "single_eval_result" in st.session_state:
            res = st.session_state["single_eval_result"]
            cand = res.get("candidate", {})
            scores = res.get("score_breakdown", {})
            skill_gap = res.get("skill_gap", {})
            recs = res.get("recommendations", {})

            st.markdown("---")

            # ATS Score
            st.markdown("### ATS Score & Overall Match")
            m1, m2, m3, m4 = st.columns(4)
            ats_val = scores.get("ats_score", 0)
            tier_badge = "Strong Match" if ats_val >= 75 else ("Good Match" if ats_val >= 60 else "Needs Improvement")

            with m1:
                st.metric("Overall ATS Score", f"{ats_val:.1f} / 100")
            with m2:
                st.metric("Job Match Rating", tier_badge)
            with m3:
                st.metric("Core Skill Match", f"{skill_gap.get('skill_match_percentage', 0):.1f}%")
            with m4:
                st.metric("Relevant Experience", f"{cand.get('experience_years', 0)} years")

            vis_col1, vis_col2 = st.columns([1, 1])
            with vis_col1:
                st.plotly_chart(create_gauge_chart(ats_val), **ST_FULL_WIDTH)
            with vis_col2:
                st.plotly_chart(create_radar_chart(scores), **ST_FULL_WIDTH)

            # Job Match Score
            st.markdown("### Job Match Score Breakdown")
            breakdown_df = pd.DataFrame([
                {"Pillar": "Keyword Matching", "Weight": "40%", "Score": f"{scores.get('keyword_matching', 0):.1f} / 100", "Weighted Contribution": f"{scores.get('keyword_matching', 0) * 0.40:.2f}"},
                {"Pillar": "Context Relevance", "Weight": "30%", "Score": f"{scores.get('semantic_similarity', 0):.1f} / 100", "Weighted Contribution": f"{scores.get('semantic_similarity', 0) * 0.30:.2f}"},
                {"Pillar": "Experience Matching", "Weight": "15%", "Score": f"{scores.get('experience_matching', 0):.1f} / 100", "Weighted Contribution": f"{scores.get('experience_matching', 0) * 0.15:.2f}"},
                {"Pillar": "Education Matching", "Weight": "5%", "Score": f"{scores.get('education_matching', 0):.1f} / 100", "Weighted Contribution": f"{scores.get('education_matching', 0) * 0.05:.2f}"},
                {"Pillar": "Certification Matching", "Weight": "10%", "Score": f"{scores.get('certification_matching', 0):.1f} / 100", "Weighted Contribution": f"{scores.get('certification_matching', 0) * 0.10:.2f}"},
            ])
            st.dataframe(breakdown_df, hide_index=True, **ST_FULL_WIDTH)

            # Missing Keywords
            st.markdown("### Missing Keywords")
            st.caption("ATS scanners look for these exact keywords in your work experience and skills sections.")
            missing_kw = recs.get("missing_keywords", []) or skill_gap.get("missing_skills", [])
            if missing_kw:
                kw_html = "".join([f"<span style='display:inline-block; background:#FEF2F2; color:#991B1B; padding:5px 12px; border-radius:9999px; margin:3px 6px 3px 0; border:1px solid #FECACA; font-weight:600;'>⚠ {kw}</span>" for kw in missing_kw])
                st.markdown(kw_html, unsafe_allow_html=True)
            else:
                st.success("🎉 No critical keywords missing! Your resume contains all target keywords.")

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

            # Skill Gap Analysis
            st.markdown("### Skill Gap Analysis")
            st.caption("Compare your resume skills with job requirements.")
            sg_col1, sg_col2 = st.columns(2)

            with sg_col1:
                st.markdown("#### Matched Skills")
                matched_skills = skill_gap.get("matched_skills", [])
                if matched_skills:
                    for s in matched_skills:
                        st.markdown(f"✓ {s}")
                else:
                    st.write("No matching skills detected.")

            with sg_col2:
                st.markdown("#### Missing Skills")
                missing_skills = skill_gap.get("missing_skills", [])
                if missing_skills:
                    for s in missing_skills:
                        st.markdown(f"⚠ {s}")
                else:
                    st.success("🎉 All required skills detected in resume!")

            skill_match_val = float(skill_gap.get("skill_match_percentage", 0))
            st.markdown(f"**Skill Match: {skill_match_val:.0f}%**")
            st.progress(min(1.0, max(0.0, skill_match_val / 100.0)))

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            if missing_skills:
                rec_pair = " and ".join(missing_skills[:2]) if len(missing_skills) >= 2 else missing_skills[0]
                st.info(f"💡 **Recommendation:** Adding {rec_pair} could improve your job match score.")
            else:
                st.success("💡 **Recommendation:** Great job! Your skills align completely with this job requirement.")

            # Download Report
            st.markdown("---")
            pdf_path = report_generator.generate_pdf_report(res)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            st.download_button(
                label="📥 Download Executive ATS Screening Report (PDF)",
                data=pdf_bytes,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf",
                type="primary",
                **ST_FULL_WIDTH,
            )
        else:
            st.markdown("---")
            st.info("Upload your resume and provide a job description to generate ATS insights.")

    # --------------------------------------------------------------------------
    # TAB 2: RESUME OPTIMIZER & IMPROVEMENT SUGGESTIONS
    # --------------------------------------------------------------------------
    with cand_tab2:
        if "single_eval_result" not in st.session_state:
            st.info("Upload your resume and provide a job description to generate ATS insights.")
        else:
            res = st.session_state["single_eval_result"]
            cand = res.get("candidate", {})
            scores = res.get("score_breakdown", {})
            skill_gap = res.get("skill_gap", {})
            recs = res.get("recommendations", {})
            cand_base_score = int(round(scores.get("ats_score", 0)))
            matched_skills = skill_gap.get("matched_skills", [])
            missing_skills = skill_gap.get("missing_skills", [])
            structural_recs = recs.get("resume_improvement_recommendations", [])

            # Resume Improvement Suggestions
            st.subheader("Resume Improvement Suggestions")
            st.caption("Actionable recommendations to enhance your resume for applicant tracking systems.")

            st.markdown(
                f"""
                <div style="background: white; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.2rem; display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <div style="font-size: 0.85rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Overall ATS Score</div>
                        <div style="font-size: 2.4rem; font-weight: 800; color: #1E3A8A; margin-top: 2px;">{cand_base_score} / 100</div>
                    </div>
                    <div style="text-align: right;">
                        <span style="display: inline-block; background: {'#ECFDF5' if cand_base_score >= 70 else '#FEF3C7'}; color: {'#065F46' if cand_base_score >= 70 else '#92400E'}; font-weight: 700; font-size: 0.95rem; padding: 6px 14px; border-radius: 9999px;">
                            {'Strong Candidate Match' if cand_base_score >= 75 else ('Moderate Match' if cand_base_score >= 60 else 'Needs Improvement')}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(1.0, max(0.0, cand_base_score / 100.0)))

            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

            col_str, col_imp, col_rec = st.columns(3)

            with col_str:
                str_items = []
                if matched_skills:
                    str_items.append(f"✓ Strong match on: {', '.join(matched_skills[:3])}")
                if cand.get("experience_years", 0) > 0:
                    str_items.append(f"✓ {cand.get('experience_years')} years relevant experience")
                if cand.get("highest_education"):
                    str_items.append(f"✓ {cand.get('highest_education')} qualification")
                if not str_items:
                    str_items = ["✓ Resume uploaded and analyzed successfully"]

                str_html = "".join([f"<p style='color: #15803D; font-size: 0.92rem; margin-bottom: 0.45rem;'>{item}</p>" for item in str_items])
                st.markdown(
                    f"""
                    <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 10px; padding: 1.2rem; height: 100%;">
                        <h4 style="color: #166534; margin-top: 0; margin-bottom: 0.8rem;">🌟 Strengths</h4>
                        {str_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_imp:
                imp_items = []
                if missing_skills:
                    imp_items.append(f"⚠ Missing target keywords: {', '.join(missing_skills[:3])}")
                if structural_recs:
                    imp_items.extend([f"⚠ {r}" for r in structural_recs[:2]])
                if not imp_items:
                    imp_items = ["✓ No critical structural deficiencies detected"]

                imp_html = "".join([f"<p style='color: #A16207; font-size: 0.92rem; margin-bottom: 0.45rem;'>{item}</p>" for item in imp_items[:3]])
                st.markdown(
                    f"""
                    <div style="background: #FEFCE8; border: 1px solid #FEF08A; border-radius: 10px; padding: 1.2rem; height: 100%;">
                        <h4 style="color: #854D0E; margin-top: 0; margin-bottom: 0.8rem;">⚠️ Improvements</h4>
                        {imp_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_rec:
                rec_items = []
                if missing_skills:
                    rec_items.append(f"✓ Incorporate {missing_skills[0]} in your work history")
                if recs.get("recommended_skills"):
                    rec_items.append(f"✓ Highlight {recs.get('recommended_skills')[0]} in skills section")
                if recs.get("suggested_certifications"):
                    rec_items.append(f"✓ Consider {recs.get('suggested_certifications')[0]}")
                if not rec_items:
                    rec_items = ["✓ Keep resume updated with recent project metrics"]

                rec_html = "".join([f"<p style='color: #1D4ED8; font-size: 0.92rem; margin-bottom: 0.45rem;'>{item}</p>" for item in rec_items[:3]])
                st.markdown(
                    f"""
                    <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 1.2rem; height: 100%;">
                        <h4 style="color: #1E40AF; margin-top: 0; margin-bottom: 0.8rem;">💡 Recommendations</h4>
                        {rec_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # Resume Optimizer
            st.subheader("Resume Optimizer")
            st.write("Select recommended skills to see your predicted ATS score increase in real time:")

            # Dynamically derive optimizer skills from actual missing skills
            optimizer_skills_list = []
            for s in missing_skills:
                if s not in optimizer_skills_list:
                    optimizer_skills_list.append(s)
            for s in recs.get("recommended_skills", []):
                if s not in optimizer_skills_list:
                    optimizer_skills_list.append(s)

            if not optimizer_skills_list:
                optimizer_skills_list = ["Communication", "Agile", "Problem Solving"]

            rec_skills_data = {}
            for idx, skill_name in enumerate(optimizer_skills_list[:5]):
                impact_pts = max(3, 7 - idx)
                rec_skills_data[skill_name] = impact_pts

            opt_col1, opt_col2 = st.columns([1, 1], gap="large")

            with opt_col1:
                st.markdown("#### Recommended Skills to Add:")
                selected_skills = []
                for skill_name in rec_skills_data.keys():
                    if st.checkbox(skill_name, key=f"cand_chk_opt_{skill_name}"):
                        selected_skills.append(skill_name)

            with opt_col2:
                added_gain = sum(rec_skills_data[s] for s in selected_skills)
                predicted_score = min(100, cand_base_score + added_gain)

                if selected_skills:
                    st.markdown(
                        f"""
                        <div style="background: #F0FDF4; border: 2px solid #86EFAC; border-radius: 12px; padding: 1.2rem; text-align: center; margin-bottom: 1.2rem;">
                            <div style="color: #166534; font-size: 0.95rem; font-weight: 700; text-transform: uppercase;">Predicted ATS Score:</div>
                            <div style="font-size: 2.3rem; font-weight: 800; color: #15803D; margin: 0.4rem 0;">
                                {cand_base_score}% <span style="color: #059669;">→ {predicted_score}%</span>
                            </div>
                            <span style="display: inline-block; background: #DCFCE7; color: #166534; font-weight: 700; font-size: 0.9rem; padding: 3px 12px; border-radius: 9999px;">
                                +{added_gain}% Score Increase
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="background: #F8FAFC; border: 2px solid #E2E8F0; border-radius: 12px; padding: 1.2rem; text-align: center; margin-bottom: 1.2rem;">
                            <div style="color: #64748B; font-size: 0.95rem; font-weight: 700; text-transform: uppercase;">Current ATS Score:</div>
                            <div style="font-size: 2.3rem; font-weight: 800; color: #1E293B; margin: 0.4rem 0;">{cand_base_score}%</div>
                            <p style="color: #64748B; font-size: 0.88rem; margin: 0;">
                                Select recommended skills on the left to see your predicted score increase.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("#### Skill Impact")
                for skill_name, impact in rec_skills_data.items():
                    is_active = skill_name in selected_skills
                    bg = "#ECFDF5" if is_active else "#F8FAFC"
                    border = "#6EE7B7" if is_active else "#E2E8F0"
                    text_color = "#065F46" if is_active else "#1E293B"

                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.55rem 0.9rem; background: {bg}; border: 1px solid {border}; border-radius: 8px; margin-bottom: 0.45rem;">
                            <span style="font-weight: 600; color: {text_color};">{skill_name}:</span>
                            <span style="font-weight: 700; color: #059669; font-size: 0.95rem;">+{impact}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Optimized Resume Download
            st.markdown("---")
            st.subheader("Optimized Resume Download")
            st.caption("Download your improved resume updated with targeted skills and keywords.")

            cand_resume_body = cand.get("raw_text", "")
            if not cand_resume_body:
                cand_resume_body = c_pasted_text or ""

            active_skills = selected_skills if selected_skills else list(rec_skills_data.keys())[:2]
            target_score_num = predicted_score if selected_skills else cand_base_score + 10
            optimized_resume_content = (
                cand_resume_body.strip()
                + "\n\n"
                + "=" * 50
                + "\nATS OPTIMIZATION ADDITIONS:\n"
                + "Added High-Impact Skills: " + ", ".join(active_skills) + "\n"
                + "Target Position: " + (c_jd_title or "Target Role") + "\n"
                + f"Target ATS Score: {target_score_num}%\n"
                + "=" * 50
                + "\n"
            )

            st.download_button(
                label="📥 Download Optimized Resume (.txt)",
                data=optimized_resume_content,
                file_name="optimized_resume.txt",
                mime="text/plain",
                type="primary",
                key="cand_dl_opt_resume",
                **ST_FULL_WIDTH,
            )

# ==============================================================================
# RECRUITER DASHBOARD
# Features:
# - Job Description Upload
# - Resume Upload
# - Candidate Screening
# - Candidate Ranking
# - Candidate Comparison
# - Shortlisted Candidates
# ==============================================================================
else:
    st.markdown(
        """
        <div style='text-align: left; margin-bottom: 1.5rem;'>
            <h1 style='color: #1E3A8A; margin-bottom: 0.2rem;'>💼 Recruiter ATS Talent Portal</h1>
            <p style='color: #4B5563; font-size: 1.05rem;'>
                Screen candidate resumes in batch, rank applicants on a unified leaderboard, compare top candidates side-by-side, and manage your interview shortlist.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rec_tab1, rec_tab2, rec_tab3 = st.tabs([
        "🏆 Candidate Screening & Ranking",
        "👥 Candidate Comparison",
        "⭐ Shortlisted Candidates",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: SCREENING & RANKING
    # --------------------------------------------------------------------------
    with rec_tab1:
        st.subheader("Upload Resumes & Job Description")

        b_col1, b_col2 = st.columns([1, 1], gap="large")

        with b_col1:
            batch_files = st.file_uploader(
                "Upload Multiple Resumes (PDF, DOCX, TXT)",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                help="Select multiple resumes from your computer.",
                key="rec_batch_files",
            )
            batch_jd_title = st.text_input(
                "Target Job Position",
                value="",
                placeholder="e.g. Senior Software Engineer",
                key="rec_batch_title",
            )

        with b_col2:
            batch_jd_text = st.text_area(
                "Job Description Requirements",
                value="",
                placeholder="Paste job requirements and responsibilities here...",
                height=160,
                key="rec_batch_jd",
            )
            batch_exp_override = st.number_input(
                "Required Experience (Years)",
                min_value=0.0,
                max_value=20.0,
                value=0.0,
                step=0.5,
                key="rec_batch_exp",
            )

        run_batch_btn = st.button("🚀 Screen & Rank Resumes", type="primary", **ST_FULL_WIDTH, key="rec_btn_screen")

        if run_batch_btn:
            if not batch_jd_title.strip() or not batch_jd_text.strip():
                st.error("Please enter the job position and job description before screening candidates.")
            elif not batch_files:
                st.error("Please upload at least one resume file.")
            else:
                with st.spinner(f"Evaluating and ranking {len(batch_files)} resumes..."):
                    batch_res = api_client.evaluate_batch(
                        resume_files=batch_files,
                        jd_text=batch_jd_text,
                        jd_title=batch_jd_title,
                        experience_required=batch_exp_override,
                    )
                    st.session_state["batch_eval_result"] = batch_res

        # Candidate Ranking & Shortlisting
        if "batch_eval_result" in st.session_state:
            batch_data = st.session_state["batch_eval_result"]
            rankings = batch_data.get("rankings", [])

            # Initialize shortlisting state
            for cand in rankings:
                c_name = cand.get("candidate", {}).get("name", "Unknown")
                if c_name not in st.session_state["shortlist_status"]:
                    st.session_state["shortlist_status"][c_name] = "Under Review"

            st.markdown("---")
            st.header(f"🏅 Candidate Ranking Leaderboard ({len(rankings)} Candidates)")

            # Leaderboard Table
            table_rows = []
            for cand in rankings:
                c_info = cand.get("candidate", {})
                c_scores = cand.get("score_breakdown", {})
                c_gap = cand.get("skill_gap", {})
                c_name = c_info.get("name", "Unknown")
                status = st.session_state["shortlist_status"].get(c_name, "Under Review")

                table_rows.append({
                    "Rank": f"#{cand.get('candidate_rank')}",
                    "Candidate Name": c_name,
                    "ATS Score": f"{c_scores.get('ats_score', 0):.1f}",
                    "Match %": f"{c_scores.get('match_percentage', 0):.1f}%",
                    "Experience": f"{c_info.get('experience_years', 0)} yrs",
                    "Matched Skills": ", ".join(c_gap.get("matched_skills", [])[:4]),
                    "Missing Skills": ", ".join(c_gap.get("missing_skills", [])[:3]),
                    "Status": status,
                })

            df_leaderboard = pd.DataFrame(table_rows)
            st.dataframe(df_leaderboard, hide_index=True, **ST_FULL_WIDTH)

            # Bar chart comparison
            st.plotly_chart(create_ranking_bar_chart(rankings), **ST_FULL_WIDTH)

            # Export Excel Report
            excel_path = report_generator.generate_excel_batch_report(rankings, job_title=batch_jd_title)
            with open(excel_path, "rb") as f:
                excel_bytes = f.read()

            st.download_button(
                label="📥 Download Complete Candidate Ranking Excel Report (.xlsx)",
                data=excel_bytes,
                file_name=os.path.basename(excel_path),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                **ST_FULL_WIDTH,
            )

            st.markdown("---")
            # Shortlisting Quick Actions
            st.subheader("Update Candidate Shortlisting Status")
            s_cols = st.columns(min(len(rankings), 3))
            for idx, cand in enumerate(rankings):
                col_idx = idx % min(len(rankings), 3)
                c_name = cand.get("candidate", {}).get("name", "Unknown")
                cur_status = st.session_state["shortlist_status"].get(c_name, "Under Review")
                options = ["⭐ Shortlisted", "⏳ Under Review", "❌ Rejected"]
                cur_idx = 0 if "Shortlisted" in cur_status else (1 if "Review" in cur_status else 2)

                with s_cols[col_idx]:
                    new_st = st.selectbox(
                        f"Status: {c_name}",
                        options,
                        index=cur_idx,
                        key=f"sel_status_{c_name}_{idx}",
                    )
                    st.session_state["shortlist_status"][c_name] = new_st
        else:
            st.markdown("---")
            st.info("Enter target job requirements and upload candidate resumes above to screen and rank applicants.")

    # --------------------------------------------------------------------------
    # TAB 2: CANDIDATE COMPARISON
    # --------------------------------------------------------------------------
    with rec_tab2:
        st.subheader("Candidate Comparison")
        st.caption("Compare candidates side-by-side across overall ATS scores, skills, experience, and education.")

        if "batch_eval_result" in st.session_state:
            rankings = st.session_state["batch_eval_result"].get("rankings", [])
            cand_names = [c.get("candidate", {}).get("name", f"Candidate {i+1}") for i, c in enumerate(rankings)]

            selected_compare = st.multiselect(
                "Select Candidates to Compare Side-by-Side:",
                cand_names,
                default=cand_names[:min(len(cand_names), 3)],
                key="rec_compare_select",
            )

            if len(selected_compare) >= 2:
                comp_cols = st.columns(len(selected_compare))
                chosen_items = [c for c in rankings if c.get("candidate", {}).get("name") in selected_compare]

                for idx, c_item in enumerate(chosen_items):
                    c_info = c_item.get("candidate", {})
                    c_scores = c_item.get("score_breakdown", {})
                    c_gap = c_item.get("skill_gap", {})
                    name = c_info.get("name", "Unknown")

                    with comp_cols[idx]:
                        st.markdown(
                            f"""
                            <div style="background: white; border: 2px solid #E2E8F0; border-radius: 12px; padding: 1.2rem; height: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                <h3 style="color: #1E3A8A; margin-top: 0; margin-bottom: 0.3rem;">{name}</h3>
                                <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 0.8rem;">{c_info.get('email', 'No email')}</div>
                                <div style="font-size: 1.8rem; font-weight: 800; color: #10B981; margin-bottom: 0.4rem;">
                                    {c_scores.get('ats_score', 0):.1f} <span style="font-size: 0.9rem; color: #64748B;">/ 100</span>
                                </div>
                                <p style="margin: 0.3rem 0; font-size: 0.9rem;"><strong>Match:</strong> {c_scores.get('match_percentage', 0):.1f}%</p>
                                <p style="margin: 0.3rem 0; font-size: 0.9rem;"><strong>Experience:</strong> {c_info.get('experience_years', 0)} years</p>
                                <p style="margin: 0.3rem 0; font-size: 0.9rem;"><strong>Education:</strong> {c_info.get('highest_education', 'Degree')}</p>
                                <hr style="margin: 0.8rem 0;"/>
                                <p style="font-size: 0.85rem; font-weight: 700; color: #065F46; margin-bottom: 0.3rem;">Matched Skills:</p>
                                <p style="font-size: 0.85rem; color: #334155; margin-bottom: 0.6rem;">{', '.join(c_gap.get('matched_skills', [])[:4]) or 'None'}</p>
                                <p style="font-size: 0.85rem; font-weight: 700; color: #991B1B; margin-bottom: 0.3rem;">Missing Skills:</p>
                                <p style="font-size: 0.85rem; color: #64748B; margin-bottom: 0;">{', '.join(c_gap.get('missing_skills', [])[:3]) or 'None'}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            else:
                st.info("Please select at least 2 candidates above to compare side-by-side.")
        else:
            st.info("Upload and screen resumes in the 'Candidate Screening & Ranking' tab first to compare candidates.")

    # --------------------------------------------------------------------------
    # TAB 3: SHORTLISTED CANDIDATES
    # --------------------------------------------------------------------------
    with rec_tab3:
        st.subheader("Shortlisting & Hiring Pipeline")
        st.caption("Manage candidates moved forward for interview scheduling.")

        if "batch_eval_result" in st.session_state:
            rankings = st.session_state["batch_eval_result"].get("rankings", [])
            total_cand = len(rankings)
            num_short = sum(1 for c in rankings if "Shortlisted" in st.session_state["shortlist_status"].get(c.get("candidate", {}).get("name", ""), ""))
            num_rev = sum(1 for c in rankings if "Review" in st.session_state["shortlist_status"].get(c.get("candidate", {}).get("name", ""), ""))
            num_rej = sum(1 for c in rankings if "Rejected" in st.session_state["shortlist_status"].get(c.get("candidate", {}).get("name", ""), ""))

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.metric("Total Screened", total_cand)
            with kpi2:
                st.metric("⭐ Shortlisted", num_short)
            with kpi3:
                st.metric("⏳ Under Review", num_rev)
            with kpi4:
                st.metric("❌ Rejected", num_rej)

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            filter_mode = st.radio(
                "Filter Pipeline Candidates:",
                ["All Candidates", "⭐ Shortlisted Only", "⏳ Under Review Only", "❌ Rejected Only"],
                horizontal=True,
                key="rec_pipeline_filter",
            )

            filtered_rows = []
            for c in rankings:
                c_info = c.get("candidate", {})
                c_scores = c.get("score_breakdown", {})
                c_name = c_info.get("name", "Unknown")
                c_status = st.session_state["shortlist_status"].get(c_name, "Under Review")

                if filter_mode == "⭐ Shortlisted Only" and "Shortlisted" not in c_status:
                    continue
                if filter_mode == "⏳ Under Review Only" and "Review" not in c_status:
                    continue
                if filter_mode == "❌ Rejected Only" and "Rejected" not in c_status:
                    continue

                filtered_rows.append({
                    "Rank": f"#{c.get('candidate_rank')}",
                    "Candidate Name": c_name,
                    "Email": c_info.get("email", "N/A"),
                    "ATS Score": f"{c_scores.get('ats_score', 0):.1f}",
                    "Match %": f"{c_scores.get('match_percentage', 0):.1f}%",
                    "Experience": f"{c_info.get('experience_years', 0)} yrs",
                    "Status": c_status,
                })

            if filtered_rows:
                st.dataframe(pd.DataFrame(filtered_rows), hide_index=True, **ST_FULL_WIDTH)
            else:
                st.info(f"No candidates currently marked as '{filter_mode}'.")
        else:
            st.info("Upload and screen resumes in the 'Candidate Screening & Ranking' tab to build your candidate pipeline.")

