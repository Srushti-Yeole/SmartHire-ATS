"""
Report Generator Module for AI Resume Screening & ATS Score Predictor.
Generates comprehensive PDF reports via ReportLab and multi-sheet Excel reports via openpyxl.
"""

import io
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd

# Check ReportLab availability
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class ReportGenerator:
    """Production report generation engine producing downloadable PDF and Excel artifacts."""

    REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))

    def __init__(self):
        os.makedirs(self.REPORTS_DIR, exist_ok=True)

    def generate_pdf_report(
        self,
        evaluation_data: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generates an executive ATS Evaluation PDF Report with scores, breakdown,
        skill gaps, and recommendations.
        """
        if not output_path:
            cand_name = evaluation_data.get("candidate", {}).get("name", "Candidate").replace(" ", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ATS_Report_{cand_name}_{timestamp}.pdf"
            output_path = os.path.join(self.REPORTS_DIR, filename)

        if not REPORTLAB_AVAILABLE:
            # Fallback text summary file if ReportLab is not present
            txt_path = output_path.replace(".pdf", ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"ATS Evaluation Report\n")
                f.write(f"Candidate: {evaluation_data.get('candidate', {}).get('name')}\n")
                f.write(f"ATS Score: {evaluation_data.get('score_breakdown', {}).get('ats_score')}/100\n")
            return txt_path

        # Build PDF with ReportLab
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#1E3A8A"),
            spaceAfter=6,
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontSize=11,
            textColor=colors.HexColor("#4B5563"),
            spaceAfter=12,
        )
        section_style = ParagraphStyle(
            "SectionHeader",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#1F2937"),
            spaceBefore=12,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "BodyTextCustom",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#374151"),
        )
        bullet_style = ParagraphStyle(
            "BulletCustom",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#374151"),
            leftIndent=12,
            spaceAfter=4,
        )

        elements = []

        cand = evaluation_data.get("candidate", {})
        jd = evaluation_data.get("job_description", {})
        scores = evaluation_data.get("score_breakdown", {})
        skill_gap = evaluation_data.get("skill_gap", {})
        recs = evaluation_data.get("recommendations", {})

        ats_score = scores.get("ats_score", 0.0)

        # Header Title
        elements.append(Paragraph("AI Resume Screening & ATS Score Report", title_style))
        elements.append(
            Paragraph(
                f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')} | Powered by Antigravity NLP",
                subtitle_style,
            )
        )
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#CBD5E1"), spaceAfter=14))

        # Candidate & Job Info Table
        info_data = [
            [
                Paragraph(f"<b>Candidate:</b> {cand.get('name', 'N/A')}", body_style),
                Paragraph(f"<b>Target Role:</b> {jd.get('title', 'N/A')}", body_style),
            ],
            [
                Paragraph(f"<b>Email:</b> {cand.get('email') or 'Not specified'}", body_style),
                Paragraph(f"<b>Required Experience:</b> {jd.get('experience_required', 0)} years", body_style),
            ],
            [
                Paragraph(f"<b>Detected Experience:</b> {cand.get('experience_years', 0)} years", body_style),
                Paragraph(f"<b>Required Education:</b> {jd.get('min_education', 'N/A')}", body_style),
            ],
            [
                Paragraph(f"<b>Highest Education:</b> {cand.get('highest_education', 'N/A')}", body_style),
                Paragraph(f"<b>Status:</b> {'Screening Complete'}", body_style),
            ],
        ]
        info_table = Table(info_data, colWidths=[270, 270])
        info_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ])
        )
        elements.append(info_table)
        elements.append(Spacer(1, 14))

        # ATS Score Badge Table
        score_color = "#10B981" if ats_score >= 75 else ("#F59E0B" if ats_score >= 50 else "#EF4444")
        score_data = [
            [
                Paragraph(
                    f"<font size='28' color='{score_color}'><b>{ats_score:.1f} / 100</b></font><br/>"
                    f"<font size='11' color='#4B5563'>Composite ATS Score (Match: {scores.get('match_percentage', ats_score):.1f}%)</font>",
                    body_style,
                )
            ]
        ]
        score_table = Table(score_data, colWidths=[540])
        score_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor(score_color)),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ])
        )
        elements.append(score_table)
        elements.append(Spacer(1, 14))

        # 5-Pillar Score Breakdown
        elements.append(Paragraph("Score Breakdown by Evaluation Pillar", section_style))
        breakdown_data = [
            ["Pillar", "Weight", "Score (out of 100)", "Weighted Contribution"],
            ["Keyword Matching", "40%", f"{scores.get('keyword_matching', 0):.1f}", f"{(scores.get('keyword_matching', 0) * 0.40):.2f}"],
            ["Semantic Similarity", "30%", f"{scores.get('semantic_similarity', 0):.1f}", f"{(scores.get('semantic_similarity', 0) * 0.30):.2f}"],
            ["Experience Matching", "15%", f"{scores.get('experience_matching', 0):.1f}", f"{(scores.get('experience_matching', 0) * 0.15):.2f}"],
            ["Education Matching", "5%", f"{scores.get('education_matching', 0):.1f}", f"{(scores.get('education_matching', 0) * 0.05):.2f}"],
            ["Certification Matching", "10%", f"{scores.get('certification_matching', 0):.1f}", f"{(scores.get('certification_matching', 0) * 0.10):.2f}"],
            ["Total Composite Score", "100%", f"{ats_score:.1f}", f"{ats_score:.2f}"],
        ]
        breakdown_table = Table(breakdown_data, colWidths=[160, 80, 150, 150])
        breakdown_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#E2E8F0")),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 5),
            ])
        )
        elements.append(breakdown_table)
        elements.append(Spacer(1, 14))

        # Skill Gap Analysis
        elements.append(Paragraph("Skill Gap Analysis", section_style))
        matched = skill_gap.get("matched_skills", [])
        missing = skill_gap.get("missing_skills", [])
        recommended = recs.get("recommended_skills", [])

        skills_content = [
            [
                Paragraph("<b>Matched Skills:</b>", body_style),
                Paragraph(", ".join(matched) if matched else "None explicitly detected", body_style),
            ],
            [
                Paragraph("<b>Missing Skills:</b>", body_style),
                Paragraph(f"<font color='#DC2626'>{', '.join(missing) if missing else 'None! Full overlap.'}</font>", body_style),
            ],
            [
                Paragraph("<b>Recommended Skills:</b>", body_style),
                Paragraph(", ".join(recommended) if recommended else "N/A", body_style),
            ],
        ]
        skills_table = Table(skills_content, colWidths=[140, 400])
        skills_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ])
        )
        elements.append(skills_table)
        elements.append(Spacer(1, 14))

        # Recommendations Section
        elements.append(Paragraph("Actionable Recommendations & Resume Enhancements", section_style))

        # Suggested Certifications
        cert_list = recs.get("suggested_certifications", [])
        if cert_list:
            elements.append(Paragraph("<b>Suggested Certifications to Bridge Skill Gaps:</b>", body_style))
            for c in cert_list:
                elements.append(Paragraph(f"• {c}", bullet_style))
            elements.append(Spacer(1, 6))

        # Suggested Projects
        proj_list = recs.get("suggested_projects", [])
        if proj_list:
            elements.append(Paragraph("<b>Recommended Portfolio Projects:</b>", body_style))
            for p in proj_list:
                elements.append(Paragraph(f"• {p}", bullet_style))
            elements.append(Spacer(1, 6))

        # Formatting Tips
        tips_list = recs.get("resume_improvement_recommendations", [])
        if tips_list:
            elements.append(Paragraph("<b>Resume Optimization Tips:</b>", body_style))
            for t in tips_list:
                elements.append(Paragraph(f"• {t}", bullet_style))

        doc.build(elements)
        return output_path

    def generate_excel_batch_report(
        self,
        rankings: List[Dict[str, Any]],
        job_title: str = "Position",
        output_path: Optional[str] = None
    ) -> str:
        """
        Generates a comprehensive multi-tab Excel report for candidate ranking and skill gaps.
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ATS_Leaderboard_{timestamp}.xlsx"
            output_path = os.path.join(self.REPORTS_DIR, filename)

        # Tab 1: Rankings Summary
        summary_rows = []
        gap_rows = []
        rec_rows = []

        for item in rankings:
            cand = item.get("candidate", {})
            scores = item.get("score_breakdown", {})
            skill_gap = item.get("skill_gap", {})
            recs = item.get("recommendations", {})

            cand_name = cand.get("name", "Unknown")
            rank = item.get("candidate_rank", item.get("rank", 0))

            summary_rows.append({
                "Rank": rank,
                "Candidate Name": cand_name,
                "ATS Score": scores.get("ats_score", 0.0),
                "Match %": scores.get("match_percentage", 0.0),
                "Keyword Score (40%)": scores.get("keyword_matching", 0.0),
                "Semantic Similarity (30%)": scores.get("semantic_similarity", 0.0),
                "Experience Score (15%)": scores.get("experience_matching", 0.0),
                "Education Score (5%)": scores.get("education_matching", 0.0),
                "Certification Score (10%)": scores.get("certification_matching", 0.0),
                "Experience Years": cand.get("experience_years", 0.0),
                "Highest Education": cand.get("highest_education", ""),
                "Email": cand.get("email", ""),
                "Phone": cand.get("phone", ""),
            })

            gap_rows.append({
                "Rank": rank,
                "Candidate Name": cand_name,
                "Skill Overlap %": skill_gap.get("skill_match_percentage", 0.0),
                "Matched Skills Count": len(skill_gap.get("matched_skills", [])),
                "Matched Skills": ", ".join(skill_gap.get("matched_skills", [])),
                "Missing Skills Count": len(skill_gap.get("missing_skills", [])),
                "Missing Skills": ", ".join(skill_gap.get("missing_skills", [])),
            })

            rec_rows.append({
                "Rank": rank,
                "Candidate Name": cand_name,
                "Suggested Certifications": " | ".join(recs.get("suggested_certifications", [])),
                "Suggested Projects": " | ".join(recs.get("suggested_projects", [])),
                "Optimization Recommendations": " | ".join(recs.get("resume_improvement_recommendations", [])),
            })

        df_summary = pd.DataFrame(summary_rows)
        df_gap = pd.DataFrame(gap_rows)
        df_recs = pd.DataFrame(rec_rows)

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_summary.to_excel(writer, sheet_name="Candidate Rankings", index=False)
            df_gap.to_excel(writer, sheet_name="Skill Gap Details", index=False)
            df_recs.to_excel(writer, sheet_name="Recommendations", index=False)

        return output_path


# Global report generator instance
report_generator = ReportGenerator()
