"""Streamlit dashboard for the financial crime detection system."""

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.ingestion import load_or_create_data
from src.validation import run_validation_on_dataframes
from src.detection import run_detection
from src.triage_agent import run_triage, run_narrative_generation
from src.config import OUTPUT_DIR, TRANSACTIONS_FILE, KYC_FILE


st.set_page_config(page_title="AI Financial Crime Detection", page_icon="💸", layout="wide")


def save_uploaded_file(uploaded_file, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


@st.cache_data(show_spinner=False)
def load_outputs():
    entity_risk, alerts, triage = None, None, None
    if (OUTPUT_DIR / "entity_risk_scores.csv").exists():
        entity_risk = pd.read_csv(OUTPUT_DIR / "entity_risk_scores.csv")
    if (OUTPUT_DIR / "high_risk_alerts.csv").exists():
        alerts = pd.read_csv(OUTPUT_DIR / "high_risk_alerts.csv")
    if (OUTPUT_DIR / "triage_results.csv").exists():
        triage = pd.read_csv(OUTPUT_DIR / "triage_results.csv")
    return entity_risk, alerts, triage


def run_full_pipeline():
    transactions, kyc = load_or_create_data()
    validation = run_validation_on_dataframes(transactions, kyc)
    run_detection()
    run_triage()
    run_narrative_generation()
    load_outputs.clear()
    entity_risk, alerts, triage = load_outputs()
    return validation, entity_risk, alerts, triage


def sidebar_controls():
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.write("Upload real-world CSVs or use existing data on disk.")
    uploaded_txn = st.sidebar.file_uploader("Transactions CSV", type="csv")
    uploaded_kyc = st.sidebar.file_uploader("KYC Profiles CSV", type="csv")
    st.sidebar.markdown("---")
    run_button = st.sidebar.button("🚀 Run Detection Pipeline", use_container_width=True)
    return uploaded_txn, uploaded_kyc, run_button


def render_validation_status(validation):
    if validation.passed:
        st.success("MRA validation PASSED – data is audit-ready.")
    else:
        st.warning("MRA validation FAILED – see detailed checks in logs / console.")


def render_overview_tab(entity_risk: Optional[pd.DataFrame], alerts: Optional[pd.DataFrame]):
    st.subheader("System Overview")
    if entity_risk is None:
        st.info("Run the pipeline to see metrics.")
        return

    total = len(entity_risk)
    high_risk = int(entity_risk.get("high_risk", pd.Series(dtype=bool)).sum())
    low_risk = total - high_risk
    avg_risk = float(entity_risk.get("R_e", pd.Series(dtype=float)).mean())
    max_risk = float(entity_risk.get("R_e", pd.Series(dtype=float)).max())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Entities", f"{total:,}")
    col2.metric("High-Risk", f"{high_risk:,}", delta=f"{high_risk/total*100:.1f}%")
    col3.metric("Avg Risk Score", f"{avg_risk:.3f}")
    col4.metric("Max Risk Score", f"{max_risk:.3f}")

    st.markdown("---")
    col_pie1, col_pie2 = st.columns(2)

    with col_pie1:
        st.markdown("#### Risk Breakdown")
        fig = px.pie(pd.DataFrame({"Category": ["High Risk", "Low Risk"], "Count": [high_risk, low_risk]}),
                     values="Count", names="Category", color="Category",
                     color_discrete_map={"High Risk": "#e74c3c", "Low Risk": "#2ecc71"}, hole=0.4)
        fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_pie2:
        st.markdown("#### Segment Distribution")
        if "segment" in entity_risk.columns:
            seg = entity_risk["segment"].value_counts().reset_index()
            seg.columns = ["Segment", "Count"]
            fig = px.pie(seg, values="Count", names="Segment", hole=0.4)
            fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col_hist, col_bar = st.columns(2)

    with col_hist:
        st.markdown("#### Risk Score Distribution")
        if "R_e" in entity_risk.columns:
            fig = px.histogram(entity_risk, x="R_e", nbins=30)
            fig.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="Threshold")
            fig.update_layout(height=350, margin=dict(t=20, b=40, l=40, r=20))
            st.plotly_chart(fig, use_container_width=True)

    with col_bar:
        st.markdown("#### Avg Risk by Segment")
        if "segment" in entity_risk.columns and "R_e" in entity_risk.columns:
            seg_risk = entity_risk.groupby("segment")["R_e"].agg(["mean", "count"]).reset_index()
            seg_risk.columns = ["Segment", "Avg Risk", "Count"]
            fig = px.bar(seg_risk.sort_values("Avg Risk", ascending=False), x="Segment", y="Avg Risk",
                         color="Avg Risk", color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"], text="Count")
            fig.update_traces(texttemplate="n=%{text}", textposition="outside")
            fig.update_layout(height=350, coloraxis_showscale=False, margin=dict(t=20, b=40, l=40, r=20))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col_scatter, col_box = st.columns(2)

    with col_scatter:
        st.markdown("#### Fan-Out vs Risk")
        if "fan_out" in entity_risk.columns:
            fig = px.scatter(entity_risk, x="fan_out", y="R_e", color="high_risk",
                             color_discrete_map={True: "#e74c3c", False: "#3498db"}, hover_data=["customer_id"])
            fig.update_layout(height=350, margin=dict(t=20, b=40, l=40, r=20))
            st.plotly_chart(fig, use_container_width=True)

    with col_box:
        st.markdown("#### Risk by Segment (Box)")
        if "segment" in entity_risk.columns:
            fig = px.box(entity_risk, x="segment", y="R_e", color="segment")
            fig.update_layout(showlegend=False, height=350, margin=dict(t=20, b=40, l=40, r=20))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Top 50 Entity Risk Scores")
    cols = [c for c in ["customer_id", "segment", "R_e", "high_risk", "txn_count", "total_amount", "fan_out"] if c in entity_risk.columns]
    st.dataframe(entity_risk[cols].sort_values("R_e", ascending=False).head(50), use_container_width=True)


def render_entities_tab(entity_risk: Optional[pd.DataFrame]):
    st.subheader("High-Risk Entity Explorer")
    if entity_risk is None or "high_risk" not in entity_risk.columns:
        st.info("Run the pipeline first.")
        return

    hr = entity_risk[entity_risk["high_risk"]].sort_values("R_e", ascending=False)
    if hr.empty:
        st.success("No high-risk entities detected.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("High-Risk Entities", f"{len(hr):,}")
    col2.metric("Avg Risk", f"{hr['R_e'].mean():.3f}")
    col3.metric("Max Risk", f"{hr['R_e'].max():.3f}")

    st.markdown("---")
    col_rank, col_seg = st.columns(2)

    with col_rank:
        st.markdown("#### Top Entities by Risk")
        top = hr.head(15)
        fig = px.bar(top, x="R_e", y="customer_id", orientation="h", color="R_e",
                     color_continuous_scale=["#f39c12", "#e74c3c"])
        fig.update_layout(yaxis=dict(categoryorder="total ascending"), coloraxis_showscale=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_seg:
        st.markdown("#### High-Risk by Segment")
        if "segment" in hr.columns:
            seg = hr["segment"].value_counts().reset_index()
            seg.columns = ["Segment", "Count"]
            fig = px.bar(seg, x="Segment", y="Count", color="Segment", text="Count")
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Inspect Entity")
    selected = st.selectbox("Select customer", hr["customer_id"].astype(str).tolist())
    row = hr[hr["customer_id"].astype(str) == selected].iloc[0]

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Risk Score", f"{row.get('R_e', 0):.3f}")
    col_b.metric("Txn Count", f"{int(row.get('txn_count', 0)):,}")
    col_c.metric("Total Amount", f"${row.get('total_amount', 0):,.2f}")
    col_d.metric("Fan-Out", f"{int(row.get('fan_out', 0)):,}")

    # radar chart
    num_cols = ["txn_count", "total_amount", "avg_amount", "velocity_max", "fan_out", "amount_deviation"]
    avail = [c for c in num_cols if c in row.index and pd.notna(row[c])]
    if len(avail) >= 3:
        vals = [row[c] / entity_risk[c].max() if entity_risk[c].max() > 0 else 0 for c in avail]
        vals.append(vals[0])
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=vals, theta=avail + [avail[0]], fill="toself",
                                       fillcolor="rgba(231, 76, 60, 0.3)", line_color="#e74c3c"))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)


def render_analytics_tab(entity_risk: Optional[pd.DataFrame]):
    st.subheader("Advanced Analytics")
    if entity_risk is None:
        st.info("Run the pipeline first.")
        return

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        risk_range = st.slider("Risk Score Range", 0.0, 1.0, (0.0, 1.0), 0.05)
    with col_f2:
        segments = ["All"] + (entity_risk["segment"].dropna().unique().tolist() if "segment" in entity_risk.columns else [])
        seg_filter = st.selectbox("Segment", segments)
    with col_f3:
        risk_filter = st.selectbox("Risk Level", ["All", "High Risk Only", "Low Risk Only"])

    df = entity_risk[(entity_risk["R_e"] >= risk_range[0]) & (entity_risk["R_e"] <= risk_range[1])].copy()
    if seg_filter != "All" and "segment" in df.columns:
        df = df[df["segment"] == seg_filter]
    if risk_filter == "High Risk Only":
        df = df[df["high_risk"]]
    elif risk_filter == "Low Risk Only":
        df = df[~df["high_risk"]]

    st.markdown(f"**Showing {len(df):,} entities**")
    st.markdown("---")

    num_cols = ["R_e", "txn_count", "total_amount", "avg_amount", "velocity_max", "fan_out", "amount_deviation"]
    avail = [c for c in num_cols if c in df.columns]

    col_corr, col_dist = st.columns(2)
    with col_corr:
        st.markdown("#### Correlation Heatmap")
        if len(avail) >= 2:
            fig = px.imshow(df[avail].corr(), text_auto=".2f", color_continuous_scale="RdBu_r")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col_dist:
        st.markdown("#### Feature Distribution")
        if avail:
            feat = st.selectbox("Feature", avail)
            fig = px.histogram(df, x=feat, nbins=30, color="high_risk" if "high_risk" in df.columns else None,
                               color_discrete_map={True: "#e74c3c", False: "#3498db"}, barmode="overlay", opacity=0.7)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("#### Txn Volume vs Risk")
        if "txn_count" in df.columns:
            fig = px.scatter(df, x="txn_count", y="R_e", color="high_risk",
                             color_discrete_map={True: "#e74c3c", False: "#3498db"},
                             trendline="ols" if len(df) > 10 else None)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col_s2:
        st.markdown("#### Total Amount vs Risk")
        if "total_amount" in df.columns:
            fig = px.scatter(df, x="total_amount", y="R_e", color="high_risk",
                             color_discrete_map={True: "#e74c3c", False: "#3498db"},
                             trendline="ols" if len(df) > 10 else None)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    if avail:
        st.markdown("#### Summary Statistics")
        st.dataframe(df[avail].describe().T.round(3), use_container_width=True)

    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("📥 Download All Scores", entity_risk.to_csv(index=False), "entity_risk_scores.csv", "text/csv")
    with col_dl2:
        st.download_button("📥 Download High-Risk", entity_risk[entity_risk["high_risk"]].to_csv(index=False), "high_risk.csv", "text/csv")


def render_sar_tab():
    st.subheader("SAR Draft Narratives")
    sar_path, triage_path = OUTPUT_DIR / "sar_drafts.csv", OUTPUT_DIR / "triage_results.csv"
    if not sar_path.exists() or not triage_path.exists():
        st.info("Run the pipeline to generate SAR drafts.")
        return

    sar = pd.read_csv(sar_path)
    triage = pd.read_csv(triage_path)
    merged = sar.merge(triage, on="customer_id", how="left")

    total = len(triage)
    review = int(triage.get("requires_review", pd.Series(dtype=bool)).sum())
    benign = total - review

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Triaged", f"{total:,}")
    col2.metric("Require Review", f"{review:,}")
    col3.metric("Benign", f"{benign:,}")

    st.markdown("---")
    col_pie, col_table = st.columns([1, 2])

    with col_pie:
        st.markdown("#### Triage Decisions")
        if "triage_decision" in triage.columns:
            dec = triage["triage_decision"].value_counts().reset_index()
            dec.columns = ["Decision", "Count"]
            fig = px.pie(dec, values="Count", names="Decision", color="Decision",
                         color_discrete_map={"REVIEW": "#e74c3c", "BENIGN": "#2ecc71"}, hole=0.4)
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("#### Triage Results")
        cols = [c for c in ["customer_id", "R_e", "triage_decision", "triage_reason"] if c in merged.columns]
        st.dataframe(merged[cols], use_container_width=True, height=250)

    st.markdown("---")
    st.markdown("#### View SAR Narrative")
    selected = st.selectbox("Select customer", merged["customer_id"].astype(str).tolist())
    row = merged[merged["customer_id"].astype(str) == selected].iloc[0]

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Customer ID**")
        st.code(row["customer_id"])
    with col_b:
        if "R_e" in row and pd.notna(row["R_e"]):
            color = "#e74c3c" if row["R_e"] >= 0.7 else "#f39c12" if row["R_e"] >= 0.5 else "#2ecc71"
            st.markdown("**Risk Score**")
            st.markdown(f"<span style='font-size:24px;color:{color};font-weight:bold'>{row['R_e']:.3f}</span>", unsafe_allow_html=True)
    with col_c:
        if "triage_decision" in row and pd.notna(row["triage_decision"]):
            color = "#e74c3c" if row["triage_decision"] == "REVIEW" else "#2ecc71"
            st.markdown("**Decision**")
            st.markdown(f"<span style='font-size:20px;color:{color};font-weight:bold'>{row['triage_decision']}</span>", unsafe_allow_html=True)

    if "triage_reason" in row and pd.notna(row["triage_reason"]):
        st.markdown(f"**Reason:** {row['triage_reason']}")

    st.markdown("---")
    st.markdown("#### Draft SAR")
    st.markdown(f"<div style='background:#f8f9fa;padding:20px;border-radius:10px;border-left:4px solid #3498db'>{row['narrative']}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.download_button("📥 Download SAR Drafts", merged.to_csv(index=False), "sar_drafts.csv", "text/csv")


def main():
    st.title("AI-Driven Financial Crime Detection Dashboard")
    st.caption("End-to-end AML & fraud detection with ML + GenAI triage.")

    uploaded_txn, uploaded_kyc, run_button = sidebar_controls()

    if uploaded_txn:
        save_uploaded_file(uploaded_txn, TRANSACTIONS_FILE)
        st.toast("Transactions saved.", icon="✅")
    if uploaded_kyc:
        save_uploaded_file(uploaded_kyc, KYC_FILE)
        st.toast("KYC profiles saved.", icon="✅")

    validation = None
    entity_risk, alerts, triage = load_outputs()

    if run_button:
        with st.spinner("Running pipeline..."):
            validation, entity_risk, alerts, triage = run_full_pipeline()
        st.success("Pipeline finished. Scroll down to explore results.")

    if validation:
        render_validation_status(validation)

    tabs = st.tabs(["📊 Overview", "🧑‍💼 High-Risk Entities", "📈 Analytics", "📄 SAR Narratives"])
    with tabs[0]:
        render_overview_tab(entity_risk, alerts)
    with tabs[1]:
        render_entities_tab(entity_risk)
    with tabs[2]:
        render_analytics_tab(entity_risk)
    with tabs[3]:
        render_sar_tab()


if __name__ == "__main__":
    main()
