"""
Streamlit Web UI for the AI-Driven Financial Crime Detection System.

Features:
- Upload real transactions & KYC CSVs (or use existing/synthetic data)
- Run full pipeline: Ingestion -> Validation -> Detection -> GenAI Triage
- Explore high-risk entities and SAR narratives in an interactive dashboard
"""

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
from src.config import DATA_DIR, OUTPUT_DIR, TRANSACTIONS_FILE, KYC_FILE


st.set_page_config(
    page_title="AI Financial Crime Detection",
    page_icon="💸",
    layout="wide",
)


def save_uploaded_file(uploaded_file, dest_path: Path) -> None:
    """Save an uploaded Streamlit file to disk."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


@st.cache_data(show_spinner=False)
def load_outputs() -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load pipeline outputs from disk if they exist."""
    entity_risk = None
    alerts = None
    triage = None

    if (OUTPUT_DIR / "entity_risk_scores.csv").exists():
        entity_risk = pd.read_csv(OUTPUT_DIR / "entity_risk_scores.csv")
    if (OUTPUT_DIR / "high_risk_alerts.csv").exists():
        alerts = pd.read_csv(OUTPUT_DIR / "high_risk_alerts.csv")
    if (OUTPUT_DIR / "triage_results.csv").exists():
        triage = pd.read_csv(OUTPUT_DIR / "triage_results.csv")

    return entity_risk, alerts, triage


def run_full_pipeline() -> tuple:
    """Run ingestion + validation + detection + triage + SAR generation."""
    # Phase 1: Data ingestion & validation
    transactions, kyc = load_or_create_data()
    validation = run_validation_on_dataframes(transactions, kyc)

    # Phase 2: Detection
    entity_risk = run_detection()

    # Phase 3: GenAI triage + SAR drafts
    triage_df = run_triage()
    run_narrative_generation()

    # Clear cache so UI reloads fresh outputs
    load_outputs.clear()
    entity_risk_cached, alerts_cached, triage_cached = load_outputs()

    return validation, entity_risk_cached, alerts_cached, triage_cached


def sidebar_controls():
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.write("Upload real-world CSVs or use existing data on disk.")

    uploaded_txn = st.sidebar.file_uploader(
        "Transactions CSV",
        type="csv",
        help="Must contain transaction_id, customer_id, amount, currency, timestamp, counterparty_id, transaction_type, channel",
    )
    uploaded_kyc = st.sidebar.file_uploader(
        "KYC Profiles CSV",
        type="csv",
        help="Must contain customer_id, segment, country_of_residence (plus optional fields)",
    )

    st.sidebar.markdown("---")
    run_button = st.sidebar.button("🚀 Run Detection Pipeline", use_container_width=True)

    return uploaded_txn, uploaded_kyc, run_button


def render_validation_status(validation) -> None:
    if validation.passed:
        st.success("MRA validation PASSED – data is audit-ready.")
    else:
        st.warning("MRA validation FAILED – see detailed checks in logs / console.")


def render_overview_tab(entity_risk: Optional[pd.DataFrame], alerts: Optional[pd.DataFrame]):
    st.subheader("System Overview")

    if entity_risk is None:
        st.info("Run the pipeline to see metrics.")
        return

    total_entities = len(entity_risk)
    high_risk_count = int(entity_risk.get("high_risk", pd.Series(dtype=bool)).sum())
    low_risk_count = total_entities - high_risk_count
    avg_risk = float(entity_risk.get("R_e", pd.Series(dtype=float)).mean())
    max_risk = float(entity_risk.get("R_e", pd.Series(dtype=float)).max())

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Entities", f"{total_entities:,}")
    col2.metric("High-Risk Entities", f"{high_risk_count:,}", delta=f"{high_risk_count/total_entities*100:.1f}%")
    col3.metric("Average Risk Score", f"{avg_risk:.3f}")
    col4.metric("Max Risk Score", f"{max_risk:.3f}")

    st.markdown("---")

    # Row 1: Pie charts
    col_pie1, col_pie2 = st.columns(2)

    with col_pie1:
        st.markdown("#### Risk Breakdown")
        risk_data = pd.DataFrame({
            "Category": ["High Risk", "Low Risk"],
            "Count": [high_risk_count, low_risk_count]
        })
        fig_risk_pie = px.pie(
            risk_data,
            values="Count",
            names="Category",
            color="Category",
            color_discrete_map={"High Risk": "#e74c3c", "Low Risk": "#2ecc71"},
            hole=0.4
        )
        fig_risk_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
        st.plotly_chart(fig_risk_pie, use_container_width=True)

    with col_pie2:
        st.markdown("#### Segment Distribution")
        if "segment" in entity_risk.columns:
            segment_counts = entity_risk["segment"].value_counts().reset_index()
            segment_counts.columns = ["Segment", "Count"]
            fig_seg_pie = px.pie(
                segment_counts,
                values="Count",
                names="Segment",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4
            )
            fig_seg_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
            st.plotly_chart(fig_seg_pie, use_container_width=True)
        else:
            st.info("Segment data not available.")

    st.markdown("---")

    # Row 2: Histogram and bar chart
    col_hist, col_bar = st.columns(2)

    with col_hist:
        st.markdown("#### Risk Score Distribution (Histogram)")
        if "R_e" in entity_risk.columns:
            fig_hist = px.histogram(
                entity_risk,
                x="R_e",
                nbins=30,
                color_discrete_sequence=["#3498db"],
                labels={"R_e": "Risk Score (R_e)"}
            )
            fig_hist.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="Threshold (0.7)")
            fig_hist.update_layout(
                xaxis_title="Risk Score (R_e)",
                yaxis_title="Number of Entities",
                margin=dict(t=20, b=40, l=40, r=20),
                height=350
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    with col_bar:
        st.markdown("#### Average Risk by Segment")
        if "segment" in entity_risk.columns and "R_e" in entity_risk.columns:
            seg_risk = entity_risk.groupby("segment")["R_e"].agg(["mean", "count"]).reset_index()
            seg_risk.columns = ["Segment", "Avg Risk", "Count"]
            seg_risk = seg_risk.sort_values("Avg Risk", ascending=False)
            fig_bar = px.bar(
                seg_risk,
                x="Segment",
                y="Avg Risk",
                color="Avg Risk",
                color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
                text="Count"
            )
            fig_bar.update_traces(texttemplate="n=%{text}", textposition="outside")
            fig_bar.update_layout(
                xaxis_title="Segment",
                yaxis_title="Average Risk Score",
                coloraxis_showscale=False,
                margin=dict(t=20, b=40, l=40, r=20),
                height=350
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Segment or risk data not available.")

    st.markdown("---")

    # Row 3: Transaction type and channel breakdown (if available)
    col_type, col_channel = st.columns(2)

    with col_type:
        st.markdown("#### Entities by Transaction Type Mix")
        if "unique_counterparties" in entity_risk.columns and "fan_out" in entity_risk.columns:
            fig_scatter = px.scatter(
                entity_risk,
                x="fan_out",
                y="R_e",
                color="high_risk",
                color_discrete_map={True: "#e74c3c", False: "#3498db"},
                labels={"fan_out": "Fan-Out (Unique Counterparties)", "R_e": "Risk Score", "high_risk": "High Risk"},
                hover_data=["customer_id"]
            )
            fig_scatter.update_layout(margin=dict(t=20, b=40, l=40, r=20), height=350)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            # Fallback: show velocity if available
            if "velocity_max" in entity_risk.columns:
                fig_scatter = px.scatter(
                    entity_risk,
                    x="velocity_max",
                    y="R_e",
                    color="high_risk",
                    color_discrete_map={True: "#e74c3c", False: "#3498db"},
                    labels={"velocity_max": "Max Velocity (Txns/Hour)", "R_e": "Risk Score"},
                    hover_data=["customer_id"]
                )
                fig_scatter.update_layout(margin=dict(t=20, b=40, l=40, r=20), height=350)
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Velocity/fan-out data not available.")

    with col_channel:
        st.markdown("#### Risk Score Box Plot by Segment")
        if "segment" in entity_risk.columns and "R_e" in entity_risk.columns:
            fig_box = px.box(
                entity_risk,
                x="segment",
                y="R_e",
                color="segment",
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={"R_e": "Risk Score", "segment": "Segment"}
            )
            fig_box.update_layout(
                showlegend=False,
                margin=dict(t=20, b=40, l=40, r=20),
                height=350
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Segment data not available.")

    st.markdown("---")

    # Data table
    st.markdown("### Entity Risk Scores (Top 50)")
    display_cols = [c for c in ["customer_id", "segment", "R_e", "high_risk", "txn_count", "total_amount", "fan_out", "velocity_max"] if c in entity_risk.columns]
    st.dataframe(
        entity_risk[display_cols].sort_values("R_e", ascending=False).head(50),
        use_container_width=True
    )


def render_entities_tab(entity_risk: Optional[pd.DataFrame]):
    st.subheader("High-Risk Entity Explorer")
    if entity_risk is None or "high_risk" not in entity_risk.columns:
        st.info("Run the pipeline to see high-risk entities.")
        return

    high_risk_df = entity_risk[entity_risk["high_risk"]].copy().sort_values("R_e", ascending=False)
    if high_risk_df.empty:
        st.success("No high-risk entities detected with current threshold.")
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("High-Risk Entities", f"{len(high_risk_df):,}")
    col2.metric("Avg Risk Score", f"{high_risk_df['R_e'].mean():.3f}")
    col3.metric("Max Risk Score", f"{high_risk_df['R_e'].max():.3f}")

    st.markdown("---")

    # Visualization row
    col_rank, col_seg = st.columns(2)

    with col_rank:
        st.markdown("#### Top High-Risk Entities by Score")
        top_n = min(15, len(high_risk_df))
        top_entities = high_risk_df.head(top_n)
        fig_rank = px.bar(
            top_entities,
            x="R_e",
            y="customer_id",
            orientation="h",
            color="R_e",
            color_continuous_scale=["#f39c12", "#e74c3c"],
            labels={"R_e": "Risk Score", "customer_id": "Customer ID"}
        )
        fig_rank.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            coloraxis_showscale=False,
            margin=dict(t=20, b=40, l=100, r=20),
            height=400
        )
        st.plotly_chart(fig_rank, use_container_width=True)

    with col_seg:
        st.markdown("#### High-Risk by Segment")
        if "segment" in high_risk_df.columns:
            seg_counts = high_risk_df["segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Count"]
            fig_seg = px.bar(
                seg_counts,
                x="Segment",
                y="Count",
                color="Segment",
                color_discrete_sequence=px.colors.qualitative.Set1,
                text="Count"
            )
            fig_seg.update_traces(textposition="outside")
            fig_seg.update_layout(
                showlegend=False,
                margin=dict(t=20, b=40, l=40, r=20),
                height=400
            )
            st.plotly_chart(fig_seg, use_container_width=True)
        else:
            st.info("Segment data not available.")

    st.markdown("---")

    # Entity selector
    st.markdown("#### Inspect Individual Entity")
    customer_ids = high_risk_df["customer_id"].astype(str).unique().tolist()
    selected_id = st.selectbox("Select customer", customer_ids)

    selected_rows = high_risk_df[high_risk_df["customer_id"].astype(str) == str(selected_id)]
    row = selected_rows.iloc[0]

    # Display as cards
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Risk Score", f"{row.get('R_e', 0):.3f}")
    col_b.metric("Transaction Count", f"{int(row.get('txn_count', 0)):,}")
    col_c.metric("Total Amount", f"${row.get('total_amount', 0):,.2f}")
    col_d.metric("Fan-Out", f"{int(row.get('fan_out', 0)):,}")

    # Radar chart for entity profile
    st.markdown("##### Entity Risk Profile")
    numeric_cols = ["txn_count", "total_amount", "avg_amount", "velocity_max", "fan_out", "amount_deviation"]
    available_cols = [c for c in numeric_cols if c in row.index and pd.notna(row[c])]

    if len(available_cols) >= 3:
        # Normalize values for radar chart
        values = []
        for col in available_cols:
            val = row[col]
            col_max = entity_risk[col].max()
            values.append(val / col_max if col_max > 0 else 0)
        values.append(values[0])  # Close the radar

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=available_cols + [available_cols[0]],
            fill="toself",
            fillcolor="rgba(231, 76, 60, 0.3)",
            line_color="#e74c3c",
            name=str(selected_id)
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            margin=dict(t=40, b=40, l=60, r=60),
            height=350
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Full details table
    st.markdown("##### Full Details")
    st.dataframe(selected_rows, use_container_width=True)


def render_analytics_tab(entity_risk: Optional[pd.DataFrame]):
    st.subheader("Advanced Analytics")

    if entity_risk is None:
        st.info("Run the pipeline to see analytics.")
        return

    # Filters
    st.markdown("#### Filters")
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        risk_range = st.slider(
            "Risk Score Range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.05
        )

    with col_f2:
        if "segment" in entity_risk.columns:
            segments = ["All"] + entity_risk["segment"].dropna().unique().tolist()
            selected_segment = st.selectbox("Segment", segments)
        else:
            selected_segment = "All"

    with col_f3:
        risk_filter = st.selectbox("Risk Level", ["All", "High Risk Only", "Low Risk Only"])

    # Apply filters
    filtered_df = entity_risk.copy()
    filtered_df = filtered_df[(filtered_df["R_e"] >= risk_range[0]) & (filtered_df["R_e"] <= risk_range[1])]

    if selected_segment != "All" and "segment" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["segment"] == selected_segment]

    if risk_filter == "High Risk Only":
        filtered_df = filtered_df[filtered_df["high_risk"] == True]
    elif risk_filter == "Low Risk Only":
        filtered_df = filtered_df[filtered_df["high_risk"] == False]

    st.markdown(f"**Showing {len(filtered_df):,} entities** (filtered from {len(entity_risk):,})")

    st.markdown("---")

    # Correlation heatmap
    col_corr, col_dist = st.columns(2)

    with col_corr:
        st.markdown("#### Feature Correlation Heatmap")
        numeric_cols = ["R_e", "txn_count", "total_amount", "avg_amount", "max_amount",
                        "velocity_mean", "velocity_max", "fan_out", "amount_deviation"]
        available_numeric = [c for c in numeric_cols if c in filtered_df.columns]

        if len(available_numeric) >= 2:
            corr_matrix = filtered_df[available_numeric].corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            fig_corr.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=400)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough numeric features for correlation analysis.")

    with col_dist:
        st.markdown("#### Feature Distribution")
        if len(available_numeric) > 0:
            selected_feature = st.selectbox("Select feature", available_numeric)
            fig_dist = px.histogram(
                filtered_df,
                x=selected_feature,
                nbins=30,
                color="high_risk" if "high_risk" in filtered_df.columns else None,
                color_discrete_map={True: "#e74c3c", False: "#3498db"},
                barmode="overlay",
                opacity=0.7
            )
            fig_dist.update_layout(margin=dict(t=20, b=40, l=40, r=20), height=400)
            st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")

    # Scatter matrix / pair plot (simplified)
    st.markdown("#### Risk Drivers Analysis")
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("##### Transaction Volume vs Risk")
        if "txn_count" in filtered_df.columns:
            fig_vol = px.scatter(
                filtered_df,
                x="txn_count",
                y="R_e",
                color="high_risk",
                color_discrete_map={True: "#e74c3c", False: "#3498db"},
                trendline="ols" if len(filtered_df) > 10 else None,
                labels={"txn_count": "Transaction Count", "R_e": "Risk Score"},
                hover_data=["customer_id"]
            )
            fig_vol.update_layout(margin=dict(t=20, b=40, l=40, r=20), height=350)
            st.plotly_chart(fig_vol, use_container_width=True)

    with col_s2:
        st.markdown("##### Total Amount vs Risk")
        if "total_amount" in filtered_df.columns:
            fig_amt = px.scatter(
                filtered_df,
                x="total_amount",
                y="R_e",
                color="high_risk",
                color_discrete_map={True: "#e74c3c", False: "#3498db"},
                trendline="ols" if len(filtered_df) > 10 else None,
                labels={"total_amount": "Total Amount ($)", "R_e": "Risk Score"},
                hover_data=["customer_id"]
            )
            fig_amt.update_layout(margin=dict(t=20, b=40, l=40, r=20), height=350)
            st.plotly_chart(fig_amt, use_container_width=True)

    st.markdown("---")

    # Summary statistics table
    st.markdown("#### Summary Statistics")
    if len(available_numeric) > 0:
        stats_df = filtered_df[available_numeric].describe().T
        stats_df = stats_df.round(3)
        st.dataframe(stats_df, use_container_width=True)

    # Download button
    st.markdown("---")
    st.markdown("#### Export Data")
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        csv_all = entity_risk.to_csv(index=False)
        st.download_button(
            label="📥 Download All Entity Scores (CSV)",
            data=csv_all,
            file_name="entity_risk_scores.csv",
            mime="text/csv"
        )

    with col_dl2:
        if "high_risk" in entity_risk.columns:
            csv_hr = entity_risk[entity_risk["high_risk"]].to_csv(index=False)
            st.download_button(
                label="📥 Download High-Risk Only (CSV)",
                data=csv_hr,
                file_name="high_risk_entities.csv",
                mime="text/csv"
            )


def render_sar_tab():
    st.subheader("SAR Draft Narratives")

    sar_path = OUTPUT_DIR / "sar_drafts.csv"
    triage_path = OUTPUT_DIR / "triage_results.csv"

    if not sar_path.exists() or not triage_path.exists():
        st.info("Run the pipeline to generate SAR drafts.")
        return

    sar_df = pd.read_csv(sar_path)
    triage_df = pd.read_csv(triage_path)

    merged = sar_df.merge(triage_df, on="customer_id", how="left")

    # Triage summary
    col_m1, col_m2, col_m3 = st.columns(3)
    total_alerts = len(triage_df)
    review_count = int(triage_df.get("requires_review", pd.Series(dtype=bool)).sum())
    benign_count = total_alerts - review_count

    col_m1.metric("Total Alerts Triaged", f"{total_alerts:,}")
    col_m2.metric("Require Review", f"{review_count:,}")
    col_m3.metric("Benign (Filtered)", f"{benign_count:,}")

    st.markdown("---")

    # Triage breakdown chart
    col_pie, col_table = st.columns([1, 2])

    with col_pie:
        st.markdown("#### Triage Decisions")
        if "triage_decision" in triage_df.columns:
            decision_counts = triage_df["triage_decision"].value_counts().reset_index()
            decision_counts.columns = ["Decision", "Count"]
            fig_triage = px.pie(
                decision_counts,
                values="Count",
                names="Decision",
                color="Decision",
                color_discrete_map={"REVIEW": "#e74c3c", "BENIGN": "#2ecc71"},
                hole=0.4
            )
            fig_triage.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=250)
            st.plotly_chart(fig_triage, use_container_width=True)

    with col_table:
        st.markdown("#### All Triage Results")
        display_cols = [c for c in ["customer_id", "R_e", "triage_decision", "triage_reason"] if c in merged.columns]
        st.dataframe(merged[display_cols], use_container_width=True, height=250)

    st.markdown("---")

    # Individual SAR viewer
    st.markdown("#### View SAR Narrative")
    customer_ids = merged["customer_id"].astype(str).tolist()
    selected_id = st.selectbox("Select customer for SAR narrative", customer_ids)

    row = merged[merged["customer_id"].astype(str) == str(selected_id)].iloc[0]

    # Display as styled cards
    col_info1, col_info2, col_info3 = st.columns(3)

    with col_info1:
        st.markdown(f"**Customer ID**")
        st.code(row["customer_id"])

    with col_info2:
        if "R_e" in row and pd.notna(row["R_e"]):
            st.markdown("**Risk Score (R_e)**")
            risk_color = "#e74c3c" if row["R_e"] >= 0.7 else "#f39c12" if row["R_e"] >= 0.5 else "#2ecc71"
            st.markdown(f"<span style='font-size: 24px; color: {risk_color}; font-weight: bold;'>{row['R_e']:.3f}</span>", unsafe_allow_html=True)

    with col_info3:
        if "triage_decision" in row and pd.notna(row["triage_decision"]):
            st.markdown("**Triage Decision**")
            decision_color = "#e74c3c" if row["triage_decision"] == "REVIEW" else "#2ecc71"
            st.markdown(f"<span style='font-size: 20px; color: {decision_color}; font-weight: bold;'>{row['triage_decision']}</span>", unsafe_allow_html=True)

    if "triage_reason" in row and pd.notna(row["triage_reason"]):
        st.markdown(f"**Triage Reason:** {row['triage_reason']}")

    st.markdown("---")
    st.markdown("#### Draft SAR Narrative")
    st.markdown(
        f"""<div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #3498db;">
        {row["narrative"]}
        </div>""",
        unsafe_allow_html=True
    )

    # Download SAR
    st.markdown("---")
    st.download_button(
        label="📥 Download All SAR Drafts (CSV)",
        data=merged.to_csv(index=False),
        file_name="sar_drafts_with_triage.csv",
        mime="text/csv"
    )


def main():
    st.title("AI-Driven Financial Crime Detection Dashboard")
    st.caption("End-to-end AML & fraud detection with ML + GenAI triage.")

    uploaded_txn, uploaded_kyc, run_button = sidebar_controls()

    # If the user uploaded files, save them to the expected locations
    if uploaded_txn is not None:
        save_uploaded_file(uploaded_txn, TRANSACTIONS_FILE)
        st.toast("Uploaded transactions.csv saved. It will be used on next run.", icon="✅")
    if uploaded_kyc is not None:
        save_uploaded_file(uploaded_kyc, KYC_FILE)
        st.toast("Uploaded kyc_profiles.csv saved. It will be used on next run.", icon="✅")

    validation = None
    entity_risk, alerts, triage = load_outputs()

    if run_button:
        with st.spinner("Running full pipeline (this may take a moment)..."):
            validation, entity_risk, alerts, triage = run_full_pipeline()
        st.success("Pipeline finished. Scroll down to explore results.")

    if validation is not None:
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

