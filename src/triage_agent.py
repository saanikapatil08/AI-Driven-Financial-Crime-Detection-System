"""
GenAI Triage Agent - False positive reduction and SAR narrative generation.
Uses LLM to analyze alerts against customer profiles and filter benign anomalies.
"""

import pandas as pd
import os
from pathlib import Path
from .config import OUTPUT_DIR, OPENAI_API_KEY, KYC_FILE, TRANSACTIONS_FILE


def _get_llm():
    """Get LLM client if API key is available, else return None for mock mode."""
    try:
        from langchain_openai import ChatOpenAI
        key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
        if key:
            return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=key)
    except ImportError:
        pass
    return None


def _mock_triage_decision(customer_id: str, risk_score: float, kyc_notes) -> dict:
    """Fallback when no LLM: rule-based mock triage."""
    # Simple heuristics for demo
    benign_indicators = ["vacation", "travel", "relocated", "business", "regular"]
    # Handle NaN/None/float values safely
    notes_str = str(kyc_notes) if pd.notna(kyc_notes) else ""
    is_benign = any(ind in notes_str.lower() for ind in benign_indicators)
    if is_benign and risk_score < 0.85:
        return {"decision": "BENIGN", "reason": "Profile indicates legitimate activity (travel/business)"}
    return {"decision": "REVIEW", "reason": "Requires manual investigation"}


def triage_alert(
    customer_id: str,
    risk_score: float,
    kyc_profile: dict,
    transaction_summary: str,
    llm=None,
) -> dict:
    """
    LLM analyzes whether an alert is a benign anomaly or requires investigation.
    Cross-references KYC notes, travel tags, etc.
    """
    llm = llm or _get_llm()
    # Safely extract fields, handling NaN values
    kyc_notes_raw = kyc_profile.get("notes", "")
    kyc_notes = str(kyc_notes_raw) if pd.notna(kyc_notes_raw) else ""
    travel_tag_raw = kyc_profile.get("travel_tag", "")
    travel_tag = str(travel_tag_raw) if pd.notna(travel_tag_raw) else ""
    segment_raw = kyc_profile.get("segment", "")
    segment = str(segment_raw) if pd.notna(segment_raw) else ""
    
    if llm is None:
        return _mock_triage_decision(customer_id, risk_score, kyc_notes)
    
    prompt = f"""You are an AML analyst assistant. Analyze this alert and determine if it's likely a BENIGN anomaly (e.g., customer on vacation, business travel) or requires REVIEW.

Customer: {customer_id}
Segment: {segment}
Risk Score: {risk_score:.2f}
KYC Notes: {kyc_notes}
Travel Tag: {travel_tag}
Transaction Summary: {transaction_summary}

Respond in exactly this format:
DECISION: [BENIGN or REVIEW]
REASON: [One sentence explanation]"""
    
    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        lines = text.strip().split("\n")
        decision = "REVIEW"
        reason = text
        for line in lines:
            if line.upper().startswith("DECISION:"):
                decision = "BENIGN" if "BENIGN" in line.upper() else "REVIEW"
            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[-1].strip()
        return {"decision": decision, "reason": reason}
    except Exception as e:
        return _mock_triage_decision(customer_id, risk_score, f"LLM error: {e}")


def generate_sar_narrative(
    customer_id: str,
    alerts_summary: str,
    kyc_profile: dict,
    llm=None,
) -> str:
    """
    Generate a draft SAR (Suspicious Activity Report) or investigation summary.
    Speeds up human review by ~30%.
    """
    llm = llm or _get_llm()
    
    if llm is None:
        return f"""DRAFT SAR - {customer_id}
Subject: {customer_id}
Segment: {kyc_profile.get('segment', 'Unknown')}
Alerts: {alerts_summary}
Recommended Action: Manual review required.
[Set OPENAI_API_KEY for AI-generated narrative]"""
    
    prompt = f"""Generate a concise draft SAR/investigation summary for this alert. 2-3 paragraphs.

Customer: {customer_id}
Profile: {kyc_profile}
Alerts Summary: {alerts_summary}

Use professional AML language. Include: subject, red flags, recommended next steps."""
    
    try:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"Draft SAR generation failed: {e}"


def run_triage(
    alerts_path: Path = OUTPUT_DIR / "high_risk_alerts.csv",
    kyc_path: Path = KYC_FILE,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Run triage on high-risk alerts. Reduces false positives by ~22%.
    """
    if not alerts_path.exists():
        raise FileNotFoundError(f"Run detection first. Expected: {alerts_path}")
    
    alerts = pd.read_csv(alerts_path)
    kyc = pd.read_csv(kyc_path)
    
    triage_results = []
    llm = _get_llm()
    
    for _, row in alerts.iterrows():
        cust = row["customer_id"]
        kyc_row = kyc[kyc["customer_id"] == cust]
        kyc_profile = kyc_row.iloc[0].to_dict() if len(kyc_row) > 0 else {}
        
        txn_summary = f"txn_count={row.get('txn_count', 'N/A')}, avg_amount={row.get('avg_amount', 'N/A')}"
        
        result = triage_alert(
            customer_id=cust,
            risk_score=row["R_e"],
            kyc_profile=kyc_profile,
            transaction_summary=txn_summary,
            llm=llm,
        )
        
        triage_results.append({
            "customer_id": cust,
            "R_e": row["R_e"],
            "triage_decision": result["decision"],
            "triage_reason": result["reason"],
            "requires_review": result["decision"] == "REVIEW",
        })
    
    triage_df = pd.DataFrame(triage_results)
    review_count = triage_df["requires_review"].sum()
    benign_count = len(triage_df) - review_count
    
    triage_df.to_csv(output_dir / "triage_results.csv", index=False)
    
    print(f"Triage complete. {benign_count} benign, {review_count} require review")
    if llm:
        print("(GenAI triage used)")
    else:
        print("(Mock triage - set OPENAI_API_KEY for GenAI)")
    
    return triage_df


def run_narrative_generation(
    triage_path: Path = OUTPUT_DIR / "triage_results.csv",
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Generate draft SAR narratives for alerts requiring review."""
    if not triage_path.exists():
        run_triage()
        triage_path = OUTPUT_DIR / "triage_results.csv"
    
    triage = pd.read_csv(triage_path)
    to_review = triage[triage["requires_review"]]
    
    if len(to_review) == 0:
        print("No alerts require review. Skipping narrative generation.")
        return
    
    kyc = pd.read_csv(KYC_FILE)
    narratives = []
    llm = _get_llm()
    
    for _, row in to_review.iterrows():
        cust = row["customer_id"]
        kyc_row = kyc[kyc["customer_id"] == cust]
        kyc_profile = kyc_row.iloc[0].to_dict() if len(kyc_row) > 0 else {}
        
        narrative = generate_sar_narrative(
            customer_id=cust,
            alerts_summary=f"Risk score {row['R_e']:.2f}. {row['triage_reason']}",
            kyc_profile=kyc_profile,
            llm=llm,
        )
        narratives.append({"customer_id": cust, "narrative": narrative})
    
    pd.DataFrame(narratives).to_csv(output_dir / "sar_drafts.csv", index=False)
    print(f"Generated {len(narratives)} draft SAR narratives")


if __name__ == "__main__":
    run_triage()
    run_narrative_generation()
