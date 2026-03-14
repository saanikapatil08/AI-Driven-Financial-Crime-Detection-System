#!/usr/bin/env python3
"""Main script - runs the full detection pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ingestion import load_or_create_data
from src.validation import run_validation_on_dataframes
from src.detection import run_detection
from src.triage_agent import run_triage, run_narrative_generation
from src.config import OUTPUT_DIR


def main():
    print("=" * 60)
    print("AI-Driven Financial Crime Detection System")
    print("=" * 60)
    
    print("\n[Phase 1] Data Ingestion & MRA Validation")
    transactions, kyc = load_or_create_data()
    
    validation = run_validation_on_dataframes(transactions, kyc)
    print(f"  {validation.message}")
    if not validation.passed:
        print("  WARNING: Validation failed. Proceeding with caution.")
    
    print("\n[Phase 2] ML Detection Engine")
    entity_risk = run_detection()
    print(f"  Output: {OUTPUT_DIR / 'entity_risk_scores.csv'}")
    print(f"  Alerts: {OUTPUT_DIR / 'high_risk_alerts.csv'}")
    
    print("\n[Phase 3] GenAI Triage & Narrative Generation")
    run_triage()
    run_narrative_generation()
    print(f"  Triage: {OUTPUT_DIR / 'triage_results.csv'}")
    print(f"  SAR drafts: {OUTPUT_DIR / 'sar_drafts.csv'}")
    
    print("\n" + "=" * 60)
    print("Pipeline complete. Check ./output/ for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
