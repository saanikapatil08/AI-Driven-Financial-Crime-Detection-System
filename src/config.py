"""Configuration for the Financial Crime Detection System."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Data files
TRANSACTIONS_FILE = DATA_DIR / "transactions.csv"
KYC_FILE = DATA_DIR / "kyc_profiles.csv"

# MRA Validation thresholds
MIN_COMPLETENESS = 0.95  # 95% required for audit
MAX_NULL_RATE = 0.05
REQUIRED_COLUMNS_TRANSACTIONS = [
    "transaction_id", "customer_id", "amount", "currency", "timestamp",
    "counterparty_id", "transaction_type", "channel"
]

# ML parameters
N_SEGMENTS = 4  # Peer groups: Retail, HNW, Small Business, etc.
CONTAMINATION = 0.05  # Expected proportion of anomalies
RISK_THRESHOLD = 0.7  # Entity-level risk score threshold for alerting

# GenAI (optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
