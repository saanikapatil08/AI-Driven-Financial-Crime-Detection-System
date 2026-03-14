"""Project configuration and thresholds."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TRANSACTIONS_FILE = DATA_DIR / "transactions.csv"
KYC_FILE = DATA_DIR / "kyc_profiles.csv"

# validation thresholds
MIN_COMPLETENESS = 0.95
MAX_NULL_RATE = 0.05
REQUIRED_COLUMNS_TRANSACTIONS = [
    "transaction_id", "customer_id", "amount", "currency", "timestamp",
    "counterparty_id", "transaction_type", "channel"
]

# ML tuning
N_SEGMENTS = 4
CONTAMINATION = 0.05
RISK_THRESHOLD = 0.7

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
