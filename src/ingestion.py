"""Data ingestion - loads real CSVs or generates synthetic data for demo."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .config import DATA_DIR, TRANSACTIONS_FILE, KYC_FILE


def generate_synthetic_transactions(n_customers: int = 500, n_transactions: int = 5000) -> pd.DataFrame:
    """Creates fake transaction data with some anomalies baked in."""
    np.random.seed(42)
    
    transaction_types = ["wire", "ach", "card", "check", "atm"]
    channels = ["online", "branch", "mobile", "atm"]
    currencies = ["USD", "USD", "USD", "EUR"]
    customers = [f"CUST_{i:05d}" for i in range(n_customers)]
    
    transactions = []
    base_date = datetime(2024, 1, 1)
    
    for i in range(n_transactions):
        cust = np.random.choice(customers)
        amount = np.random.lognormal(mean=5, sigma=2)
        
        # ~5% are high-value outliers (anomalies)
        if np.random.random() < 0.05:
            amount *= np.random.uniform(10, 100)
        
        transactions.append({
            "transaction_id": f"TXN_{i:08d}",
            "customer_id": cust,
            "amount": round(amount, 2),
            "currency": np.random.choice(currencies),
            "timestamp": base_date + timedelta(minutes=np.random.randint(0, 525600)),
            "counterparty_id": f"CP_{np.random.randint(1, 200):04d}",
            "transaction_type": np.random.choice(transaction_types),
            "channel": np.random.choice(channels),
        })
    
    df = pd.DataFrame(transactions)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def generate_synthetic_kyc(customer_ids: list) -> pd.DataFrame:
    """Creates fake KYC profiles with segments and notes."""
    segments = ["Retail", "Retail", "Retail", "HNW", "Small Business"]
    
    kyc_data = []
    for cust in set(customer_ids):
        kyc_data.append({
            "customer_id": cust,
            "segment": np.random.choice(segments),
            "account_open_date": (datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 1200))).strftime("%Y-%m-%d"),
            "country_of_residence": np.random.choice(["US", "US", "US", "UK", "CA"]),
            "occupation": np.random.choice(["employed", "retired", "self_employed", "business_owner"]),
            "notes": np.random.choice([
                "Regular customer, no concerns",
                "Travels frequently for business",
                "Recently relocated",
                "High-value investor",
                ""
            ]),
            "travel_tag": np.random.choice(["domestic", "international", "international", "domestic", None]),
        })
    
    return pd.DataFrame(kyc_data)


def load_or_create_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads existing CSVs or creates synthetic data if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if TRANSACTIONS_FILE.exists():
        transactions = pd.read_csv(TRANSACTIONS_FILE)
        transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
        print(f"Loaded {len(transactions)} transactions from {TRANSACTIONS_FILE}")
    else:
        transactions = generate_synthetic_transactions()
        transactions.to_csv(TRANSACTIONS_FILE, index=False)
        print(f"Generated and saved {len(transactions)} synthetic transactions to {TRANSACTIONS_FILE}")
    
    if KYC_FILE.exists():
        kyc = pd.read_csv(KYC_FILE)
        print(f"Loaded {len(kyc)} KYC profiles from {KYC_FILE}")
    else:
        kyc = generate_synthetic_kyc(transactions["customer_id"].tolist())
        kyc.to_csv(KYC_FILE, index=False)
        print(f"Generated and saved {len(kyc)} synthetic KYC profiles to {KYC_FILE}")
    
    return transactions, kyc


if __name__ == "__main__":
    txn, kyc = load_or_create_data()
    print("\nTransactions sample:")
    print(txn.head())
    print("\nKYC sample:")
    print(kyc.head())
