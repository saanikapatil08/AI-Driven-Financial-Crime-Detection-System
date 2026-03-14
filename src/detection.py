"""
ML Detection Engine - Anomaly detection and Entity-Level Risk Scoring.
Uses K-Means segmentation, velocity/fan-in/fan-out features, and Isolation Forest.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from .config import (
    TRANSACTIONS_FILE, KYC_FILE, DATA_DIR, OUTPUT_DIR,
    N_SEGMENTS, CONTAMINATION, RISK_THRESHOLD
)


def engineer_features(transactions: pd.DataFrame, kyc: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate behavioral features: velocity, fan-in/fan-out, deviation from segment.
    """
    transactions = transactions.copy()
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
    
    # Merge with KYC for segment
    txn_kyc = transactions.merge(
        kyc[["customer_id", "segment"]],
        on="customer_id",
        how="left"
    )
    
    # Customer-level aggregates
    cust_stats = txn_kyc.groupby("customer_id").agg(
        txn_count=("transaction_id", "count"),
        total_amount=("amount", "sum"),
        avg_amount=("amount", "mean"),
        max_amount=("amount", "max"),
        unique_counterparties=("counterparty_id", "nunique"),
    ).reset_index()
    
    # Velocity: transactions per hour (approximate)
    txn_kyc["hour"] = txn_kyc["timestamp"].dt.floor("h")
    velocity = txn_kyc.groupby(["customer_id", "hour"]).size().reset_index(name="txns_per_hour")
    velocity_agg = velocity.groupby("customer_id")["txns_per_hour"].agg(["mean", "max"]).reset_index()
    velocity_agg.columns = ["customer_id", "velocity_mean", "velocity_max"]
    
    # Fan-in: how many unique senders to this customer (counterparty -> customer)
    # Fan-out: how many unique receivers from this customer
    fan_out = txn_kyc.groupby("customer_id")["counterparty_id"].nunique().reset_index()
    fan_out.columns = ["customer_id", "fan_out"]
    
    # Segment averages for deviation
    segment_avgs = cust_stats.merge(
        kyc[["customer_id", "segment"]],
        on="customer_id"
    ).groupby("segment").agg(
        seg_avg_amount=("avg_amount", "mean"),
        seg_avg_count=("txn_count", "mean"),
    ).reset_index()
    
    # Build feature matrix
    features = cust_stats.merge(velocity_agg, on="customer_id").merge(fan_out, on="customer_id")
    features = features.merge(kyc[["customer_id", "segment"]], on="customer_id")
    features = features.merge(segment_avgs, on="segment")
    
    features["amount_deviation"] = (features["avg_amount"] - features["seg_avg_amount"]) / (
        features["seg_avg_amount"].replace(0, np.nan) + 1e-6
    )
    features["count_deviation"] = (features["txn_count"] - features["seg_avg_count"]) / (
        features["seg_avg_count"].replace(0, np.nan) + 1e-6
    )
    
    return features


def run_segmentation(kyc: pd.DataFrame, n_segments: int = N_SEGMENTS) -> pd.DataFrame:
    """
    Segment customers into peer groups using K-Means on simple profile features.
    """
    # Use segment as proxy - in production, use numeric KYC features
    segment_map = {"Retail": 0, "HNW": 1, "Small Business": 2}
    kyc_num = kyc.copy()
    kyc_num["segment_encoded"] = kyc_num["segment"].map(
        lambda x: segment_map.get(x, 0)
    )
    
    # For clustering, we'd use more features; here we use segment + placeholder
    X = kyc_num[["segment_encoded"]].copy()
    X["placeholder"] = np.random.randn(len(X)) * 0.1  # Add noise for clustering
    
    kmeans = KMeans(n_clusters=min(n_segments, len(X)), random_state=42, n_init=10)
    kyc_num["peer_group"] = kmeans.fit_predict(X)
    
    return kyc_num[["customer_id", "segment", "peer_group"]]


def detect_anomalies(features: pd.DataFrame) -> pd.DataFrame:
    """
    Use Isolation Forest to flag anomalous entities.
    """
    numeric_cols = [
        "txn_count", "total_amount", "avg_amount", "max_amount",
        "unique_counterparties", "velocity_mean", "velocity_max",
        "fan_out", "amount_deviation", "count_deviation"
    ]
    numeric_cols = [c for c in numeric_cols if c in features.columns]
    
    X = features[numeric_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    iso = IsolationForest(contamination=CONTAMINATION, random_state=42)
    iso.fit(X_scaled)
    features = features.copy()
    features["anomaly_score"] = -iso.decision_function(X_scaled)  # Higher = more anomalous
    features["is_anomaly"] = iso.predict(X_scaled) == -1
    
    return features


def aggregate_entity_risk(features: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction-level scores to Entity (customer) level.
    R_e = unified risk weight per customer.
    """
    entity_risk = features.copy()
    # Normalize anomaly score to [0, 1]
    min_s, max_s = entity_risk["anomaly_score"].min(), entity_risk["anomaly_score"].max()
    if max_s > min_s:
        entity_risk["R_e"] = (entity_risk["anomaly_score"] - min_s) / (max_s - min_s)
    else:
        entity_risk["R_e"] = 0
    
    entity_risk["high_risk"] = entity_risk["R_e"] >= RISK_THRESHOLD
    return entity_risk


def run_detection(
    transactions_path: Path = TRANSACTIONS_FILE,
    kyc_path: Path = KYC_FILE,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Run full detection pipeline: feature engineering, segmentation, anomaly detection,
    and entity-level risk scoring.
    Returns entity risk dataframe with alerts.
    """
    # Load data
    if not transactions_path.exists() or not kyc_path.exists():
        raise FileNotFoundError(
            "Run ingestion first: python -c \"from src.ingestion import load_or_create_data; load_or_create_data()\""
        )
    
    transactions = pd.read_csv(transactions_path)
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
    kyc = pd.read_csv(kyc_path)
    
    # Segmentation
    kyc_segmented = run_segmentation(kyc)
    
    # Feature engineering
    features = engineer_features(transactions, kyc_segmented)
    
    # Anomaly detection
    features = detect_anomalies(features)
    
    # Entity-level risk
    entity_risk = aggregate_entity_risk(features)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    entity_risk.to_csv(output_dir / "entity_risk_scores.csv", index=False)
    
    alerts = entity_risk[entity_risk["high_risk"]]
    alerts.to_csv(output_dir / "high_risk_alerts.csv", index=False)
    
    n_alerts = len(alerts)
    print(f"Detection complete. {n_alerts} high-risk entities flagged (R_e >= {RISK_THRESHOLD})")
    return entity_risk


if __name__ == "__main__":
    run_detection()
