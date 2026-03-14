"""ML detection engine - feature engineering, anomaly detection, risk scoring."""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from .config import (
    TRANSACTIONS_FILE, KYC_FILE, OUTPUT_DIR,
    N_SEGMENTS, CONTAMINATION, RISK_THRESHOLD
)


def engineer_features(transactions: pd.DataFrame, kyc: pd.DataFrame) -> pd.DataFrame:
    """Builds behavioral features: velocity, fan-out, deviation from peers."""
    transactions = transactions.copy()
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
    
    txn_kyc = transactions.merge(kyc[["customer_id", "segment"]], on="customer_id", how="left")
    
    # aggregate stats per customer
    cust_stats = txn_kyc.groupby("customer_id").agg(
        txn_count=("transaction_id", "count"),
        total_amount=("amount", "sum"),
        avg_amount=("amount", "mean"),
        max_amount=("amount", "max"),
        unique_counterparties=("counterparty_id", "nunique"),
    ).reset_index()
    
    # velocity = txns per hour
    txn_kyc["hour"] = txn_kyc["timestamp"].dt.floor("h")
    velocity = txn_kyc.groupby(["customer_id", "hour"]).size().reset_index(name="txns_per_hour")
    velocity_agg = velocity.groupby("customer_id")["txns_per_hour"].agg(["mean", "max"]).reset_index()
    velocity_agg.columns = ["customer_id", "velocity_mean", "velocity_max"]
    
    # fan-out = how many unique counterparties
    fan_out = txn_kyc.groupby("customer_id")["counterparty_id"].nunique().reset_index()
    fan_out.columns = ["customer_id", "fan_out"]
    
    # segment averages to compare against
    segment_avgs = cust_stats.merge(kyc[["customer_id", "segment"]], on="customer_id").groupby("segment").agg(
        seg_avg_amount=("avg_amount", "mean"),
        seg_avg_count=("txn_count", "mean"),
    ).reset_index()
    
    # combine everything
    features = cust_stats.merge(velocity_agg, on="customer_id").merge(fan_out, on="customer_id")
    features = features.merge(kyc[["customer_id", "segment"]], on="customer_id")
    features = features.merge(segment_avgs, on="segment")
    
    # how much does this customer deviate from their segment?
    features["amount_deviation"] = (features["avg_amount"] - features["seg_avg_amount"]) / (
        features["seg_avg_amount"].replace(0, np.nan) + 1e-6
    )
    features["count_deviation"] = (features["txn_count"] - features["seg_avg_count"]) / (
        features["seg_avg_count"].replace(0, np.nan) + 1e-6
    )
    
    return features


def run_segmentation(kyc: pd.DataFrame, n_segments: int = N_SEGMENTS) -> pd.DataFrame:
    """Groups customers into peer groups using K-Means."""
    segment_map = {"Retail": 0, "HNW": 1, "Small Business": 2}
    kyc_num = kyc.copy()
    kyc_num["segment_encoded"] = kyc_num["segment"].map(lambda x: segment_map.get(x, 0))
    
    X = kyc_num[["segment_encoded"]].copy()
    X["placeholder"] = np.random.randn(len(X)) * 0.1
    
    kmeans = KMeans(n_clusters=min(n_segments, len(X)), random_state=42, n_init=10)
    kyc_num["peer_group"] = kmeans.fit_predict(X)
    
    return kyc_num[["customer_id", "segment", "peer_group"]]


def detect_anomalies(features: pd.DataFrame) -> pd.DataFrame:
    """Runs Isolation Forest to find outliers."""
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
    features["anomaly_score"] = -iso.decision_function(X_scaled)
    features["is_anomaly"] = iso.predict(X_scaled) == -1
    
    return features


def aggregate_entity_risk(features: pd.DataFrame) -> pd.DataFrame:
    """Normalizes scores to 0-1 range and flags high-risk entities."""
    entity_risk = features.copy()
    
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
    """Full pipeline: features -> segmentation -> anomaly detection -> risk scores."""
    if not transactions_path.exists() or not kyc_path.exists():
        raise FileNotFoundError("Run ingestion first to create data files")
    
    transactions = pd.read_csv(transactions_path)
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
    kyc = pd.read_csv(kyc_path)
    
    kyc_segmented = run_segmentation(kyc)
    features = engineer_features(transactions, kyc_segmented)
    features = detect_anomalies(features)
    entity_risk = aggregate_entity_risk(features)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    entity_risk.to_csv(output_dir / "entity_risk_scores.csv", index=False)
    
    alerts = entity_risk[entity_risk["high_risk"]]
    alerts.to_csv(output_dir / "high_risk_alerts.csv", index=False)
    
    print(f"Detection complete. {len(alerts)} high-risk entities flagged (R_e >= {RISK_THRESHOLD})")
    return entity_risk


if __name__ == "__main__":
    run_detection()
