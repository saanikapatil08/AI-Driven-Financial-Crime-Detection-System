"""MRA-aligned data validation - checks completeness, accuracy, timeliness."""

import pandas as pd
from pathlib import Path
from typing import Optional
from .config import (
    TRANSACTIONS_FILE, KYC_FILE, MIN_COMPLETENESS, MAX_NULL_RATE,
    REQUIRED_COLUMNS_TRANSACTIONS
)


class MRAValidationResult:
    def __init__(self, passed: bool, checks: dict, message: str = ""):
        self.passed = passed
        self.checks = checks
        self.message = message


def validate_schema(df: pd.DataFrame, required_cols: list) -> tuple[bool, dict]:
    """Checks if all required columns exist."""
    missing = set(required_cols) - set(df.columns)
    passed = len(missing) == 0
    return passed, {"schema": {"required_columns": required_cols, "missing": list(missing)}}


def validate_completeness(df: pd.DataFrame, critical_cols: Optional[list] = None) -> tuple[bool, dict]:
    """Checks null rates - critical columns must have <5% nulls."""
    critical_cols = critical_cols or list(df.columns)
    critical_cols = [c for c in critical_cols if c in df.columns]
    results = {}
    all_passed = True

    for col in df.columns:
        null_rate = df[col].isna().mean()
        is_critical = col in critical_cols
        passed = (null_rate <= MAX_NULL_RATE) if is_critical else True
        results[col] = {"null_rate": float(null_rate), "passed": passed, "critical": is_critical}
        if is_critical and not passed:
            all_passed = False

    critical_df = df[critical_cols] if critical_cols else df
    overall_completeness = 1 - critical_df.isna().mean().mean()
    return all_passed and overall_completeness >= MIN_COMPLETENESS, results


def validate_accuracy_transactions(df: pd.DataFrame) -> tuple[bool, dict]:
    """Checks amounts are positive, timestamps valid, IDs non-empty."""
    checks = {}
    
    invalid_amounts = (df["amount"] <= 0).sum()
    checks["amount_positive"] = {"invalid_count": int(invalid_amounts), "passed": invalid_amounts == 0}
    
    if "timestamp" in df.columns:
        df_ts = pd.to_datetime(df["timestamp"], errors="coerce")
        invalid_ts = df_ts.isna().sum()
        checks["timestamp_valid"] = {"invalid_count": int(invalid_ts), "passed": invalid_ts == 0}
    
    for col in ["customer_id", "transaction_id"]:
        if col in df.columns:
            empty_ids = df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
            checks[f"{col}_non_empty"] = {"empty_count": int(empty_ids), "passed": empty_ids == 0}
    
    all_passed = all(c["passed"] for c in checks.values())
    return all_passed, checks


def validate_timeliness(df: pd.DataFrame, timestamp_col: str = "timestamp") -> tuple[bool, dict]:
    """Checks no future-dated or very old records."""
    if timestamp_col not in df.columns:
        return True, {}
    
    df_ts = pd.to_datetime(df[timestamp_col])
    now = pd.Timestamp.now()
    future_dated = (df_ts > now).sum()
    very_old = (df_ts < now - pd.Timedelta(days=365 * 5)).sum()
    
    passed = future_dated == 0 and very_old == 0
    return passed, {"future_dated": int(future_dated), "very_old": int(very_old)}


def run_validation(
    transactions_path: Path = TRANSACTIONS_FILE,
    kyc_path: Path = KYC_FILE,
) -> MRAValidationResult:
    """Runs full validation on files."""
    checks = {"transactions": {}, "kyc": {}}
    all_passed = True
    
    if transactions_path.exists():
        txn = pd.read_csv(transactions_path)
        
        schema_ok, schema_checks = validate_schema(txn, REQUIRED_COLUMNS_TRANSACTIONS)
        checks["transactions"]["schema"] = schema_checks
        if not schema_ok:
            all_passed = False
        
        complete_ok, complete_checks = validate_completeness(txn, REQUIRED_COLUMNS_TRANSACTIONS)
        checks["transactions"]["completeness"] = complete_checks
        if not complete_ok:
            all_passed = False
        
        accuracy_ok, accuracy_checks = validate_accuracy_transactions(txn)
        checks["transactions"]["accuracy"] = accuracy_checks
        if not accuracy_ok:
            all_passed = False
        
        timeliness_ok, timeliness_checks = validate_timeliness(txn)
        checks["transactions"]["timeliness"] = timeliness_checks
        if not timeliness_ok:
            all_passed = False
    else:
        all_passed = False
        checks["transactions"]["error"] = "Transactions file not found"
    
    if kyc_path.exists():
        kyc = pd.read_csv(kyc_path)
        schema_ok, schema_checks = validate_schema(kyc, ["customer_id", "segment", "country_of_residence"])
        checks["kyc"]["schema"] = schema_checks
        complete_ok, complete_checks = validate_completeness(kyc, ["customer_id", "segment", "country_of_residence"])
        checks["kyc"]["completeness"] = complete_checks
        if not schema_ok or not complete_ok:
            all_passed = False
    else:
        checks["kyc"]["error"] = "KYC file not found"
    
    msg = "MRA validation PASSED - data is audit-ready" if all_passed else "MRA validation FAILED - review checks"
    return MRAValidationResult(passed=all_passed, checks=checks, message=msg)


def run_validation_on_dataframes(transactions_df: pd.DataFrame, kyc_df: pd.DataFrame) -> MRAValidationResult:
    """Runs validation on in-memory dataframes."""
    checks = {"transactions": {}, "kyc": {}}
    all_passed = True
    
    schema_ok, schema_checks = validate_schema(transactions_df, REQUIRED_COLUMNS_TRANSACTIONS)
    checks["transactions"]["schema"] = schema_checks
    if not schema_ok:
        all_passed = False

    complete_ok, complete_checks = validate_completeness(transactions_df, REQUIRED_COLUMNS_TRANSACTIONS)
    checks["transactions"]["completeness"] = complete_checks
    if not complete_ok:
        all_passed = False
    
    accuracy_ok, accuracy_checks = validate_accuracy_transactions(transactions_df)
    checks["transactions"]["accuracy"] = accuracy_checks
    if not accuracy_ok:
        all_passed = False
    
    timeliness_ok, timeliness_checks = validate_timeliness(transactions_df)
    checks["transactions"]["timeliness"] = timeliness_checks
    if not timeliness_ok:
        all_passed = False
    
    schema_ok_kyc, schema_checks_kyc = validate_schema(kyc_df, ["customer_id", "segment", "country_of_residence"])
    checks["kyc"]["schema"] = schema_checks_kyc
    complete_ok_kyc, complete_checks_kyc = validate_completeness(kyc_df, ["customer_id", "segment", "country_of_residence"])
    checks["kyc"]["completeness"] = complete_checks_kyc
    if not schema_ok_kyc or not complete_ok_kyc:
        all_passed = False
    
    msg = "MRA validation PASSED - data is audit-ready" if all_passed else "MRA validation FAILED - review checks"
    return MRAValidationResult(passed=all_passed, checks=checks, message=msg)


if __name__ == "__main__":
    result = run_validation()
    print(result.message)
    print("Checks:", result.checks)
