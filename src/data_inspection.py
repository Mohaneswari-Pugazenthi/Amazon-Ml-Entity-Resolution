"""
data_inspection.py
------------------
Comprehensive data inspection and EDA functions for the Amazon ML Entity Resolution datasets.
Designed to be fast, vectorized, and memory-efficient.
Does NOT mutate or modify any raw datasets.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd


def inspect_dataframe_basic(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """
    Performs fast, vectorized basic inspection of a single source dataset DataFrame.
    Returns a dictionary of statistics.
    """
    n_rows, n_cols = df.shape

    # Missing / Blank analysis per column
    missing_stats = {}
    for col in df.columns:
        series = df[col].astype(str)
        is_empty = (series == "") | (series.str.strip() == "")
        empty_count = int(is_empty.sum())
        missing_stats[col] = {
            "empty_count": empty_count,
            "empty_pct": float(round((empty_count / n_rows) * 100, 4)) if n_rows > 0 else 0.0,
        }

    # Entity ID uniqueness
    unique_ids = int(df["entity_id"].nunique()) if "entity_id" in df.columns else 0
    duplicate_ids = n_rows - unique_ids
    duplicate_id_pct = float(round((duplicate_ids / n_rows) * 100, 4)) if n_rows > 0 else 0.0

    # Fast vectorized prefix breakdown
    prefix_counts = {}
    if "entity_id" in df.columns:
        prefixes = df["entity_id"].str.split("-").str[0]
        prefix_counts = {str(k): int(v) for k, v in prefixes.value_counts().to_dict().items()}

    # Business Name & Address duplicates
    unique_names = int(df["business_name"].nunique()) if "business_name" in df.columns else 0
    dup_names = n_rows - unique_names if "business_name" in df.columns else 0
    dup_name_pct = float(round((dup_names / n_rows) * 100, 4)) if n_rows > 0 else 0.0

    unique_addrs = int(df["business_address"].nunique()) if "business_address" in df.columns else 0
    dup_addrs = n_rows - unique_addrs if "business_address" in df.columns else 0
    dup_addr_pct = float(round((dup_addrs / n_rows) * 100, 4)) if n_rows > 0 else 0.0

    # Country distribution
    country_counts = {}
    country_pcts = {}
    if "country" in df.columns:
        c_series = df["country"].str.strip()
        c_counts = c_series.value_counts().to_dict()
        country_counts = {str(k): int(v) for k, v in c_counts.items()}
        country_pcts = {
            str(k): float(round((v / n_rows) * 100, 2)) for k, v in c_counts.items()
        }

    # Vectorized string length statistics
    length_stats = {}
    for col in ["business_name", "business_address"]:
        if col in df.columns:
            lens = df[col].astype(str).str.len()
            length_stats[col] = {
                "min": int(lens.min()) if len(lens) > 0 else 0,
                "max": int(lens.max()) if len(lens) > 0 else 0,
                "mean": float(round(lens.mean(), 2)) if len(lens) > 0 else 0.0,
                "median": float(round(lens.median(), 2)) if len(lens) > 0 else 0.0,
                "std": float(round(lens.std(), 2)) if len(lens) > 0 else 0.0,
                "p95": float(round(np.percentile(lens, 95), 2)) if len(lens) > 0 else 0.0,
            }

    return {
        "dataset_name": dataset_name,
        "rows": n_rows,
        "cols": n_cols,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_stats": missing_stats,
        "unique_entity_ids": unique_ids,
        "duplicate_entity_ids": duplicate_ids,
        "duplicate_entity_id_pct": duplicate_id_pct,
        "unique_business_names": unique_names,
        "duplicate_business_names": dup_names,
        "duplicate_business_name_pct": dup_name_pct,
        "unique_business_addresses": unique_addrs,
        "duplicate_business_addresses": dup_addrs,
        "duplicate_business_address_pct": dup_addr_pct,
        "prefix_counts": prefix_counts,
        "country_counts": country_counts,
        "country_pcts": country_pcts,
        "length_stats": length_stats,
    }


def inspect_ground_truth(gt_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes the train_ground_truth.tsv DataFrame using vectorized string operations.
    Returns distribution of matches per Source 1 entity.
    """
    n_rows = len(gt_df)

    matched_series = gt_df["matched_entity_ids"].astype(str).str.strip()
    is_empty = matched_series == ""

    # Count commas + 1 for non-empty rows
    match_counts = np.where(is_empty, 0, matched_series.str.count(",") + 1)
    match_counts_series = pd.Series(match_counts)

    n_singletons = int((match_counts == 0).sum())
    n_one_match = int((match_counts == 1).sum())
    n_multi_matches = int((match_counts > 1).sum())

    freq_dict = match_counts_series.value_counts().sort_index().to_dict()
    freq_dict_clean = {int(k): int(v) for k, v in freq_dict.items()}

    return {
        "total_source1_entities": n_rows,
        "singletons_count": n_singletons,
        "singletons_pct": float(round((n_singletons / n_rows) * 100, 2)) if n_rows > 0 else 0.0,
        "one_match_count": n_one_match,
        "one_match_pct": float(round((n_one_match / n_rows) * 100, 2)) if n_rows > 0 else 0.0,
        "multi_match_count": n_multi_matches,
        "multi_match_pct": float(round((n_multi_matches / n_rows) * 100, 2)) if n_rows > 0 else 0.0,
        "match_count_min": int(match_counts.min()) if n_rows > 0 else 0,
        "match_count_max": int(match_counts.max()) if n_rows > 0 else 0,
        "match_count_mean": float(round(match_counts.mean(), 2)) if n_rows > 0 else 0.0,
        "match_count_median": float(round(match_counts.median(), 2)) if n_rows > 0 else 0.0,
        "match_count_p95": float(round(np.percentile(match_counts, 95), 2)) if n_rows > 0 else 0.0,
        "match_frequency_distribution": freq_dict_clean,
    }


def find_noise_examples(train_gt: pd.DataFrame, train_s1: pd.DataFrame, train_s2: pd.DataFrame, train_s3: pd.DataFrame, sample_size: int = 2000) -> Dict[str, List[Dict[str, Any]]]:
    """
    Identifies representative real-world noise examples from ground truth matches.
    """
    s1_dict = train_s1.set_index("entity_id").to_dict(orient="index")
    s2_dict = train_s2.set_index("entity_id").to_dict(orient="index")
    s3_dict = train_s3.set_index("entity_id").to_dict(orient="index")

    def get_record(eid: str):
        if eid.startswith("S1-"):
            return s1_dict.get(eid)
        elif eid.startswith("S2-"):
            return s2_dict.get(eid)
        elif eid.startswith("S3-"):
            return s3_dict.get(eid)
        return None

    examples = {
        "abbreviations": [],
        "punctuation_case": [],
        "address_formatting": [],
        "multiple_matches": [],
        "singletons": [],
    }

    sample_gt = train_gt.head(sample_size)

    for _, row in sample_gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = str(row["matched_entity_ids"]).strip()
        s1_rec = get_record(s1_id)
        if not s1_rec:
            continue

        if not matched_str:
            if len(examples["singletons"]) < 5:
                examples["singletons"].append(
                    {
                        "source1_id": s1_id,
                        "business_name": s1_rec.get("business_name", ""),
                        "business_address": s1_rec.get("business_address", ""),
                        "country": s1_rec.get("country", ""),
                    }
                )
            continue

        matched_ids = [m.strip() for m in matched_str.split(",") if m.strip()]

        if len(matched_ids) > 1 and len(examples["multiple_matches"]) < 5:
            m_recs = [get_record(mid) for mid in matched_ids if get_record(mid) is not None]
            examples["multiple_matches"].append(
                {
                    "source1_id": s1_id,
                    "s1_name": s1_rec.get("business_name", ""),
                    "s1_address": s1_rec.get("business_address", ""),
                    "matches": [
                        {
                            "entity_id": mid,
                            "business_name": r.get("business_name", ""),
                            "business_address": r.get("business_address", ""),
                        }
                        for mid, r in zip(matched_ids, m_recs)
                    ],
                }
            )

        for mid in matched_ids:
            m_rec = get_record(mid)
            if not m_rec:
                continue

            name1 = str(s1_rec.get("business_name", ""))
            name2 = str(m_rec.get("business_name", ""))
            addr1 = str(s1_rec.get("business_address", ""))
            addr2 = str(m_rec.get("business_address", ""))

            # Abbreviation heuristic
            abbrev_tokens = ["corp", "corporation", "ltd", "limited", "pvt", "private", "inc", "st", "street", "rd", "road", "pvt.", "ltd."]
            n1_lower = name1.lower()
            n2_lower = name2.lower()
            if any(tok in n1_lower or tok in n2_lower for tok in abbrev_tokens) and n1_lower != n2_lower:
                if len(examples["abbreviations"]) < 5:
                    examples["abbreviations"].append(
                        {
                            "source1_id": s1_id,
                            "matched_id": mid,
                            "s1_name": name1,
                            "matched_name": name2,
                            "s1_address": addr1,
                            "matched_address": addr2,
                        }
                    )

            # Punctuation / Case
            punc = set(".,&-/()'\"")
            if (any(c in name1 for c in punc) != any(c in name2 for c in punc)) or (name1.isupper() != name2.isupper()):
                if len(examples["punctuation_case"]) < 5:
                    examples["punctuation_case"].append(
                        {
                            "source1_id": s1_id,
                            "matched_id": mid,
                            "s1_name": name1,
                            "matched_name": name2,
                        }
                    )

            # Address formatting
            if addr1.lower() != addr2.lower():
                if len(examples["address_formatting"]) < 5:
                    examples["address_formatting"].append(
                        {
                            "source1_id": s1_id,
                            "matched_id": mid,
                            "s1_address": addr1,
                            "matched_address": addr2,
                        }
                    )

    return examples
