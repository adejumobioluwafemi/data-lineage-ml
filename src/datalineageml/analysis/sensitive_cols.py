"""
src/datalineageml/analysis/sensitive_cols.py

Automatic sensitive column discovery for DataLineageML.

Answers: "Which columns in this DataFrame are likely demographic
attributes I should be tracking for fairness?"

Nobody wants to manually specify sensitive_cols=["gender","ethnicity",
"age_group","region","religion","lga"] on every @track decorator.
This module removes that friction.

Three heuristics run in combination:

1. Name matching — column names that match known demographic keywords
   in English, plus common Nigerian/West African administrative terms
   (LGA, geopolitical zone, tribe, state of origin).

2. Cardinality check — low-cardinality categorical columns (≤ 20 unique
   non-null values) are candidates. High-cardinality columns (free text,
   IDs) are excluded. Numeric columns with many unique values are excluded.

3. Distribution check — columns where the value distribution is uneven
   (some values appear much more than others, consistent with demographic
   stratification) are ranked higher.

Usage:
    from datalineageml.analysis.sensitive_cols import discover_sensitive_cols

    candidates = discover_sensitive_cols(df)
    print(candidates)
    # [SensitiveColCandidate(column='gender', confidence=0.97, reason='name match'),
    #  SensitiveColCandidate(column='zone', confidence=0.82, reason='name match + low cardinality'),
    #  SensitiveColCandidate(column='age_group', confidence=0.91, reason='name match')]

    # Get just the column names above a confidence threshold:
    cols = [c.column for c in candidates if c.confidence >= 0.7]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Keyword dictionaries

# Primary demographic keywords — very high confidence if matched
_PRIMARY_KEYWORDS = {
    "gender", "sex", "race", "ethnicity", "religion", "caste",
    "disability", "nationality", "citizenship", "marital_status",
    "pregnancy", "sexual_orientation",
}

# Secondary demographic keywords — high confidence
_SECONDARY_KEYWORDS = {
    "age", "age_group", "age_band", "agegroup", "ageband",
    "income_group", "income_band", "income_bracket",
    "education", "edu_level", "education_level",
    "occupation", "employment", "job_type",
    "tribe", "ethnic_group", "ethnicity_group",
    "zone", "geopolitical_zone", "geo_zone", "region", "region_group",
    "lga", "local_government", "local_govt", "local_area",
    "state", "state_of_origin", "origin_state",
    "urban_rural", "urbanrural", "residence_type",
    "household_type", "hh_type",
    "language", "mother_tongue",
    "land_tenure", "land_type", "ownership_type",
    "farmer_type", "farm_type",
}

# Partial match patterns — medium confidence if substring matches
_PARTIAL_PATTERNS = [
    r"gender", r"sex$", r"^sex_",
    r"race$", r"^race_", r"ethnic",
    r"relig", r"caste",
    r"age_?grp", r"age_?band", r"age_?cat", r"age_?group",
    r"income_?grp", r"income_?band", r"income_?cat",
    r"edu", r"school",
    r"zone", r"region", r"district", r"lga",
    r"tribe", r"tribe_?grp",
    r"disab", r"impair",
    r"marital", r"married",
    r"rural", r"urban",
    r"origin", r"native",
    r"farm_?type", r"land_?type", r"tenure",
]

_PARTIAL_RE = re.compile(
    "|".join(f"(?:{p})" for p in _PARTIAL_PATTERNS),
    re.IGNORECASE,
)


@dataclass
class SensitiveColCandidate:
    """A column identified as a candidate sensitive attribute.

    Attributes:
        column:     Column name in the DataFrame.
        confidence: Score in [0.0, 1.0]. Higher = more likely sensitive.
        reasons:    List of reasons for the score.
        n_unique:   Number of unique non-null values in the column.
        dtype:      pandas dtype string.
        top_values: Up to 5 most common values (for review).
    """
    column:     str
    confidence: float
    reasons:    List[str]
    n_unique:   int
    dtype:      str
    top_values: List[Any]

    def __repr__(self):
        reasons_str = ", ".join(self.reasons)
        return (f"SensitiveColCandidate(column={self.column!r}, "
                f"confidence={self.confidence:.2f}, "
                f"reasons=[{reasons_str}], "
                f"n_unique={self.n_unique})")


def discover_sensitive_cols(
    df,
    min_confidence: float = 0.5,
    max_unique:     int   = 20,
    include_numeric: bool = False,
) -> List[SensitiveColCandidate]:
    """Discover candidate sensitive / demographic columns in a DataFrame.

    Runs three heuristics — name matching, cardinality check, distribution
    check — and returns candidates sorted by confidence descending.

    Args:
        df:              pandas DataFrame to analyse.
        min_confidence:  Minimum confidence to include in results. Default 0.5.
                         Set to 0.0 to see all candidates.
        max_unique:      Maximum number of unique values for a column to be
                         considered categorical. Columns with more unique values
                         than this are excluded unless they match a primary
                         keyword. Default: 20.
        include_numeric: If True, include numeric columns that match keyword
                         patterns (e.g. age as a raw integer). Default: False.

    Returns:
        List of ``SensitiveColCandidate``, sorted by confidence descending.

    Example::

        from datalineageml.analysis.sensitive_cols import discover_sensitive_cols

        candidates = discover_sensitive_cols(df, min_confidence=0.7)
        sensitive_cols = [c.column for c in candidates]
        # Pass directly to @track:
        # @track(name="clean", snapshot=True, sensitive_cols=sensitive_cols)
    """
    candidates = []

    for col in df.columns:
        col_lower   = col.lower().strip()
        series      = df[col].dropna()
        n_unique    = int(series.nunique())
        dtype_str   = str(df[col].dtype)
        is_numeric  = dtype_str.startswith(("int", "float"))
        top_vals    = list(series.value_counts().head(5).index)

        if is_numeric and not include_numeric:
            # Only include numeric if it matches a primary keyword
            if col_lower not in _PRIMARY_KEYWORDS and not _is_primary_partial(col_lower):
                continue

        score   = 0.0
        reasons = []

        # Heuristic 1: Name matching
        if col_lower in _PRIMARY_KEYWORDS:
            score += 0.80
            reasons.append("primary keyword match")
        elif col_lower in _SECONDARY_KEYWORDS:
            score += 0.55
            reasons.append("secondary keyword match")
        elif _PARTIAL_RE.search(col_lower):
            score += 0.40
            reasons.append("partial name match")

        # Heuristic 2: Cardinality
        if n_unique == 0:
            continue  # empty column — skip entirely

        if 2 <= n_unique <= 5:
            score += 0.20
            reasons.append(f"low cardinality (n_unique={n_unique})")
        elif 6 <= n_unique <= max_unique:
            score += 0.10
            reasons.append(f"medium cardinality (n_unique={n_unique})")
        elif n_unique > max_unique and col_lower not in _PRIMARY_KEYWORDS:
            score -= 0.20   # high cardinality penalty unless primary keyword

        # Heuristic 3: Distribution unevenness
        # Demographic columns often have uneven distributions
        # (e.g. 60% M / 40% F rather than uniform).
        # We measure this only for low-cardinality non-numeric columns.
        if not is_numeric and 2 <= n_unique <= max_unique and len(series) > 0:
            try:
                vc     = series.value_counts(normalize=True)
                max_f  = float(vc.max())
                min_f  = float(vc.min())
                spread = max_f - min_f
                if spread > 0.3:
                    score += 0.10
                    reasons.append(f"uneven distribution (spread={spread:.2f})")
            except Exception:
                pass

        # Final score clamp and filter
        score = min(1.0, max(0.0, round(score, 3)))

        if score >= min_confidence:
            candidates.append(SensitiveColCandidate(
                column     = col,
                confidence = score,
                reasons    = reasons,
                n_unique   = n_unique,
                dtype      = dtype_str,
                top_values = top_vals,
            ))

    candidates.sort(key=lambda c: -c.confidence)
    return candidates


def suggest_sensitive_cols(df, min_confidence: float = 0.6) -> List[str]:
    """Return just the column names above the confidence threshold.

    Convenience wrapper around ``discover_sensitive_cols`` for direct use
    in ``@track`` decorators.

    Args:
        df:             pandas DataFrame.
        min_confidence: Confidence threshold. Default: 0.6.

    Returns:
        List of column name strings.

    Example::

        from datalineageml.analysis.sensitive_cols import suggest_sensitive_cols

        cols = suggest_sensitive_cols(df)
        # cols = ["gender", "zone", "age_group"]

        @track(name="clean", snapshot=True, sensitive_cols=cols)
        def clean(df): return df.dropna()
    """
    return [c.column for c in discover_sensitive_cols(df, min_confidence)]


def print_sensitive_col_report(df, min_confidence: float = 0.5) -> None:
    """Print a human-readable sensitive column discovery report.

    Example output::

        ── Sensitive column discovery ──────────────────────────────────
          Column          Conf   Unique  Dtype   Top values
          ──────────────  ─────  ──────  ──────  ─────────────────────
          gender          0.97      2    object  F, M
          zone            0.82      6    object  SW, SE, NC, SS, NW, NE
          age_group       0.91      4    object  25-34, 35-44, 18-24, 45+
          land_title      0.51      2    object  registered, unregistered

        Suggested: sensitive_cols=["gender", "zone", "age_group"]
    """
    w = 65
    candidates = discover_sensitive_cols(df, min_confidence)

    print(f"\n{'─' * w}")
    print(f"  Sensitive column discovery")
    print(f"{'─' * w}")
    if not candidates:
        print(f"  No candidate sensitive columns found above "
              f"confidence {min_confidence:.0%}.")
        print(f"  Try lowering min_confidence or check column names.")
        print(f"{'─' * w}\n")
        return

    print(f"  {'Column':20s}  {'Conf':>5}  {'Unique':>6}  {'Dtype':>8}  Top values")
    print(f"  {'─'*20}  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*24}")
    for c in candidates:
        top = ", ".join(str(v)[:8] for v in c.top_values[:4])
        if len(c.top_values) > 4:
            top += f" (+{len(c.top_values)-4})"
        print(f"  {c.column:20s}  {c.confidence:>5.2f}  {c.n_unique:>6}  "
              f"{c.dtype:>8}  {top}")

    suggested = [c.column for c in candidates if c.confidence >= 0.6]
    if suggested:
        cols_str = ", ".join(f'"{c}"' for c in suggested)
        print(f"\n  Suggested: sensitive_cols=[{cols_str}]")
    print(f"{'─' * w}\n")


def _is_primary_partial(col_lower: str) -> bool:
    """Check if a column name partially matches a primary demographic keyword."""
    primary_patterns = [r"gender", r"^sex$", r"^sex_", r"race", r"ethnic",
                        r"relig", r"disab", r"age_?grp", r"age_?band"]
    return any(re.search(p, col_lower) for p in primary_patterns)