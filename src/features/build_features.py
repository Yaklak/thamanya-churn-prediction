# Transform raw data into ML-ready features
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.io_utils import save_csv


def _last_non_null(s: pd.Series) -> object:
    # last observed non-null value (stable for 'level', 'gender')
    return s.loc[s.last_valid_index()] if s.notna().any() else np.nan


def _parse_os(ua: object) -> str:
    """
    Classify OS from a user-agent string.

    ORDER MATTERS:
    Some UA strings contain multiple keywords. For example, iPhone UAs often
    include "Mac OS X" as a substring. We must check for iPhone/iPad *before*
    Mac OS to avoid misclassifying iOS devices as macOS.
    """
    if not isinstance(ua, str):
        return "unknown"

    s = ua.strip().lower()
    if "iphone" in s or "ipad" in s:  # iOS first: these often include "mac os x" too
        return "ios"
    elif "android" in s:
        return "android"
    elif "windows" in s:
        return "windows"
    elif "macintosh" in s or "mac os" in s:
        return "mac"
    elif "linux" in s:
        return "linux"
    else:
        return "other"


def build_user_features(
    df: pd.DataFrame, inactivity_days: int, target: str
) -> pd.DataFrame:
    """Aggregate raw event logs into per-user features + churn label."""
    df = df.copy()

    # global "now" = last event in the log
    if "ts" not in df.columns:
        raise ValueError("Expected 'ts' in raw data.")
    global_last_ts = df["ts"].max()

    # --- per-user aggregates ---
    if "userId" not in df.columns:
        raise ValueError("Expected 'userId' in raw data.")
    g = df.groupby("userId", dropna=True)

    # Counts & time features
    if {"sessionId", "registration", "event_date"}.issubset(df.columns):
        feats_index = g.size().index.astype(str)
        feats = pd.DataFrame(index=feats_index)
        feats.index.name = "userId"
        feats["events"] = g.size()
        feats["sessions"] = g["sessionId"].nunique()
        feats["first_ts"] = g["ts"].min()
        feats["last_ts"] = g["ts"].max()
        feats["registration"] = g["registration"].min()
        feats["days_active"] = g["event_date"].nunique()
        feats["tenure_days"] = (feats["last_ts"] - feats["registration"]).dt.days
        feats["recency_days"] = (global_last_ts - feats["last_ts"]).dt.days

    # Song-based (NextSong only) features
    if {"page", "song", "artist", "length"}.issubset(df.columns):
        song_df = df.loc[df["page"] == "nextsong"].copy()
        sg = song_df.groupby("userId")
        song_feats = pd.DataFrame(
            {
                "songs_played": sg.size(),
                "unique_songs": sg["song"].nunique(),
                "unique_artists": sg["artist"].nunique(),
                "total_song_length": sg["length"].sum(),
                "avg_song_len": sg["length"].mean(),
            }
        )
        feats = feats.join(song_feats, how="left")

    # HTTP status profile features
    if {"page", "status"}.issubset(df.columns):
        status_pivot = (
            df.pivot_table(
                index="userId",
                columns="status",
                values="page",
                aggfunc="count",
                fill_value=0,
            )
            .add_prefix("status_")
            .astype("int64")
        )
        feats = feats.join(status_pivot, how="left").fillna(0)
        feats["status_200"] = feats.get("status_200", 0)
        feats["error_events"] = (feats["events"] - feats["status_200"]).clip(lower=0)
        feats["error_rate"] = np.where(
            feats["events"] > 0, feats["error_events"] / feats["events"], 0.0
        )

    # gender usage features (ratios)
    if "gender" in df.columns:
        gender_counts = df.pivot_table(
            index="userId",
            columns="gender",
            values="ts",
            aggfunc="count",
            fill_value=0,
        ).add_prefix("gender_usage_")

        # compute ratios in a single batched operation to avoid fragmentation
        total_gender_usage = gender_counts.sum(axis=1).replace(0, np.nan)
        gender_ratios = gender_counts.div(total_gender_usage, axis=0).add_suffix(
            "_ratio"
        )

        feats = pd.concat([feats, gender_ratios], axis=1).fillna(0)

    # Level-based usage features (paid vs free event ratios)
    if "level" in df.columns:
        level_counts = df.pivot_table(
            index="userId", columns="level", values="ts", aggfunc="count", fill_value=0
        ).add_prefix("level_")
        feats = feats.join(level_counts, how="left").fillna(0)

        if "level_paid" in feats.columns:
            total_level = feats.filter(like="level_").sum(axis=1)
            feats["paid_ratio"] = feats["level_paid"] / (total_level)

    # device OS usage features (counts)
    if "userAgent" in df.columns and "events" in feats.columns:
        df["os_usage"] = df["userAgent"].apply(_parse_os)
        os_counts = df.pivot_table(
            index="userId",
            columns="os_usage",
            values="ts",
            aggfunc="count",
            fill_value=0,
        ).add_prefix("os_usage_")

        # compute OS usage ratios across ALL user events (batched to avoid fragmentation)
        os_ratios = os_counts.div(feats["events"].replace(0, np.nan), axis=0)
        feats = pd.concat([feats, os_ratios.add_suffix("_ratio")], axis=1)

    # state usage features (counts)
    # def _parse_state(loc: object) -> str:
    #     """Extract state/region code from 'City, ST' style location strings."""
    #     s = str(loc or "").strip()
    #     if "," not in s:
    #         return "NA"
    #     return s.split(",")[-1].strip() or "NA"

    # if "location" in df.columns:
    #     df["_state"] = df["location"].map(_parse_state)
    #     state_counts = df.pivot_table(
    #         index="userId",
    #         columns="_state",
    #         values="ts",
    #         aggfunc="count",
    #         fill_value=0,
    #     ).add_prefix("state_usage_")

    #     # compute ratios per state relative to all events (batched)
    #     total_events = state_counts.sum(axis=1).replace(0, np.nan)
    #     state_ratios = state_counts.div(total_events, axis=0)
    #     feats = pd.concat([feats, state_ratios], axis=1)

    # page‑level success & failed ratios (status==200 vs failed=total-success)
    if {"page", "status"}.issubset(df.columns) and "events" in feats.columns:
        page_events = [
            (p, p.strip().lower().replace(" ", "_"))
            for p in df["page"].dropna().unique()
        ]

        to_concat = {}
        events_denom = feats["events"].replace(0, np.nan)  # guard against div-by-zero

        for page, col in page_events:
            # compute success / total / failed for this page
            m_total = df["page"].eq(page)
            total = df.loc[m_total].groupby("userId").size()

            m_success = m_total & df["status"].eq(200)
            success = df.loc[m_success].groupby("userId").size()

            failed = total.sub(success, fill_value=0)

            to_concat[f"{col}_success_ratio"] = success / events_denom
            to_concat[f"{col}_failed_ratio"] = failed / events_denom

        if to_concat:
            feats = pd.concat([feats, pd.DataFrame(to_concat)], axis=1)
            feats = feats.copy()  # defragment blocks to avoid PerformanceWarning

    # Fill NaNs produced by users with no such event
    feats = feats.fillna(0)

    # --- label: explicit OR inactivity (N days) ---
    # explicit = (
    #     df[(df["page"] == "cancellation confirmation") & (df["status"] == 200)]
    #     .groupby("userId")
    #     .size()
    #     .rename("explicit_churn")
    # )
    # feats = feats.join(explicit, how="left").fillna({"explicit_churn": 0})
    # feats["explicit_churn"] = (feats["explicit_churn"] > 0).astype(int)

    feats[target] = (
        (global_last_ts - feats["last_ts"]).dt.days >= int(inactivity_days)
    ).astype(int)
    # feats[target] = np.maximum(feats["explicit_churn"], inactivity)

    out = feats.reset_index()  # `userId` becomes a regular column
    save_csv(out, "customer_churn_engineered", "data/processed")
    return out
