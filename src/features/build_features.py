# Transform raw data into ML-ready features
import pandas as pd

def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    def _mode_or_unknown(s, default="unknown"):
        m = s.mode(dropna=True)
        return m.iat[0] if not m.empty else default

    agg = (
        df.groupby("userId")
          .agg(
              events=("page", "count"),
              unique_songs=("song", lambda s: s.dropna().nunique()),
              unique_artists=("artist", lambda s: s.dropna().nunique()),
              avg_song_len=("length", "mean"),
              last_ts=("ts", "max"),
              first_ts=("ts", "min"),
              plan_tier=("level", _mode_or_unknown),
              gender=("gender", _mode_or_unknown),
              location=("location", _mode_or_unknown),
          )
          .reset_index()
    )

    agg["tenure_days"] = (agg["last_ts"] - agg["first_ts"]).dt.days.clip(lower=0)

    global_max = agg["last_ts"].max()
    cutoff = global_max - pd.Timedelta(days=7)
    agg["churn"] = (agg["last_ts"] < cutoff).astype(int)

    return agg.fillna({"avg_song_len": 0})