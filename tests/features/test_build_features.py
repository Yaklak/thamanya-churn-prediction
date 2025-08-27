from src.features.build_features import (
    _parse_os,
    build_user_features,
)  # <-- update if needed


def test_parse_os_ordering_and_defaults():
    samples = {
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)": "ios",
        "Mozilla/5.0 (iPad; CPU OS 13_3 like Mac OS X)": "ios",
        "Mozilla/5.0 (Linux; Android 10; SM-G975F)": "android",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)": "windows",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)": "mac",
        "Mozilla/5.0 (X11; Linux x86_64)": "linux",
        "Totally Unknown UA": "other",
        None: "unknown",
    }

    for ua, expected in samples.items():
        # function lowercases internally; this tests mixed casing paths too
        assert _parse_os(ua) == expected


def test_build_user_features(raw_events_df):
    """Test the build_user_features function."""
    features_df = build_user_features(raw_events_df, inactivity_days=30, target="churn")

    # Expect two users
    assert len(features_df) == 2

    # Check features for user '1'
    user1_features = features_df.query("userId == '1'").iloc[0]
    assert user1_features["events"] == 3
    assert user1_features["sessions"] == 1
    assert user1_features["days_active"] == 1
    assert user1_features["songs_played"] == 1
    assert user1_features["unique_songs"] == 1
    assert user1_features["total_song_length"] == 240.0

    # Check features for user '2'
    user2_features = features_df.query("userId == '2'").iloc[0]
    assert user2_features["events"] == 1
    assert user2_features["sessions"] == 1
    assert user2_features["days_active"] == 1
    assert user2_features["songs_played"] == 0

    # Check churn label
    # User 1 was last active recently, so should not be churned
    # User 2 was last active a long time ago, so should be churned
    assert user1_features["churn"] == 0
    assert user2_features["churn"] == 1
