from src.features.build_features import _parse_os  # <-- update if needed


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
