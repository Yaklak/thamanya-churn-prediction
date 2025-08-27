from src.data.load import load_raw_events
import pandas as pd


def test_load_raw_events(raw_data_path):
    """Test the load_raw_events function."""
    df = load_raw_events(raw_data_path)

    # Expect 4 rows, as the function just reads the file
    assert len(df) == 4

    # Check columns
    expected_columns = [
        "ts",
        "userId",
        "sessionId",
        "page",
        "auth",
        "method",
        "status",
        "level",
        "itemInSession",
        "location",
        "userAgent",
        "lastName",
        "firstName",
        "registration",
        "gender",
        "song",
        "artist",
        "length",
    ]
    assert all(col in df.columns for col in expected_columns)

    # Check dtypes
    assert pd.api.types.is_integer_dtype(df["ts"])
    assert pd.api.types.is_string_dtype(df["userId"])
