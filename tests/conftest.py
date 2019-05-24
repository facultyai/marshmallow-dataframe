import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    df = pd.DataFrame(
        {
            "int": [1, 2],
            "boolean": [True, False],
            "string": ["a", "b"],
            "float": [1.1, 2.2],
            "datetime": ["2018-01-01T12:00:00Z", "2018-01-02T12:00:00Z"],
            "categorical_int": [0, 1],
            "categorical_str": ["zero", "one"],
        }
    )

    df["categorical_int"] = df["categorical_int"].astype("category")
    df["categorical_str"] = df["categorical_str"].astype("category")

    df["datetime"] = pd.to_datetime(df["datetime"]).astype("datetime64[ns]")

    return df
