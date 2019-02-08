import numpy as np
import pandas as pd
import pytest
from marshmallow import ValidationError, fields
from pandas.util.testing import assert_frame_equal

from marshmallow_numerical.dataframe import (
    get_dataframe_schema,
    _create_records_data_field_from_dataframe,
    BaseRecordsDataFrameSchema,
)


@pytest.fixture
def sample_df():
    df = pd.DataFrame(
        {
            "int": [1, 2],
            "boolean": [True, False],
            "string": ["a", "b"],
            "float": [1.1, 2.2],
            "datetime": ["2018-01-01T12:00:00Z", "2018-01-02T12:00:00Z"],
        }
    )

    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


@pytest.fixture
def dataframe_field(sample_df):
    test_df_field = _create_records_data_field_from_dataframe(sample_df)
    return test_df_field


def test_dataframe_field(sample_df):
    sample_df = sample_df.copy()
    sample_df["datetime"] = sample_df["datetime"].astype(str)
    serialized_df = sample_df.to_dict(orient="records")

    records_field = _create_records_data_field_from_dataframe(sample_df)
    output = records_field.deserialize(serialized_df)

    assert_frame_equal(output, sample_df)


def test_dataframe_field_missing_column(dataframe_field):
    input_df = [
        {
            "int": 1,
            "boolean": True,
            "string": "a",
            "datetime": "2018-01-01T12:00:00Z",
        }
    ]
    with pytest.raises(ValidationError) as exception:
        dataframe_field.deserialize(input_df)
    assert "missing" in exception.value.messages[0]["float"][0].lower()


def test_dataframe_field_two_missing_column(dataframe_field):
    input_df = [
        {"int": 1, "boolean": True, "datetime": "2018-01-01T12:00:00Z"}
    ]
    with pytest.raises(ValidationError) as exception:
        dataframe_field.deserialize(input_df)
    assert "missing" in exception.value.messages[0]["float"][0].lower()
    assert "missing" in exception.value.messages[0]["string"][0].lower()


def test_dataframe_field_wrong_type(dataframe_field):
    input_df = [
        {
            "int": "notint",
            "boolean": True,
            "string": "a",
            "float": 1.1,
            "datetime": "2018-01-01T12:00:00Z",
        }
    ]
    with pytest.raises(ValidationError) as exception:
        dataframe_field.deserialize(input_df)
    assert (
        "not a valid integer" in exception.value.messages[0]["int"][0].lower()
    )


def test_dataframe_field_type_none():
    sample_df = pd.DataFrame({"float": [10.2, 10.0]})
    test_df_field = _create_records_data_field_from_dataframe(sample_df)

    serialized_df = [{"float": None}, {"float": 42.42}]
    expected_df = pd.DataFrame.from_dict(serialized_df).astype(
        sample_df.dtypes
    )

    output = test_df_field.deserialize(serialized_df)
    assert_frame_equal(output.isnull(), expected_df.isnull(), check_like=True)


@pytest.mark.parametrize(
    "input_df", [["this is a list of strings"], [1, 2, 3], np.array([1, 2, 3])]
)
def test_dataframe_field_wrong_schema_iter(dataframe_field, input_df):
    with pytest.raises(ValidationError) as exception:
        dataframe_field.deserialize(input_df)
    assert "invalid" in exception.value.messages[0]["_schema"][0].lower()


def test_dataframe_field_wrong_schema_none(dataframe_field):
    with pytest.raises(ValidationError, match="null"):
        dataframe_field.deserialize(None)


@pytest.mark.parametrize(
    "input_data", ["", "string", 123, True, {"key": "value"}]
)
def test_dataframe_field_wrong_schema_notiter(dataframe_field, input_data):
    with pytest.raises(ValidationError, match="Invalid"):
        dataframe_field.deserialize(input_data)


def test_get_dataframe_schema_orient_records(sample_df):
    DataFrameSchema = get_dataframe_schema(sample_df, orient="records")
    schema = DataFrameSchema()

    assert schema.__class__.__name__ == "RequestRecordsDataFrameSchema"

    data_field = schema.fields["data"]

    assert isinstance(data_field, fields.Nested)
    assert isinstance(data_field.schema, BaseRecordsDataFrameSchema)

    serialized_df = sample_df.copy()
    serialized_df["datetime"] = serialized_df["datetime"].astype(str)
    result = schema.load({"data": serialized_df.to_dict(orient="records")})

    assert_frame_equal(result, sample_df)


def test_get_dataframe_schema_orient_split(sample_df):
    DataFrameSchema = get_dataframe_schema(sample_df, orient="split")
    schema = DataFrameSchema()

    assert schema.__class__.__name__ == "RequestSplitDataFrameSchema"

    assert isinstance(schema.fields["data"].container, fields.Tuple)
    assert isinstance(schema.fields["columns"].container, fields.String)
    assert isinstance(schema.fields["index"].container, fields.Int)

    serialized_df = sample_df.copy()
    serialized_df["datetime"] = serialized_df["datetime"].astype(str)
    result = schema.load(serialized_df.to_dict(orient="split"))

    assert_frame_equal(result, sample_df)
