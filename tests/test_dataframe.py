import numpy as np
import pandas as pd
import pytest
from marshmallow import ValidationError, fields
from dateutil.tz import tzutc
from pandas.util.testing import assert_frame_equal
from pandas.api.types import DatetimeTZDtype

from marshmallow_numerical import (
    Dtypes,
    SplitDataFrameSchema,
    RecordsDataFrameSchema,
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

    # pd.to_datetime defaults to 'UTC' timezone, whereas datetime objects
    # deserialized by marshmallow have a 'tzutc()' timezone. They are
    # functionally equivalent, but fail equality comparison, so here we set the
    # dtype to the one that marshmallow returns.
    df["datetime"] = pd.to_datetime(df["datetime"]).astype(
        DatetimeTZDtype("ns", tzutc())
    )

    return df


def serialize_df(df, orient="split"):
    test_df = df.copy()
    if "datetime" in test_df.columns:
        test_df["datetime"] = test_df["datetime"].astype(str)
    if orient == "records":
        return {"data": test_df.to_dict(orient="records")}
    elif orient == "split":
        return test_df.to_dict(orient="split")


def test_records_schema(sample_df):

    print(type(sample_df.dtypes))

    class MySchema(RecordsDataFrameSchema):
        dtypes = sample_df.dtypes

    schema = MySchema()

    output = schema.load(serialize_df(sample_df, orient="records"))

    assert_frame_equal(output, sample_df)


def test_split_schema(sample_df):
    class MySchema(SplitDataFrameSchema):
        dtypes = sample_df.dtypes
        index_dtype = sample_df.index.dtype

    schema = MySchema()

    output = schema.load(serialize_df(sample_df, orient="split"))

    assert_frame_equal(output, sample_df)


def test_records_schema_missing_column(sample_df):
    class MySchema(RecordsDataFrameSchema):
        dtypes = sample_df.dtypes

    schema = MySchema()

    input_df = serialize_df(sample_df, orient="records")

    # Remove float column from first record
    input_df["data"][0].pop("float")
    input_df["data"][1].pop("string")

    with pytest.raises(ValidationError) as exception:
        schema.load(input_df)

    assert "missing" in exception.value.messages["data"][0]["float"][0].lower()
    assert (
        "missing" in exception.value.messages["data"][1]["string"][0].lower()
    )


def test_records_schema_wrong_type(sample_df):
    class MySchema(RecordsDataFrameSchema):
        dtypes = sample_df.dtypes

    schema = MySchema()

    input_df = serialize_df(sample_df, orient="records")

    # Replace int with string
    input_df["data"][0]["int"] = "notint"

    with pytest.raises(ValidationError) as exception:
        schema.load(input_df)

    assert (
        "not a valid integer"
        in exception.value.messages["data"][0]["int"][0].lower()
    )


def test_records_schema_nulls():
    test_dtypes = pd.Series(index=["float"], data=[np.dtype(np.float)])

    class MySchema(RecordsDataFrameSchema):
        dtypes = test_dtypes

    schema = MySchema()

    test_df = [{"float": None}, {"float": 42.42}]
    expected_df = pd.DataFrame.from_dict(test_df).astype(test_dtypes)

    output = schema.load({"data": test_df})

    assert_frame_equal(output, expected_df)


@pytest.mark.parametrize(
    "input_df", [["this is a list of strings"], [1, 2, 3], np.array([1, 2, 3])]
)
def test_records_schema_invalid_input_type(input_df):
    class MySchema(RecordsDataFrameSchema):
        dtypes = Dtypes(columns=["float"], dtypes=[np.dtype(np.float)])

    schema = MySchema()

    with pytest.raises(ValidationError) as exception:
        schema.load({"data": input_df})
    assert (
        "invalid input type"
        in exception.value.messages["data"][0]["_schema"][0].lower()
    )


def test_records_schema_none():
    class MySchema(RecordsDataFrameSchema):
        dtypes = Dtypes(columns=["float"], dtypes=[np.dtype(np.float)])

    schema = MySchema()

    with pytest.raises(ValidationError, match="null"):
        schema.load({"data": None})


def test_records_schema_missing_data_field():
    class MySchema(RecordsDataFrameSchema):
        dtypes = Dtypes(columns=["float"], dtypes=[np.dtype(np.float)])

    schema = MySchema()

    with pytest.raises(ValidationError) as exception:
        schema.load({})

    assert (
        "missing data for required field"
        in exception.value.messages["data"][0].lower()
    )


@pytest.mark.parametrize(
    "input_data", ["", "string", 123, True, {"key": "value"}]
)
def test_dataframe_field_wrong_schema_notiter(dataframe_field, input_data):
    with pytest.raises(ValidationError, match="Invalid"):
        dataframe_field.deserialize(input_data)


def _serialize_df(input_df, orient):
    df = input_df.copy()
    # convert all datetimes to strings to enforce validation
    df["datetime"] = df["datetime"].astype(str)
    return df.to_dict(orient=orient)


def test_get_dataframe_schema_orient_records(sample_df):
    DataFrameSchema = get_dataframe_schema(sample_df, orient="records")
    schema = DataFrameSchema()

    assert schema.__class__.__name__ == "RequestRecordsDataFrameSchema"

    data_field = schema.fields["data"]

    assert isinstance(data_field, fields.Nested)
    assert isinstance(data_field.schema, BaseRecordsDataFrameSchema)

    result = schema.load({"data": _serialize_df(sample_df, orient="records")})

    assert_frame_equal(result, sample_df)


def test_get_dataframe_schema_orient_split(sample_df):
    DataFrameSchema = get_dataframe_schema(sample_df, orient="split")
    schema = DataFrameSchema()

    assert schema.__class__.__name__ == "RequestSplitDataFrameSchema"

    assert isinstance(schema.fields["data"].container, fields.Tuple)
    assert isinstance(schema.fields["columns"].container, fields.String)
    assert isinstance(schema.fields["index"].container, fields.Integer)

    df = sample_df.copy()
    df["datetime"] = df["datetime"].astype(str)
    result = schema.load(_serialize_df(sample_df, orient="split"))

    assert_frame_equal(result, sample_df)


def test_get_dataframe_schema_orient_split_str_index(sample_df):
    input_df = sample_df.copy()
    input_df.index = input_df.index.astype(str)

    DataFrameSchema = get_dataframe_schema(input_df, orient="split")
    schema = DataFrameSchema()

    assert isinstance(schema.fields["index"].container, fields.String)

    result = schema.load(_serialize_df(input_df, orient="split"))

    assert_frame_equal(result, input_df)


def test_get_dataframe_schema_orient_split_missing_column(sample_df):

    DataFrameSchema = get_dataframe_schema(sample_df, orient="split")
    schema = DataFrameSchema()

    serialized_df = _serialize_df(sample_df, orient="split")

    # delete one column name
    serialized_df["columns"].pop(0)

    with pytest.raises(ValidationError, match="Must be equal to") as exc:
        schema.load(serialized_df)

    assert (
        exc.value.messages["columns"][0]
        == f"Must be equal to {list(sample_df.columns)}."
    )


def test_get_dataframe_schema_orient_split_swapped_column(sample_df):

    DataFrameSchema = get_dataframe_schema(sample_df, orient="split")
    schema = DataFrameSchema()

    serialized_df = _serialize_df(sample_df, orient="split")

    # randomly permutate column names in list
    old_columns = serialized_df["columns"]
    serialized_df["columns"] = [
        old_columns[i] for i in np.random.permutation(len(old_columns))
    ]

    with pytest.raises(ValidationError, match="Must be equal to") as exc:
        schema.load(serialized_df)

    assert (
        exc.value.messages["columns"][0]
        == f"Must be equal to {list(sample_df.columns)}."
    )


@pytest.mark.parametrize(
    "base_class", [SplitDataFrameSchema, RecordsDataFrameSchema]
)
def test_schema_no_dtypes(base_class):
    class NewSchema(base_class):
        pass

    with pytest.raises(
        NotImplementedError, match="must define the `dtypes` attribute"
    ):
        NewSchema()


@pytest.mark.parametrize(
    "base_class", [SplitDataFrameSchema, RecordsDataFrameSchema]
)
def test_schema_wrong_dtypes(base_class):
    class NewSchema(base_class):
        dtypes = "wrong type for dtypes"

    with pytest.raises(
        ValueError, match="must be either a pandas DataFrame or"
    ):
        NewSchema()
