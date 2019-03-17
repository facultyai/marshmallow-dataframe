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
        # convert all datetimes to strings to enforce validation
        test_df["datetime"] = test_df["datetime"].astype(str)
    if orient == "records":
        return {"data": test_df.to_dict(orient="records")}
    elif orient == "split":
        return test_df.to_dict(orient="split")


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

    with pytest.raises(ValueError, match="must be either a pandas Series or"):
        NewSchema()


def test_records_schema(sample_df):
    class MySchema(RecordsDataFrameSchema):
        dtypes = sample_df.dtypes

    schema = MySchema()

    output = schema.load(serialize_df(sample_df, orient="records"))

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
    "input_data",
    [["this is a list of strings"], [1, 2, 3], np.array([1, 2, 3])],
)
def test_records_schema_invalid_input_type_iter(input_data):
    class MySchema(RecordsDataFrameSchema):
        dtypes = Dtypes(columns=["float"], dtypes=[np.dtype(np.float)])

    schema = MySchema()

    with pytest.raises(ValidationError) as exception:
        schema.load({"data": input_data})
    assert (
        "invalid input type"
        in exception.value.messages["data"][0]["_schema"][0].lower()
    )


@pytest.mark.parametrize(
    "input_data", ["", "string", 123, True, {"key": "value"}]
)
def test_records_schema_invalid_input_type_notiter(input_data):
    class MySchema(RecordsDataFrameSchema):
        dtypes = Dtypes(columns=["float"], dtypes=[np.dtype(np.float)])

    schema = MySchema()

    with pytest.raises(ValidationError):
        schema.load({"data": input_data})


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


@pytest.fixture
def split_sample_schema(sample_df):
    class MySchema(SplitDataFrameSchema):
        dtypes = sample_df.dtypes
        index_dtype = sample_df.index.dtype

    return MySchema()


@pytest.fixture
def split_serialized_df(sample_df):
    return serialize_df(sample_df, orient="split")


def test_split_schema(sample_df, split_sample_schema, split_serialized_df):

    assert isinstance(
        split_sample_schema.fields["data"].container, fields.Tuple
    )
    assert isinstance(
        split_sample_schema.fields["columns"].container, fields.String
    )
    assert isinstance(
        split_sample_schema.fields["index"].container, fields.Integer
    )

    output = split_sample_schema.load(split_serialized_df)

    assert_frame_equal(output, sample_df)


@pytest.mark.parametrize("key", ["data", "index", "columns"])
def test_split_schema_missing_top_level_key(
    key, split_sample_schema, split_serialized_df
):
    del split_serialized_df[key]

    with pytest.raises(ValidationError) as exc:
        split_sample_schema.load(split_serialized_df)

    assert exc.value.messages[key] == ["Missing data for required field."]


def test_split_schema_str_index(sample_df):
    test_df = sample_df.copy()
    test_df.index = test_df.index.astype(str)

    class MySchema(SplitDataFrameSchema):
        dtypes = test_df.dtypes
        index_dtype = test_df.index.dtype

    schema = MySchema()

    assert isinstance(schema.fields["index"].container, fields.String)

    result = schema.load(serialize_df(test_df, orient="split"))

    assert_frame_equal(result, test_df)


def test_split_schema_missing_column(
    sample_df, split_sample_schema, split_serialized_df
):
    # delete one column name
    split_serialized_df["columns"].pop(0)

    with pytest.raises(ValidationError) as exc:
        split_sample_schema.load(split_serialized_df)

    assert (
        exc.value.messages["columns"][0]
        == f"Must be equal to {list(sample_df.columns)}."
    )


def test_split_schema_swapped_column(
    sample_df, split_sample_schema, split_serialized_df
):
    # randomly permutate column names in list
    old_columns = split_serialized_df["columns"]
    split_serialized_df["columns"] = [
        old_columns[i] for i in np.random.permutation(len(old_columns))
    ]

    with pytest.raises(ValidationError) as exc:
        split_sample_schema.load(split_serialized_df)

    assert (
        exc.value.messages["columns"][0]
        == f"Must be equal to {list(sample_df.columns)}."
    )


def test_split_schema_wrong_row_length(
    sample_df, split_sample_schema, split_serialized_df
):
    # delete an item from data
    del split_serialized_df["data"][0][-1]

    with pytest.raises(ValidationError) as exc:
        split_sample_schema.load(split_serialized_df)

    assert (
        exc.value.messages["data"][0][0]
        == f"Length must be {len(sample_df.columns)}."
    )


def test_split_schema_wrong_type_in_data(
    split_sample_schema, split_serialized_df
):
    # set an item from int column to a non-int value
    split_serialized_df["data"][0][0] = "notanint"

    with pytest.raises(ValidationError) as exc:
        split_sample_schema.load(split_serialized_df)

    assert exc.value.messages["data"][0][0][0] == f"Not a valid integer."


@pytest.mark.parametrize("key", ["index", "data"])
def test_split_schema_index_data_length_mismatch(
    key, split_sample_schema, split_serialized_df
):
    # set an item from int column to a non-int value
    split_serialized_df[key].pop(0)

    with pytest.raises(ValidationError) as exc:
        split_sample_schema.load(split_serialized_df)

    assert (
        exc.value.messages["data"][0]
        == f"Length of `index` and `data` must be equal."
    )


# TODO on both schemas:
# setup hypothesis testing
