import hypothesis
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis.extra.pandas import column, data_frames, indexes
from marshmallow import ValidationError
from marshmallow_dataframe import Dtypes, RecordsDataFrameSchema
from pandas.util.testing import assert_frame_equal

from .utils import serialize_df


def test_schema_no_dtypes():
    with pytest.raises(
        ValueError, match="must define the `dtypes` Meta option"
    ):

        class NewSchema(RecordsDataFrameSchema):
            pass


def test_schema_wrong_dtypes():
    with pytest.raises(ValueError, match="must be either a pandas Series or"):

        class NewSchema(RecordsDataFrameSchema):
            class Meta:
                dtypes = "wrong type for dtypes"


def test_records_schema(sample_df):
    class MySchema(RecordsDataFrameSchema):
        class Meta:
            dtypes = sample_df.dtypes

    schema = MySchema()

    print(schema._declared_fields)

    output = schema.load(serialize_df(sample_df, orient="records"))

    assert_frame_equal(output, sample_df)


@hypothesis.given(
    test_df=data_frames(
        columns=[
            column("int", dtype=int),
            column("float", dtype=float),
            column("bool", dtype=bool),
            column("chars", elements=st.characters()),
            column(
                "datetime",
                elements=st.datetimes(
                    min_value=pd.Timestamp.min, max_value=pd.Timestamp.max
                ),
                dtype="datetime64[s]",
            ),
        ],
        # records serialization format does not record indices, so we always
        # set them to an integer index.
        index=(
            indexes(
                elements=st.integers(
                    min_value=np.iinfo(np.int64).min,
                    max_value=np.iinfo(np.int64).max,
                )
            )
        ),
    )
)
def test_records_schema_hypothesis(test_df):

    if not len(test_df.index):
        # ignore empty datasets as dtype is impossible to infer from serialized
        return

    class MySchema(RecordsDataFrameSchema):
        class Meta:
            dtypes = test_df.dtypes

    schema = MySchema()

    output = schema.load(serialize_df(test_df, orient="records"))

    # Ignore indices in the test as the records serialization format does not
    # support them
    output.index = test_df.index

    assert_frame_equal(output, test_df)


def test_records_schema_missing_column(sample_df):
    class MySchema(RecordsDataFrameSchema):
        class Meta:
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
        class Meta:
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
        class Meta:
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
        class Meta:
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
        class Meta:
            dtypes = Dtypes(columns=["float"], dtypes=[np.dtype(np.float)])

    schema = MySchema()

    with pytest.raises(ValidationError):
        schema.load({"data": input_data})


def test_records_schema_none():
    class MySchema(RecordsDataFrameSchema):
        class Meta:
            dtypes = Dtypes(columns=["float"], dtypes=[np.dtype(np.float)])

    schema = MySchema()

    with pytest.raises(ValidationError, match="null"):
        schema.load({"data": None})


def test_records_schema_missing_data_field():
    class MySchema(RecordsDataFrameSchema):
        class Meta:
            dtypes = Dtypes(columns=["float"], dtypes=[np.dtype(np.float)])

    schema = MySchema()

    with pytest.raises(ValidationError) as exception:
        schema.load({})

    assert (
        "missing data for required field"
        in exception.value.messages["data"][0].lower()
    )


@pytest.mark.parametrize(
    "column,data",
    (["categorical_int", [0, 10]], ["categorical_str", ["one", "ten"]]),
)
def test_records_schema_wrong_category(sample_df, column, data):
    class MySchema(RecordsDataFrameSchema):
        class Meta:
            dtypes = sample_df.dtypes

    schema = MySchema()

    # Replace categorical column data
    new_df = sample_df.copy()
    new_df[column] = data

    input_df = serialize_df(new_df, orient="records")

    with pytest.raises(ValidationError) as exception:
        schema.load(input_df)

    assert exception.value.messages["data"][1][column][0].startswith(
        "Must be one of"
    )
