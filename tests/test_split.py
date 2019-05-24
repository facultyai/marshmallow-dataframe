import hypothesis
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis.extra.pandas import column, data_frames, indexes
from marshmallow import ValidationError, fields
from marshmallow_dataframe import SplitDataFrameSchema
from pandas.util.testing import assert_frame_equal

from .utils import serialize_df


def test_schema_no_dtypes():
    with pytest.raises(
        ValueError, match="must define the `dtypes` Meta option"
    ):

        class NewSchema(SplitDataFrameSchema):
            pass


def test_schema_wrong_dtypes():
    with pytest.raises(ValueError, match="must be either a pandas Series or"):

        class NewSchema(SplitDataFrameSchema):
            class Meta:
                dtypes = "wrong type for dtypes"


@pytest.fixture
def split_sample_schema(sample_df):
    class MySchema(SplitDataFrameSchema):
        class Meta:
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
        index=(
            indexes(
                elements=st.integers(
                    min_value=np.iinfo(np.int64).min,
                    max_value=np.iinfo(np.int64).max,
                )
            )
            | indexes(elements=st.characters())
        ),
    )
)
def test_split_schema_hypothesis(test_df):

    if not len(test_df.index):
        # ignore empty datasets as dtype is impossible to infer from serialized
        return

    class MySchema(SplitDataFrameSchema):
        class Meta:
            dtypes = test_df.dtypes
            index_dtype = test_df.index.dtype

    schema = MySchema()

    output = schema.load(serialize_df(test_df, orient="split"))

    assert_frame_equal(output, test_df)


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
        class Meta:
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
    # reverse order of columns so it does not match schema
    split_serialized_df["columns"] = split_serialized_df["columns"][::-1]

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


@pytest.mark.parametrize(
    "column,data",
    (["categorical_int", [0, 10]], ["categorical_str", ["one", "ten"]]),
)
def test_split_schema_wrong_category(
    split_sample_schema, sample_df, column, data
):
    # Replace categorical column data
    new_df = sample_df.copy()
    new_df[column] = data

    column_index = list(new_df.columns).index(column)

    input_df = serialize_df(new_df, orient="split")

    with pytest.raises(ValidationError) as exception:
        split_sample_schema.load(input_df)

    assert exception.value.messages["data"][1][column_index][0].startswith(
        "Must be one of"
    )
