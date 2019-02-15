import pandas as pd
import numpy as np
from marshmallow import fields, Schema, post_load, validate

from .base import BaseSchema

_FIELD_OPTIONS = {"required": True, "allow_none": True}

# Integer columns in pandas cannot have null values, so here we allow_none for
# all types except int
DTYPE_KIND_TO_FIELD = {
    "i": fields.Int(required=True),
    "u": fields.Int(**_FIELD_OPTIONS),
    "f": fields.Float(**_FIELD_OPTIONS),
    "O": fields.Str(**_FIELD_OPTIONS),
    "b": fields.Bool(**_FIELD_OPTIONS),
    "M": fields.DateTime(**_FIELD_OPTIONS),
    "m": fields.TimeDelta(**_FIELD_OPTIONS),
}


class DtypeToFieldConversionoError(Exception):
    pass


def _dtype_to_field(dtype):
    try:
        return DTYPE_KIND_TO_FIELD[dtype.kind]
    except KeyError as exc:
        raise DtypeToFieldConversionoError(
            f"The conversion of the dtype {dtype} with kind {dtype.kind} "
            "into marshmallow fields is unknown. Known kinds are: "
            f"{DTYPE_KIND_TO_FIELD.keys()}"
        ) from exc


class BaseRecordsDataFrameSchema(Schema):
    """Base schema to generate pandas DataFrame from list of records"""

    class Meta:
        strict = True
        ordered = True

    @post_load(pass_many=True)
    def make_df(self, data, many):
        index_data = {i: row for i, row in enumerate(data)}
        return pd.DataFrame.from_dict(
            index_data, orient="index", columns=self._dtypes.keys().tolist()
        ).astype(self._dtypes)


class BaseSplitDataFrameSchema(Schema):
    """Base schema to generate pandas DataFrame from list of records"""

    class Meta:
        strict = True
        ordered = True

    @post_load(pass_many=True)
    def make_df(self, data, many):
        return pd.DataFrame(dtype=None, **data)


def _create_records_data_field_from_dataframe(df):

    # create marshmallow fields
    input_df_types = {k: v for k, v in zip(df.dtypes.index, df.dtypes.values)}
    input_fields = {k: _dtype_to_field(v) for k, v in input_df_types.items()}
    input_fields["_dtypes"] = df.dtypes

    # create schema dynamically
    DataFrameSchema = type(
        "DataFrameSchema", (BaseRecordsDataFrameSchema,), input_fields
    )

    # return nested schema field
    return fields.Nested(DataFrameSchema, many=True, required=True)


def _create_data_row_field_from_dataframe(df):
    tuple_fields = [_dtype_to_field(dtype) for dtype in df.dtypes.values]
    return fields.Tuple(tuple_fields)


def get_dataframe_schema(sample_input, orient="split"):
    if orient == "records":
        records_data_field = _create_records_data_field_from_dataframe(
            sample_input
        )
        DataFrameSchema = type(
            "RequestRecordsDataFrameSchema",
            (BaseSchema,),
            {"data": records_data_field},
        )
    elif orient == "split":
        data_row_field = _create_data_row_field_from_dataframe(sample_input)
        index_field = _dtype_to_field(sample_input.index.dtype)
        DataFrameSchema = type(
            "RequestSplitDataFrameSchema",
            (BaseSplitDataFrameSchema,),
            {
                "columns": fields.List(
                    fields.String,
                    required=True,
                    validate=validate.Equal(list(sample_input.columns)),
                ),
                "index": fields.List(index_field, required=True),
                "data": fields.List(data_row_field, required=True),
            },
        )
    else:
        raise ValueError(
            f"orient={orient} not supported. Only `split` and "
            "`records are supported"
        )

    return DataFrameSchema
