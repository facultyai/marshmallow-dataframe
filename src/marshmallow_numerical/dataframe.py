import pandas as pd
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


def _create_records_data_field(dtypes):
    # create marshmallow fields
    input_df_types = {k: v for k, v in zip(dtypes.index, dtypes.values)}
    input_fields = {k: _dtype_to_field(v) for k, v in input_df_types.items()}
    input_fields["_dtypes"] = dtypes

    # create schema dynamically
    DataFrameSchema = type(
        "DataFrameSchema", (BaseRecordsDataFrameSchema,), input_fields
    )

    # return nested schema field
    return fields.Nested(DataFrameSchema, many=True, required=True)


def _create_data_row_field(dtypes):
    tuple_fields = [_dtype_to_field(dtype) for dtype in dtypes.values]
    return fields.Tuple(tuple_fields)


def build_records_schema(dtypes):
    records_data_field = _create_records_data_field(dtypes)
    return type(
        "RequestRecordsDataFrameSchema",
        (BaseSchema,),
        {"data": records_data_field},
    )


def build_split_schema(dtypes, index_dtype=None):
    data_row_field = _create_data_row_field(dtypes)
    if index_dtype is None:
        index_field = fields.Raw()
    else:
        index_field = _dtype_to_field(index_dtype)

    return type(
        "RequestSplitDataFrameSchema",
        (BaseSplitDataFrameSchema,),
        {
            "columns": fields.List(
                fields.String,
                required=True,
                validate=validate.Equal(list(dtypes.index)),
            ),
            "index": fields.List(index_field, required=True),
            "data": fields.List(data_row_field, required=True),
        },
    )


def get_dataframe_schema(sample_input, orient="split"):
    if orient == "records":
        DataFrameSchema = build_records_schema(sample_input.dtypes)
    elif orient == "split":
        DataFrameSchema = build_split_schema(
            sample_input.dtypes, index_dtype=sample_input.index.dtype
        )
    else:
        raise ValueError(
            f"orient={orient} not supported. Only `split` and "
            "`records are supported"
        )

    return DataFrameSchema
