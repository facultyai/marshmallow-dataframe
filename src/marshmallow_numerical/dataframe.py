import pandas as pd
import numpy as np
from marshmallow import fields, Schema, post_load, validate
from typing import NamedTuple, List, Union

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


class Dtypes(NamedTuple):
    columns: List[str]
    dtypes: List[np.dtype]

    @classmethod
    def from_pandas_dtypes(cls, pd_dtypes: pd.DataFrame) -> "Dtypes":
        return cls(
            columns=list(pd_dtypes.index), dtypes=list(pd_dtypes.values)
        )

    def to_pandas_dtypes(self) -> pd.DataFrame:
        return pd.DataFrame(index=self.columns, data=self.dtypes)


class DtypeToFieldConversionoError(Exception):
    pass


def _dtype_to_field(dtype: np.dtype) -> fields.Field:
    try:
        kind = dtype.kind
    except AttributeError as exc:
        raise DtypeToFieldConversionoError(
            f"The dtype {dtype} does not have a `kind` attribute, "
            "unable to map dtype into marshmallow field type"
        ) from exc

    try:
        return DTYPE_KIND_TO_FIELD[kind]
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
    def make_df(self, data: dict, many: bool) -> pd.DataFrame:
        index_data = {i: row for i, row in enumerate(data)}
        return pd.DataFrame.from_dict(
            index_data, orient="index", columns=self._dtypes.columns
        ).astype(
            {k: v for k, v in zip(self._dtypes.columns, self._dtypes.dtypes)}
        )


class BaseSplitDataFrameSchema(Schema):
    """Base schema to generate pandas DataFrame from list of records"""

    class Meta:
        strict = True
        ordered = True

    @post_load(pass_many=True)
    def make_df(self, data: dict, many: bool) -> pd.DataFrame:
        return pd.DataFrame(dtype=None, **data)


def _create_records_data_field(dtypes: Dtypes) -> fields.Field:
    # create marshmallow fields
    input_df_types = {k: v for k, v in zip(dtypes.columns, dtypes.dtypes)}
    input_fields = {k: _dtype_to_field(v) for k, v in input_df_types.items()}
    input_fields["_dtypes"] = dtypes

    # create schema dynamically
    DataFrameSchema = type(
        "DataFrameSchema", (BaseRecordsDataFrameSchema,), input_fields
    )

    # return nested schema field
    return fields.Nested(DataFrameSchema, many=True, required=True)


def _create_data_row_field(dtypes: Dtypes) -> fields.Field:
    tuple_fields = [_dtype_to_field(dtype) for dtype in dtypes.dtypes]
    return fields.Tuple(tuple_fields)


def build_records_schema(dtypes: Union[Dtypes, pd.DataFrame]) -> Schema:
    """Build a records-oriented DataFrame Schema"""
    if not isinstance(dtypes, Dtypes):
        dtypes = Dtypes.from_pandas_dtypes(dtypes)
    records_data_field = _create_records_data_field(dtypes)
    return type(
        "RequestRecordsDataFrameSchema",
        (BaseSchema,),
        {"data": records_data_field},
    )


def build_split_schema(
    dtypes: Union[Dtypes, pd.DataFrame], index_dtype: np.dtype = None
) -> Schema:
    """Build a split-oriented DataFrame Schema"""
    if not isinstance(dtypes, Dtypes):
        dtypes = Dtypes.from_pandas_dtypes(dtypes)
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
                validate=validate.Equal(dtypes.columns),
            ),
            "index": fields.List(index_field, required=True),
            "data": fields.List(data_row_field, required=True),
        },
    )


def get_dataframe_schema(
    sample_input: pd.DataFrame, orient: str = "split"
) -> Schema:
    """Build a DataFrame schema from an example dataframe"""
    dtypes = Dtypes.from_pandas_dtypes(sample_input.dtypes)
    if orient == "records":
        DataFrameSchema = build_records_schema(dtypes)
    elif orient == "split":
        DataFrameSchema = build_split_schema(
            dtypes, index_dtype=sample_input.index.dtype
        )
    else:
        raise ValueError(
            f"orient={orient} not supported. Only `split` and "
            "`records are supported"
        )

    return DataFrameSchema
