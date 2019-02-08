import pandas as pd
import numpy as np
from pandas.core.dtypes import dtypes as pd_dtypes
from marshmallow import fields, Schema, post_load

from .base import BaseSchema

FIELD_OPTIONS = {"required": True, "allow_none": True}

# Integer columns in pandas cannot have null values, so here we allow_none for
# all types except int
DTYPE_TO_FIELD = {
    np.dtype(np.int64): fields.Int(required=True),
    np.dtype(np.float64): fields.Float(**FIELD_OPTIONS),
    np.dtype(object): fields.Str(**FIELD_OPTIONS),
    np.dtype(bool): fields.Bool(**FIELD_OPTIONS),
    np.dtype("datetime64[ns]"): fields.DateTime(**FIELD_OPTIONS),
    pd_dtypes.DatetimeTZDtype('ns', 'UTC'): fields.DateTime(**FIELD_OPTIONS),
    np.dtype("timedelta64[ns]"): fields.TimeDelta(**FIELD_OPTIONS),
}


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
        raise NotImplementedError


def _create_records_data_field_from_dataframe(df):

    # create marshmallow fields
    input_df_types = {k: v for k, v in zip(df.dtypes.index, df.dtypes.values)}
    input_fields = {k: DTYPE_TO_FIELD[v] for k, v in input_df_types.items()}
    input_fields["_dtypes"] = df.dtypes

    # create schema dynamically
    DataFrameSchema = type(
        "DataFrameSchema", (BaseRecordsDataFrameSchema,), input_fields
    )

    # return nested schema field
    return fields.Nested(DataFrameSchema, many=True, required=True)


def _create_data_row_field_from_dataframe(df):
    raise NotImplementedError


def get_dataframe_schema(sample_input, orient="split"):
    if orient == "records":
        records_data_field = _create_records_data_field_from_dataframe(
            sample_input
        )
        DataFrameSchema = type(
            "RequestDataFrameSchema",
            (BaseSchema,),
            {"data": records_data_field},
        )
    elif orient == "split":
        data_row_field = _create_data_row_field_from_dataframe(sample_input)
        DataFrameSchema = type(
            "RequestSplitDataFrameSchema",
            (BaseSplitDataFrameSchema,),
            {
                "columns": fields.String(many=True, required=True),
                "index": fields.String(many=True, required=True),
                "data": fields.List(data_row_field, required=True),
            },
        )
    else:
        raise ValueError(
            f"orient={orient} not supported. Only `split` and "
            "`records are supported"
        )

    return DataFrameSchema
