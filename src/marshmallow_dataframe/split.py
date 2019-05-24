from typing import Dict

import marshmallow as ma
import pandas as pd

from .base import DataFrameSchemaMeta, DataFrameSchemaOpts
from .converters import dtype_to_field


class SplitDataFrameSchemaMeta(DataFrameSchemaMeta):
    @classmethod
    def get_fields(
        mcs, opts: DataFrameSchemaOpts, dict_cls
    ) -> Dict[str, ma.fields.Field]:

        if opts.dtypes is not None:
            index_dtype = opts.index_dtype

            fields: Dict[str, ma.fields.Field] = dict_cls()

            data_tuple_fields = [
                dtype_to_field(dtype) for dtype in opts.dtypes.dtypes
            ]
            fields["data"] = ma.fields.List(
                ma.fields.Tuple(data_tuple_fields), required=True
            )

            index_field = (
                ma.fields.Raw()
                if index_dtype is None
                else dtype_to_field(index_dtype)
            )

            fields["index"] = ma.fields.List(index_field, required=True)

            fields["columns"] = ma.fields.List(
                ma.fields.String,
                required=True,
                validate=ma.validate.Equal(opts.dtypes.columns),
            )

            return fields

        return dict_cls()


class SplitDataFrameSchema(ma.Schema, metaclass=SplitDataFrameSchemaMeta):
    """Schema to generate pandas DataFrame from split oriented JSON"""

    OPTIONS_CLASS = DataFrameSchemaOpts

    @ma.validates_schema(skip_on_field_errors=True)
    def validate_index_data_length(self, data: dict) -> None:
        if len(data["index"]) != len(data["data"]):
            raise ma.ValidationError(
                "Length of `index` and `data` must be equal.", "data"
            )

    @ma.post_load
    def make_df(self, data: dict) -> pd.DataFrame:
        df = pd.DataFrame(dtype=None, **data).astype(
            dict(zip(self.opts.dtypes.columns, self.opts.dtypes.dtypes))
        )

        return df
