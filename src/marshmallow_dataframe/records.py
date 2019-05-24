from typing import Dict

import marshmallow as ma
import pandas as pd

from .base import DataFrameSchemaMeta, DataFrameSchemaOpts
from .converters import dtype_to_field


class RecordsDataFrameSchemaMeta(DataFrameSchemaMeta):
    @classmethod
    def get_fields(
        mcs, opts: DataFrameSchemaOpts, dict_cls
    ) -> Dict[str, ma.fields.Field]:

        if opts.dtypes is not None:

            # create marshmallow fields
            input_fields = {
                k: dtype_to_field(v)
                for k, v in zip(opts.dtypes.columns, opts.dtypes.dtypes)
            }

            # create schema dynamically
            RecordSchema = type("RecordSchema", (ma.Schema,), input_fields)

            fields: Dict[str, ma.fields.Field] = dict_cls()

            fields["data"] = ma.fields.Nested(
                RecordSchema, many=True, required=True
            )

            return fields

        return dict_cls()


class RecordsDataFrameSchema(ma.Schema, metaclass=RecordsDataFrameSchemaMeta):
    """Schema to generate pandas DataFrame from list of records"""

    OPTIONS_CLASS = DataFrameSchemaOpts

    @ma.post_load
    def make_df(self, data: dict) -> pd.DataFrame:
        records_data = data["data"]
        index_data = {i: row for i, row in enumerate(records_data)}
        return pd.DataFrame.from_dict(
            index_data, orient="index", columns=self.opts.dtypes.columns
        ).astype(dict(zip(self.opts.dtypes.columns, self.opts.dtypes.dtypes)))
