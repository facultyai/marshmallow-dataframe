from typing import Dict, List, NamedTuple, Union

import marshmallow as ma
import numpy as np
import pandas as pd


class Dtypes(NamedTuple):
    columns: List[str]
    dtypes: List[np.dtype]

    @classmethod
    def from_pandas_dtypes(cls, pd_dtypes: pd.Series) -> "Dtypes":
        return cls(
            columns=list(pd_dtypes.index), dtypes=list(pd_dtypes.values)
        )

    def to_pandas_dtypes(self) -> pd.DataFrame:
        return pd.DataFrame(index=self.columns, data=self.dtypes)


def _validate_dtypes(dtypes: Union[Dtypes, pd.DataFrame]) -> Dtypes:
    if isinstance(dtypes, pd.Series):
        dtypes = Dtypes.from_pandas_dtypes(dtypes)
    elif not isinstance(dtypes, Dtypes):
        raise ValueError(
            "The `dtypes` Meta option on a DataFrame Schema must be either a "
            "pandas Series or an instance of marshmallow_dataframe.Dtypes"
        )

    return dtypes


class DataFrameSchemaOpts(ma.SchemaOpts):
    """Options class for BaseDataFrameSchema

    Adds the following options:
    - ``dtypes``
    - ``index_dtype``
    """

    def __init__(self, meta, *args, **kwargs):
        super().__init__(meta, *args, **kwargs)
        self.dtypes = getattr(meta, "dtypes", None)
        if self.dtypes is not None:
            self.dtypes = _validate_dtypes(self.dtypes)
        self.index_dtype = getattr(meta, "index_dtype", None)
        self.strict = getattr(meta, "strict", True)


class DataFrameSchemaMeta(ma.schema.SchemaMeta):
    """Base metaclass for DataFrame schemas"""

    def __new__(meta, name, bases, class_dict):
        """Only validate subclasses of our schemas"""
        klass = super().__new__(meta, name, bases, class_dict)

        if bases != (ma.Schema,) and klass.opts.dtypes is None:
            raise ValueError(
                "Subclasses of marshmallow_dataframe Schemas must define "
                "the `dtypes` Meta option"
            )

        return klass

    @classmethod
    def get_declared_fields(
        mcs, klass, cls_fields, inherited_fields, dict_cls
    ) -> Dict[str, ma.fields.Field]:
        """
        Updates declared fields with fields generated from DataFrame dtypes
        """

        opts = klass.opts
        declared_fields = super().get_declared_fields(
            klass, cls_fields, inherited_fields, dict_cls
        )
        fields = mcs.get_fields(opts, dict_cls)
        fields.update(declared_fields)
        return fields

    @classmethod
    def get_fields(
        mcs, opts: DataFrameSchemaOpts, dict_cls
    ) -> Dict[str, ma.fields.Field]:
        """
        Generate fields from DataFrame dtypes

        To be implemented in subclasses of DataFrameSchemaMeta
        """
        pass
