import pandas as pd
import numpy as np
import marshmallow as ma
from typing import NamedTuple, List, Union, Dict

__all__ = ["Dtypes", "RecordsDataFrameSchema", "SplitDataFrameSchema"]

_FIELD_OPTIONS = {"required": True, "allow_none": True}

# Integer columns in pandas cannot have null values, so here we allow_none for
# all types except int
DTYPE_KIND_TO_FIELD = {
    "i": ma.fields.Int(required=True),
    "u": ma.fields.Int(**_FIELD_OPTIONS),
    "f": ma.fields.Float(allow_nan=True, **_FIELD_OPTIONS),
    "O": ma.fields.Str(**_FIELD_OPTIONS),
    "b": ma.fields.Bool(**_FIELD_OPTIONS),
    "M": ma.fields.DateTime(**_FIELD_OPTIONS),
    "m": ma.fields.TimeDelta(**_FIELD_OPTIONS),
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


class DtypeToFieldConversionError(Exception):
    pass


def _dtype_to_field(dtype: np.dtype) -> ma.fields.Field:
    try:
        kind = dtype.kind
    except AttributeError as exc:
        raise DtypeToFieldConversionError(
            f"The dtype {dtype} does not have a `kind` attribute, "
            "unable to map dtype into marshmallow field type"
        ) from exc

    try:
        return DTYPE_KIND_TO_FIELD[kind]
    except KeyError as exc:
        raise DtypeToFieldConversionError(
            f"The conversion of the dtype {dtype} with kind {dtype.kind} "
            "into marshmallow fields is unknown. Known kinds are: "
            f"{DTYPE_KIND_TO_FIELD.keys()}"
        ) from exc


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


class RecordsDataFrameSchemaMeta(DataFrameSchemaMeta):
    @classmethod
    def get_fields(
        mcs, opts: DataFrameSchemaOpts, dict_cls
    ) -> Dict[str, ma.fields.Field]:

        if opts.dtypes is not None:

            # create marshmallow fields
            input_fields = {
                k: _dtype_to_field(v)
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


class SplitDataFrameSchemaMeta(DataFrameSchemaMeta):
    @classmethod
    def get_fields(
        mcs, opts: DataFrameSchemaOpts, dict_cls
    ) -> Dict[str, ma.fields.Field]:

        if opts.dtypes is not None:
            index_dtype = opts.index_dtype

            fields: Dict[str, ma.fields.Field] = dict_cls()

            data_tuple_fields = [
                _dtype_to_field(dtype) for dtype in opts.dtypes.dtypes
            ]
            fields["data"] = ma.fields.List(
                ma.fields.Tuple(data_tuple_fields), required=True
            )

            index_field = (
                ma.fields.Raw()
                if index_dtype is None
                else _dtype_to_field(index_dtype)
            )

            fields["index"] = ma.fields.List(index_field, required=True)

            fields["columns"] = ma.fields.List(
                ma.fields.String,
                required=True,
                validate=ma.validate.Equal(opts.dtypes.columns),
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
        ).astype(
            {
                k: v
                for k, v in zip(
                    self.opts.dtypes.columns, self.opts.dtypes.dtypes
                )
            }
        )


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
        return pd.DataFrame(dtype=None, **data)
