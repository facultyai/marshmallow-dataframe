import pandas as pd
import numpy as np
from marshmallow import (
    fields,
    Schema,
    post_load,
    validate,
    ValidationError,
    validates_schema,
)
from typing import NamedTuple, List, Union, Optional, Dict

__all__ = ["Dtypes", "RecordsDataFrameSchema", "SplitDataFrameSchema"]

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


def _validate_dtypes_attribute(schema: Schema) -> Dtypes:
    try:
        dtypes = schema.dtypes
    except AttributeError as exc:
        raise NotImplementedError(
            f"Subclasses of {schema.__class__.__name__} must define the "
            "`dtypes` attribute"
        ) from exc

    if isinstance(dtypes, pd.Series):
        dtypes = Dtypes.from_pandas_dtypes(dtypes)
    elif not isinstance(dtypes, Dtypes):
        raise ValueError(
            "The `dtypes` attribute on subclasses of "
            f"{schema.__class__.__name__} must be either a pandas Series "
            "or an instance of marshmallow_numerical.Dtypes"
        )

    return dtypes


class RecordsDataFrameSchema(Schema):
    """Schema to generate pandas DataFrame from list of records"""

    # Configuration attributes, should be implemented by subclasses
    dtypes: Union[Dtypes, pd.DataFrame]

    def __init__(self, *args, **kwargs):

        self._dtypes = _validate_dtypes_attribute(self)

        # create marshmallow fields
        input_fields = {
            k: _dtype_to_field(v)
            for k, v in zip(self._dtypes.columns, self._dtypes.dtypes)
        }

        # create schema dynamically
        RecordSchema = type("RecordSchema", (Schema,), input_fields)

        df_fields: Dict[str, fields.Field] = {
            "data": fields.Nested(RecordSchema, many=True, required=True)
        }

        self._declared_fields.update(df_fields)

        super().__init__(*args, **kwargs)

    class Meta:
        strict = True

    @post_load
    def make_df(self, data: dict) -> pd.DataFrame:
        records_data = data["data"]
        index_data = {i: row for i, row in enumerate(records_data)}
        return pd.DataFrame.from_dict(
            index_data, orient="index", columns=self._dtypes.columns
        ).astype(
            {k: v for k, v in zip(self._dtypes.columns, self._dtypes.dtypes)}
        )


class SplitDataFrameSchema(Schema):
    """Schema to generate pandas DataFrame from split oriented JSON"""

    # Configuration attributes, should be implemented by subclasses
    dtypes: Union[Dtypes, pd.DataFrame]
    index_dtype: Optional[np.dtype] = None

    def __init__(self, *args, **kwargs):
        self._dtypes = _validate_dtypes_attribute(self)

        df_fields: Dict[str, fields.Field] = {}

        data_tuple_fields = [
            _dtype_to_field(dtype) for dtype in self._dtypes.dtypes
        ]
        df_fields["data"] = fields.List(
            fields.Tuple(data_tuple_fields), required=True
        )

        index_field = (
            fields.Raw()
            if self.index_dtype is None
            else _dtype_to_field(self.index_dtype)
        )

        df_fields["index"] = fields.List(index_field, required=True)

        df_fields["columns"] = fields.List(
            fields.String,
            required=True,
            validate=validate.Equal(self._dtypes.columns),
        )

        self._declared_fields.update(df_fields)

        super().__init__(*args, **kwargs)

    class Meta:
        strict = True

    @validates_schema(skip_on_field_errors=True)
    def validate_index_data_length(self, data: dict) -> None:
        if len(data["index"]) != len(data["data"]):
            raise ValidationError(
                "Length of `index` and `data` must be equal.", "data"
            )

    @post_load
    def make_df(self, data: dict) -> pd.DataFrame:
        return pd.DataFrame(dtype=None, **data)
