from collections import defaultdict
from typing import Dict

import marshmallow as ma
import numpy as np
import pandas as pd

DTYPE_KIND_TO_FIELD_CLASS = {
    "i": ma.fields.Int,
    "u": ma.fields.Int,
    "f": ma.fields.Float,
    "O": ma.fields.Str,
    "U": ma.fields.Str,
    "S": ma.fields.Str,
    "b": ma.fields.Bool,
    "M": ma.fields.DateTime,
    "m": ma.fields.TimeDelta,
}

_DEFAULT_FIELD_OPTIONS = {"required": True, "allow_none": True}

DTYPE_KIND_TO_FIELD_OPTIONS: Dict[str, Dict[str, bool]] = defaultdict(
    lambda: _DEFAULT_FIELD_OPTIONS
)
# Integer columns in pandas cannot have null values, so we allow_none for all
# types except int
DTYPE_KIND_TO_FIELD_OPTIONS["i"] = {"required": True}
DTYPE_KIND_TO_FIELD_OPTIONS["f"] = {
    "allow_nan": True,
    **_DEFAULT_FIELD_OPTIONS,
}


class DtypeToFieldConversionError(Exception):
    pass


def dtype_to_field(dtype: np.dtype) -> ma.fields.Field:
    # Object dtypes require more detailed mapping
    if pd.api.types.is_categorical_dtype(dtype):
        categories = dtype.categories.values.tolist()
        kind = dtype.categories.dtype.kind
        field_class = DTYPE_KIND_TO_FIELD_CLASS[kind]
        field_options = DTYPE_KIND_TO_FIELD_OPTIONS[kind]
        return field_class(
            validate=ma.validate.OneOf(categories), **field_options
        )

    try:
        kind = dtype.kind
    except AttributeError as exc:
        raise DtypeToFieldConversionError(
            f"The dtype {dtype} does not have a `kind` attribute, "
            "unable to map dtype into marshmallow field type"
        ) from exc

    try:
        field_class = DTYPE_KIND_TO_FIELD_CLASS[kind]
        field_options = DTYPE_KIND_TO_FIELD_OPTIONS[kind]
        return field_class(**field_options)
    except KeyError as exc:
        raise DtypeToFieldConversionError(
            f"The conversion of the dtype {dtype} with kind {dtype.kind} "
            "into marshmallow fields is unknown. Known kinds are: "
            f"{DTYPE_KIND_TO_FIELD_CLASS.keys()}"
        ) from exc
