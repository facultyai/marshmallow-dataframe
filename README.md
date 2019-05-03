# marshmallow-numerical

[![Build Status](https://travis-ci.org/zblz/marshmallow-numerical.svg?branch=master)](https://travis-ci.org/zblz/marshmallow-numerical)
[![PyPI](https://img.shields.io/pypi/v/marshmallow-numerical.svg)](https://pypi.org/project/marshmallow-numerical/)
[![License](https://img.shields.io/github/license/zblz/marshmallow-numerical.svg)](https://github.com/zblz/marshmallow-numerical/blob/master/LICENSE)

`marshmallow-numerical` is a library that helps you generate
[marshmallow](https://marshmallow.readthedocs.io/) Schemas for Pandas and Numpy
data structures.

# Usage

Let's start by creating an example dataframe for which we want to create a
`Schema`. This dataframe has four columns: two of them are of string type, one
is a float, and the last one is an integer.

```python
import pandas as pd
import numpy as np
from marshmallow_numerical import SplitDataFrameSchema

animal_df = pd.DataFrame(
    [
        ("falcon", "bird", 389.0, 2),
        ("parrot", "bird", 24.0, 2),
        ("lion", "mammal", 80.5, 4),
        ("monkey", "mammal", np.nan, 4),
    ],
    columns=["name", "class", "max_speed", "num_legs"],
)
```

You can then create a marshmallow schema that will validate and load dataframes
that follow the same structure as the one above and that have been serialized
with `DataFrame.to_json` with the [`orient=split`
format](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_json.html#pandas.DataFrame.to_json):

```python
class AnimalSchema(SplitDataFrameSchema):
    """Automatically generated schema for animal dataframe"""

    dtypes = animal_df.dtypes
```

When passing a valid payload for a new animal, this schema will validate it and
build a dataframe:

```python
animal_schema = AnimalSchema()

new_animal = {
    "data": [("leopard", "mammal", 58.0, 4)],
    "columns": ["name", "class", "max_speed", "num_legs"],
    "index": [0],
}

new_animal_df = schema.load(new_animal)

print(new_animal_df)
#       name   class  max_speed  num_legs
# 0  leopard  mammal     58.000         4
# 1      ant  insect      0.288         6
```

However, if we pass a payload that doesn't conform to the schema, it will raise
a marshmallow `ValidationError` exception with informative message about errors:

```python
invalid_animal = {
    "data": [("leopard", "mammal", 58.0, "four")],  # num_legs is not an int
    "columns": ["name", "class", "num_legs"],  # missing  max_speed column
    "index": [0],
}

schema.load(invalid_animal)

# Raises:
# marshmallow.exceptions.ValidationError: {'columns': ["Must be equal to ['name', 'class', 'max_speed', 'num_legs']."], 'data': {0: {3: ['Not a valid integer.']}}}
```

`marshmallow_numerical` can also generate Schemas for the `orient=records`
format by following the above steps but using
`marshmallow_numerical.RecordsDataFrameSchema` as the superclass for
`AnimalSchema`.

# Installation

marshmallow-numerical requires Python >= 3.6 and marshmallow >= 3.0. You can
install it with pip:

```
pip install marshmallow-numerical
```

# Contributing

Contributions are welcome!

You can report a problem or feature request in the [issue
tracker](https://github.org/zblz/marshmallow_numerical/issues). If you feel that
you can fix it or implement it, please submit a pull request referencing the
issues it solves.

Unit tests written using the [`pytest`](https://pytest.org) framework are in the
`tests` directory, and are run using
[tox](https://tox.readthedocs.io/en/latest/) on Python 3.6 and 3.7. You can run
the tests by installing tox:
```
pip install tox
```
and running the linters and tests for all Python versions by running `tox`, or
for a specific Python version by running:
```
tox -e py36
```

We format the code with [black](https://github.com/ambv/black), and you can
format your checkout of the code before commiting it by running:
```
tox -e black -- .
```
