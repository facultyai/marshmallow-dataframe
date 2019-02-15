from marshmallow import fields, Schema, post_load, ValidationError


class BaseSchema(Schema):
    """Default schema to use on input"""

    data = fields.Raw(required=True)

    class Meta:
        strict = True

    @post_load()
    def extract_data(self, input_data):
        if input_data is not None:
            return input_data["data"]
        else:
            raise ValidationError(
                {"_schema": ["Invalid input. No data provided."]}
            )
