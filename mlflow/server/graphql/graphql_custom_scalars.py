import graphene
from graphql.language.ast import IntValueNode


class LongString(graphene.Scalar):
    """
    LongString Scalar type to prevent truncation to max integer in JavaScript.
    """

    description = "Long converted to string to prevent truncation to max integer in JavaScript"

    @staticmethod
    def serialize(long):
        return str(long)

    @staticmethod
    def parse_literal(node):
        if isinstance(node, IntValueNode):
            return int(node.value)
        return None

    @staticmethod
    def parse_value(value):
        return int(value)
