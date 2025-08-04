import json

from sqlalchemy.dialects import mysql, postgresql
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.types import TEXT, TypeDecorator


class JSONEncodedDict(TypeDecorator):
    """Represents a JSON structure with automatic marshalling and per-backend optimization.

    This type:
    - Uses native JSON types for PostgreSQL and MySQL
    - Falls back to TEXT with JSON marshalling for MSSQL and others
    - Automatically handles Python dict to/from JSON conversion
    - Supports mutation tracking when used with MutableDict.as_mutable()

    Usage:
        # In model definition
        tags = Column(MutableJSON)

        # Or if you want immutable JSON
        tags = Column(JSONEncodedDict)
    """

    impl = TEXT
    cache_ok = True

    def load_dialect_impl(self, dialect):
        """Choose the best implementation based on the dialect."""
        if dialect.name == "postgresql":
            # Use native PostgreSQL JSON type
            return dialect.type_descriptor(postgresql.JSON())
        elif dialect.name == "mysql":
            # Use native MySQL JSON type
            return dialect.type_descriptor(mysql.JSON())
        else:
            # For MSSQL, SQLite, and others, use TEXT
            return dialect.type_descriptor(TEXT())

    def process_bind_param(self, value, dialect):
        """Convert Python dict to storage format."""
        if value is None:
            return None

        # For native JSON types, pass through as-is
        if dialect.name in ("postgresql", "mysql"):
            return value

        # For others, serialize to JSON string
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        """Convert storage format to Python dict."""
        if value is None:
            return None

        # For native JSON types, it's already deserialized
        if dialect.name in ("postgresql", "mysql"):
            return value

        # For others, deserialize from JSON string
        if isinstance(value, str):
            return json.loads(value)
        return value


# Create a mutable variant that tracks changes
MutableJSON = MutableDict.as_mutable(JSONEncodedDict)
