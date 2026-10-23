import sqlite3

from sqlalchemy.exc import DBAPIError
from sqlparse.exceptions import SQLParseError

from mlflow.exceptions import MlflowException


def scoped_search_limit_error(error: MlflowException) -> MlflowException:
    """Classify known scope limits without changing or retrying the query."""
    cause = error
    seen = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        original = cause.orig if isinstance(cause, DBAPIError) else cause
        sqlite_limit = (
            isinstance(original, sqlite3.OperationalError)
            and str(original) == "too many SQL variables"
        )
        # PostgreSQL's program_limit_exceeded and numeric_value_out_of_range SQLSTATEs.
        sqlstate = getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)
        postgres_limit = sqlstate in {"54000", "22003"}
        parser_limit = isinstance(original, SQLParseError) and str(original).startswith((
            "Maximum number of tokens exceeded",
            "Maximum grouping depth exceeded",
        ))
        if sqlite_limit or postgres_limit or parser_limit:
            return MlflowException.invalid_parameter_value(
                "The combined search request and authorization scope exceeds a backend limit. "
                "Narrow the search or contact an administrator to reduce the authorization scope."
            )
        cause = cause.__cause__ or cause.__context__
    return error
