from typing import NamedTuple, Optional

from graphql.error import GraphQLError
from graphql.execution import ExecutionResult
from graphql.language.ast import DocumentNode, FieldNode

from mlflow.environment_variables import (
    MLFLOW_SERVER_GRAPHQL_MAX_ALIASES,
    MLFLOW_SERVER_GRAPHQL_MAX_DEPTH,
    MLFLOW_SERVER_GRAPHQL_MAX_ROOT_FIELDS,
)


class QueryInfo(NamedTuple):
    root_fields: int
    max_depth: int
    max_aliases: int


def scan_query(ast_node: DocumentNode) -> QueryInfo:
    root_fields = 0
    max_depth = 0
    max_aliases = 0

    def get_field_depth(selection_set, current_depth):
        nonlocal max_depth, max_aliases
        if not selection_set:
            return

        # Count aliases at current level
        current_aliases = sum(
            1
            for selection in selection_set.selections
            if isinstance(selection, FieldNode) and selection.alias
        )
        max_aliases = max(max_aliases, current_aliases)

        for selection in selection_set.selections:
            if isinstance(selection, FieldNode):
                if selection.selection_set:
                    get_field_depth(selection.selection_set, current_depth + 1)
                max_depth = max(max_depth, current_depth)

    for definition in ast_node.definitions:
        if selection_set := getattr(definition, "selection_set", None):
            # Count aliases at root level
            root_aliases = sum(
                1
                for selection in selection_set.selections
                if isinstance(selection, FieldNode) and selection.alias
            )
            max_aliases = max(max_aliases, root_aliases)

            for selection in selection_set.selections:
                if isinstance(selection, FieldNode):
                    root_fields += 1
                    if selection.selection_set:
                        get_field_depth(selection.selection_set, 1)

    return QueryInfo(root_fields, max_depth, max_aliases)


def check_query_safety(ast_node: DocumentNode) -> Optional[ExecutionResult]:
    query_info = scan_query(ast_node)
    if query_info.root_fields > MLFLOW_SERVER_GRAPHQL_MAX_ROOT_FIELDS.get():
        msg = "root fields"
        env_var = MLFLOW_SERVER_GRAPHQL_MAX_ROOT_FIELDS
        value = query_info.root_fields
    elif query_info.max_aliases > MLFLOW_SERVER_GRAPHQL_MAX_ALIASES.get():
        msg = "aliases"
        env_var = MLFLOW_SERVER_GRAPHQL_MAX_ALIASES
        value = query_info.max_aliases
    elif query_info.max_depth > MLFLOW_SERVER_GRAPHQL_MAX_DEPTH.get():
        msg = "levels of nesting"
        env_var = MLFLOW_SERVER_GRAPHQL_MAX_DEPTH
        value = query_info.max_depth
    else:
        return None
    return ExecutionResult(
        data=None,
        errors=[
            GraphQLError(
                f"GraphQL queries should have at most {env_var.get()} {msg}, "
                f"got {value} {msg}. To increase the limit, set the "
                f"{env_var.name} environment variable."
            )
        ],
    )
