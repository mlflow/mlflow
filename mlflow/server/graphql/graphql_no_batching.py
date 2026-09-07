from typing import NamedTuple

from graphql.error import GraphQLError
from graphql.execution import ExecutionResult
from graphql.language.ast import (
    DocumentNode,
    FieldNode,
    FragmentDefinitionNode,
    FragmentSpreadNode,
    InlineFragmentNode,
    OperationDefinitionNode,
)

from mlflow.environment_variables import (
    MLFLOW_SERVER_GRAPHQL_MAX_ALIASES,
    MLFLOW_SERVER_GRAPHQL_MAX_ROOT_FIELDS,
)

_MAX_DEPTH = 10
_MAX_SELECTIONS = 1000


class QueryInfo(NamedTuple):
    root_fields: int
    max_aliases: int


def scan_query(ast_node: DocumentNode) -> QueryInfo:
    """
    Scan a GraphQL query and return its information.
    """
    root_fields = 0
    max_aliases = 0
    total_selections = 0

    # Build a map of fragment definitions for lookup when resolving fragment spreads
    fragment_defs = {
        defn.name.value: defn
        for defn in ast_node.definitions
        if isinstance(defn, FragmentDefinitionNode)
    }

    # Only process operation definitions, not fragment definitions
    for definition in ast_node.definitions:
        if not isinstance(definition, OperationDefinitionNode):
            continue

        if selection_set := getattr(definition, "selection_set", None):
            # Stack tracks (selection_set, depth, visited_fragments) to detect cycles
            stack = [(selection_set, 1, frozenset())]
            while stack:
                selection_set, depth, visited_fragments = stack.pop()

                # check current level depth
                if depth > _MAX_DEPTH:
                    raise GraphQLError(f"Query exceeds maximum depth of {_MAX_DEPTH}")

                selections = getattr(selection_set, "selections", [])

                # check current level aliases
                current_aliases = 0
                for selection in selections:
                    if isinstance(selection, FieldNode):
                        if depth == 1:
                            root_fields += 1
                        if selection.alias:
                            current_aliases += 1
                        if selection.selection_set:
                            stack.append((selection.selection_set, depth + 1, visited_fragments))
                        total_selections += 1
                        if total_selections > _MAX_SELECTIONS:
                            raise GraphQLError(
                                f"Query exceeds maximum total selections of {_MAX_SELECTIONS}"
                            )
                    elif isinstance(selection, InlineFragmentNode):
                        # Inline fragments should have their selections counted at the current depth
                        if selection.selection_set:
                            stack.append((selection.selection_set, depth, visited_fragments))
                    elif isinstance(selection, FragmentSpreadNode):
                        # Fragment spreads: count selections at current depth, guard cycles
                        fragment_name = selection.name.value
                        if fragment_name not in visited_fragments:
                            fragment_def = fragment_defs.get(fragment_name)
                            if fragment_def and fragment_def.selection_set:
                                new_visited = visited_fragments | {fragment_name}
                                stack.append((fragment_def.selection_set, depth, new_visited))
                max_aliases = max(max_aliases, current_aliases)

    return QueryInfo(root_fields, max_aliases)


def check_query_safety(ast_node: DocumentNode) -> ExecutionResult | None:
    try:
        query_info = scan_query(ast_node)
    except GraphQLError as e:
        return ExecutionResult(
            data=None,
            errors=[e],
        )

    if query_info.root_fields > MLFLOW_SERVER_GRAPHQL_MAX_ROOT_FIELDS.get():
        msg = "root fields"
        env_var = MLFLOW_SERVER_GRAPHQL_MAX_ROOT_FIELDS
        value = query_info.root_fields
    elif query_info.max_aliases > MLFLOW_SERVER_GRAPHQL_MAX_ALIASES.get():
        msg = "aliases"
        env_var = MLFLOW_SERVER_GRAPHQL_MAX_ALIASES
        value = query_info.max_aliases
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
