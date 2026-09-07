from collections import deque
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
    SelectionSetNode,
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


def _collect_fields_and_aliases(
    selection_set: SelectionSetNode,
    fragment_defs: dict[str, FragmentDefinitionNode],
    visited_fragments: frozenset[str],
) -> tuple[list[FieldNode], int]:
    """
    Recursively collects all FieldNode selections and counts total aliases
    from a selection set, including those nested in inline fragments and
    fragment spreads at the same level. This ensures aliases from all
    sibling fragments are aggregated into one alias count.

    Returns (field_selections, total_aliases) where:
    - field_selections: list of FieldNode selections to process
    - total_aliases: count of all aliases from this level
    """
    field_selections = []
    total_aliases = 0
    # deque so popleft is O(1); order does not affect the field/alias counts.
    selections_to_process = deque(selection_set.selections)
    # Track fragments visited during this collection to prevent cycles
    local_visited = set(visited_fragments)

    while selections_to_process:
        selection = selections_to_process.popleft()

        if isinstance(selection, FieldNode):
            field_selections.append(selection)
            if selection.alias:
                total_aliases += 1
        elif isinstance(selection, InlineFragmentNode):
            # Inline fragment selections at this level merge into parent scope
            if selection.selection_set:
                selections_to_process.extend(selection.selection_set.selections)
        elif isinstance(selection, FragmentSpreadNode):
            # Fragment spread selections at this level merge into parent scope
            fragment_name = selection.name.value
            if fragment_name not in local_visited:
                local_visited.add(fragment_name)
                fragment_def = fragment_defs.get(fragment_name)
                if fragment_def and fragment_def.selection_set:
                    selections_to_process.extend(fragment_def.selection_set.selections)

    return field_selections, total_aliases


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

                # Collect all fields and aggregate aliases from this level,
                # including those inside inline fragments and fragment spreads.
                # This prevents attackers from distributing aliases across multiple
                # sibling fragments to bypass the alias limit.
                field_selections, current_aliases = _collect_fields_and_aliases(
                    selection_set, fragment_defs, visited_fragments
                )

                for field in field_selections:
                    if depth == 1:
                        root_fields += 1
                    if field.selection_set:
                        stack.append((field.selection_set, depth + 1, visited_fragments))
                    total_selections += 1
                    if total_selections > _MAX_SELECTIONS:
                        raise GraphQLError(
                            f"Query exceeds maximum total selections of {_MAX_SELECTIONS}"
                        )

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
