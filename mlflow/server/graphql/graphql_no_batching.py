from graphql.language.ast import DocumentNode, FieldNode


def count_total_field_calls(ast_node: DocumentNode) -> int:
    """Counts the number of top-level fields in the query (aliased or not)."""
    count = 0

    for definition in ast_node.definitions:
        if selection_set := getattr(definition, "selection_set", None):
            for selection in selection_set.selections:
                if isinstance(selection, FieldNode):
                    count += 1
    return count
