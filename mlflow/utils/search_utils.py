def does_run_match_clause(run, search_expression):
    key_type = search_expression.WhichOneof('expression')
    if key_type == 'metric':
        key = search_expression.metric.key
        comparator = search_expression.metric.float.comparator
        value = search_expression.metric.float.value
        metric = next((m for m in run.data.metrics if m.key == key), None)
        if metric is None:
            return False
        if comparator == '>':
            return metric.value > value
        elif comparator == '>=':
            return metric.value >= value
        elif comparator == '=':
            return metric.value == value
        elif comparator == '!=':
            return metric.value != value
        elif comparator == '<=':
            return metric.value <= value
        elif comparator == '<':
            return metric.value < value
        else:
            raise Exception("Invalid comparator '%s' not one of '>, >=, =, !=, <=, <"
                            % comparator)
    if key_type == 'parameter':
        key = search_expression.parameter.key
        comparator = search_expression.parameter.string.comparator
        value = search_expression.parameter.string.value
        param = next((p for p in run.data.params if p.key == key), None)
        if param is None:
            return False
        if comparator == '=':
            return param.value == value
        elif comparator == '!=':
            return param.value != value
        else:
            raise Exception("Invalid comparator '%s' not one of '=, !=" % comparator)
    return False
