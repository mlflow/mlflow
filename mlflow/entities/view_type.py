class ViewType(object):
    """Enum to qualify `ListExperiments` API query for requested experiment types."""
    ACTIVE_ONLY, DELETED_ONLY, ALL = range(1, 4)
