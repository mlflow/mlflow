class ViewType(object):
    """Enum to filter requested experiment types."""
    ACTIVE_ONLY, DELETED_ONLY, ALL = range(1, 4)
    _VIEW_TO_STRING = {
        ACTIVE_ONLY: "active_only",
        DELETED_ONLY: "deleted_only",
        ALL: "all",
    }
    _STRING_TO_VIEW = {value: key for key, value in _VIEW_TO_STRING.items()}

    @staticmethod
    def from_string(view_str):
        if view_str not in ViewType._STRING_TO_VIEW:
            raise Exception(
                "Could not get valid view type corresponding to string %s. "
                "Valid view types are %s" % (view_str, list(ViewType._STRING_TO_VIEW.keys())))
        return ViewType._STRING_TO_VIEW[view_str]

    @staticmethod
    def to_string(view_type):
        if view_type not in ViewType._VIEW_TO_STRING:
            raise Exception(
                "Could not get valid view type corresponding to string %s. "
                "Valid view types are %s" % (view_type, list(ViewType._VIEW_TO_STRING.keys())))
        return ViewType._VIEW_TO_STRING[view_type]
