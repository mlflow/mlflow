import six


def to_text(val):
    """ Converts the passed-in value to a text type value. """
    if isinstance(val, six.text_type):
        return val
    if isinstance(val, six.binary_type):
        return val.decode("utf-8")
    # Assume we can convert the value to an ASCII string -> unicode
    return u"%s" % val
