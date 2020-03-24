from functools import wraps
import click


def parse_custom_arguments(f):
    """
    This function decorator acts as a CLI decorator for all the plugin CLI functions to
    parse the extra arguments which are supposed to be plugin arguments.

    .. Note::
        To make the flow easier, this function will only take long options (options
        prefixed with double hyphen and has more than one character) and parse them
        as keyword arguments. Hence each argument is excepted to be a pair. For example:
        --host localhost.

    .. Warning::
        This function will not parse any boolean flag options. A flag option is essentially
        returns a boolean based on the presence of the option in the command and does not
        associate with a value. A good example is "-v" for verbose execution.
    """
    @wraps(f)
    @click.pass_context
    def wrapper(ctx, *args, **kwargs):
        try:
            for i in range(0, len(ctx.args), 2):
                key = ctx.args[i]
                if not key.startswith('--'):
                    raise RuntimeError("Could not parse argument {}. All the plugin specific "
                                       "arguments must be long options i.e, it should "
                                       "be prefixed with `--` and will have more than one "
                                       "character".format(key))
                try:
                    val = ctx.args[i + 1]
                except IndexError:
                    raise RuntimeError("Could not find the value "
                                       "for the keyword option {}".format(key))
                kwargs[key[2:]] = val
            return f(*args, **kwargs)
        except Exception as e:
            raise click.ClickException(e)
    return wrapper
