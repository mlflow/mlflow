from functools import wraps
import click


def parse_custom_arguments(f):
    @wraps(f)
    @click.pass_context
    def wrapper(ctx, *args, **kwargs):
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
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise click.ClickException(e)
    return wrapper
