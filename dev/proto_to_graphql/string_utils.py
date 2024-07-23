import re


def camel_to_snake(string):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", string).lower()


def snake_to_camel(string):
    return "".join(x[0].upper() + x[1:] for x in string.split("_"))


def snake_to_pascal(string):
    temp = snake_to_camel(string)
    return temp[0].upper() + temp[1:]
