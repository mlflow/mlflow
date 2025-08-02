import re


def camel_to_snake(string):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", string).lower()


def snake_to_camel(string):
    return "".join(x[0].upper() + x[1:] for x in string.split("_"))


def snake_to_pascal(string):
    temp = snake_to_camel(string)
    return temp[0].upper() + temp[1:]


def camel_to_snake_(string):
    # we will result an empty array
    result=[]
    #Iterating using enumerate to get both string index i and index value which is character char
    for i, char in enumerate(string):
# using if condition to check whether it is upper or lower excepth 0th position character
        if(char.isupper() & (i!=0)):
            # we will add _ if any upper character found in here
            result.append('_')
        # this will make every character lower even if it is upper
        result.append(char.lower())
# now we will return the list here
    return ''.join(result)
