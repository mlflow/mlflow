import cohere
import requests
from unittest import mock

co = cohere.Client()
o = requests.Session.request


def p(*args, **kwargs):
    r = o(*args, **kwargs)
    print(r.headers)
    for l in r.iter_lines():
        print(l)
    return r


with mock.patch("requests.Session.request", p):
    for x in co.generate(prompt="Please explain to me how LLMs work", stream=True):
        print(x)
