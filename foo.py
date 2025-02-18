# import transformers

# print(transformers.__file__)  # noqa

with open("/opt/hostedtoolcache/Python/3.9.18/x64/lib/python3.9/site-packages/transformers/__init__.py", "a") as f:
    f.write("import traceback; traceback.print_stack()")
