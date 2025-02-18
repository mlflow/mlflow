import transformers

print(transformers.__file__)  # noqa

with open(transformers.__file__, "a") as f:
    f.write("import traceback; traceback.print_stack()")
