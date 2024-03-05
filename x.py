import datasets

for _ in range(100):
    ds = datasets.load_dataset("rotten_tomatoes", split="train")
    print(ds.builder_name)  # noqa
