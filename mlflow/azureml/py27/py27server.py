import sys
import pandas as pd
from mlflow.pyfunc import load_pyfunc
from mlflow.utils import get_jsonable_obj


def main(model_path):
    model = load_pyfunc(model_path)
    for json_content in sys.stdin:
        print(score(model, json_content.decode("utf-8")))


def score(model, content):
    input_df = pd.read_json(content, orient="records")
    return get_jsonable_obj(model.predict(input_df))


if __name__ == "__main__":
    model_path = sys.argv[1]
    main(model_path)
