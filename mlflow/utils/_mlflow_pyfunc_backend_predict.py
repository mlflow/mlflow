"""
This script should be executed in a fresh python interpreter process using `subprocess`.
"""
import argparse

from mlflow.pyfunc.scoring_server import _predict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--content-type", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    return _predict(
        model_uri=args.model_uri,
        input_path=args.input_path,
        output_path=args.output_path,
        content_type=args.content_type,
    )


if __name__ == "__main__":
    main()
