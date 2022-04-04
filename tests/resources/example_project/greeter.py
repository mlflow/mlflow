"""
Example program helping verify functionality for passing parameters other than those required in
the MLproject file.
"""
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("greeting", help="Greeting to use", type=str)
    parser.add_argument("name", help="Name of person to greet", type=str)
    parser.add_argument("--excitement", help="Excitement level (int) of greeting", type=int)
    args = parser.parse_args()
    greeting = [args.greeting, args.name]
    if args.excitement is not None:
        greeting.append("!" * args.excitement)
    # pylint: disable-next=print-function
    print(" ".join(greeting))
