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
    parser.add_argument("--goodbye", "-g", help="Say goodbye after greeting", action='store_true')
    args = parser.parse_args()
    greeting = [args.greeting, args.name]
    if args.excitement is not None:
        greeting.append("!" * args.excitement)
    print(" ".join(greeting))
    if args.goodbye:
        print("Goodbye {}".format(args.name))
