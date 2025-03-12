import argparse
import json

import openai


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uc-function-name",
        type=str,
        required=True,
        help="Name of the UC function to use",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    client = openai.OpenAI(base_url="http://localhost:7000/v1")

    print("----- UC function -----")
    uc_function = {
        "type": "uc_function",
        "uc_function": {
            "name": args.uc_function_name,
        },
    }

    resp = client.chat.completions.create(
        model="chat",
        messages=[
            {
                "role": "user",
                "content": "What is the result of 1 + 2?",
            }
        ],
        tools=[uc_function],
    )
    print(resp.choices[0].message.content)

    print("----- UC function + User-defined function -----")
    user_defined_function = {
        "type": "function",
        "function": {
            "description": "Multiply numbers",
            "name": "multiply",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "description": "First number",
                    },
                    "y": {
                        "type": "integer",
                        "description": "Second number",
                    },
                },
                "required": ["x", "y"],
            },
        },
    }

    def multiply(x: int, y: int) -> int:
        return x * y

    msg = {
        "role": "user",
        "content": (
            "What is the result of 1 + 2? What is the result of 3 + 4? What is the result of 5 * 6?"
        ),
    }
    resp = client.chat.completions.create(
        model="chat",
        messages=[msg],
        tools=[
            user_defined_function,
            uc_function,
        ],
    )

    print(resp.choices[0].message.content)
    print(resp.choices[0].message.tool_calls)

    multiply_call = resp.choices[0].message.tool_calls[0].function
    assert multiply_call.name == "multiply"
    resp = client.chat.completions.create(
        model="chat",
        messages=[
            msg,
            {
                "role": "assistant",
                "content": resp.choices[0].message.content,
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": resp.choices[0].message.tool_calls,
            },
            {
                "role": "tool",
                "tool_call_id": resp.choices[0].message.tool_calls[0].id,
                "content": str(multiply(**json.loads(multiply_call.arguments))),
            },
        ],
    )

    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()
