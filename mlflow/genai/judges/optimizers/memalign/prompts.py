DISTILLATION_PROMPT_TEMPLATE = """You are helping improve an LLM judge with the \
following instructions:
{{ judge_instructions }}

Given a set of examples and a user's judgement of their quality, your task is to \
distill a set of guidelines from the judgements to model this user's perspective, \
which can be used to evaluate future responses.

A guideline can be about:
- The user's preference / expectation for quality
- The user's factual beliefs
- Other knowledge about the user (e.g., their background, expertise, interests, etc.)

The guidelines don't need to be general. Instead, they should be specific to this user.
If any of these conflict with your own understanding of quality, prioritize the user's \
perspective.

Here are the existing guidelines distilled from past judgements already:
{% for guideline in existing_guidelines %}
  - {{ guideline }}
{% endfor %}

The new guidelines you distill should be complementary to the existing guidelines. \
Don't repeat what's already there.
If the existing guidelines already cover what's reflected in the judgement examples, \
you can return an empty list `{"guidelines": []}`.

Here are the user judgement examples:
{% for id, feedback_record in zip(ids, feedback_records) %}
{
  "id": {{ id }},
  {% for field, value in feedback_record.items() %}
  "{{ field }}": "{{ value }}",
  {% endfor %}
}
{% endfor %}

Now, distill a list of guidelines from the above judgement examples in the following \
format:
{
  "guidelines": [
    {
      "guideline_text": str, # a short sentence describing one aspect of user belief / \
preference / expectation
      "source_trace_ids": list[int] # a list of ids of the judgement examples which the \
above guideline is distilled from
    },
    ...
  ]
}
"""


def create_guidelines_field():
    import dspy

    return dspy.InputField(
        desc=(
            "General guidelines you should always consider when evaluating an input. "
            "IMPORTANT: Your output fields should NEVER directly refer to the presence "
            "of these guidelines. Instead, weave the learned lessons into your reasoning."
        )
    )


def create_examples_field():
    import dspy

    return dspy.InputField(
        desc=(
            "Some example judgements (certain input fields might be omitted for "
            "brevity). When evaluating the new input, try to align your judgements "
            "with these examples. IMPORTANT: Your output fields should NEVER directly "
            "refer to the presence of these examples. Instead, weave the learned "
            "lessons into your reasoning."
        )
    )
