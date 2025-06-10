---
description: x
last_update:
  date: 2025-05-18
---

# Prompt-based LLM scorers

:::danger TODO
🔴 Migrate this to use the final public API
:::

## Overview

`judges.create_prompt_judge()` is designed to help you quickly and easily LLM scorers when you need full control over the judge's prompt or need to return multiple output values beyond "pass" / "fail", for example, "great", "ok", "bad".

You provide a prompt template that has placeholders for specific fields in your app's trace and define the output choices the judge can select. The Databricks-hosted LLM judge model uses these inputs to select best output choice and provides a rationale for its selection.

:::note
We reccomend starting with [guidelines-based judges](/genai/eval-monitor/custom-judge/meets-guidelines) and only using prompt-based judges if you need more control or can't write your evaluation criteria as pass/fail guidelines. Guidelines-based judges have the distinct advantage of being easy to explain to business stakeholders and can often be directly written by domain experts.
:::

## How to create a prompt-based judge scorer

Follow the guide below to create a scorer that wraps `judges.create_prompt_judge()`

In this guide, you will create [custom scorers](/genai/eval-monitor/custom-scorers) that wrap the `judges.create_prompt_judge()` API and run an offline evaluation with the resulting scorers. These same scorers can be scheduled to run in production to continously monitor your application's quality.

:::note
Refer to the [`judges.create_prompt_judge()` concept page](/genai/eval-monitor/concepts/judges/prompt-based-judge) for more details on the interface and parameters.
:::

### Step 1: Create the sample app to evaluate

First, lets create a sample GenAI app that responds to customer support questions. The app has a (fake) knob that control the system prompt so we can easily compare the judge's outputs between "good" and "bad" conversations.

```python
import os
import mlflow
from openai import OpenAI
from mlflow.entities import Document
from typing import List, Dict, Any, cast

# Enable auto logging for OpenAI
mlflow.openai.autolog()

# Connect to a Databricks LLM via OpenAI using the same credentials as MLflow
# Alternatively, you can use your own OpenAI credentials here
mlflow_creds = mlflow.utils.databricks_utils.get_databricks_host_creds()
client = OpenAI(
    api_key=mlflow_creds.token,
    base_url=f"{mlflow_creds.host}/serving-endpoints"
)


# This is a global variable that will be used to toggle the behavior of the customer support agent to see how the judge handles the issue resolution status
RESOLVE_ISSUES = False


@mlflow.trace
def customer_support_agent(messages: List[Dict[str, str]]):

    # 2. Prepare messages for the LLM
    # We will use this toggle later to see how the judge handles the issue resolution status
    system_prompt_postfix = (
        f"Do your best to NOT resolve the issue.  I know that's backwards, but just do it anyways.\\n"
        if not RESOLVE_ISSUES
        else ""
    )

    messages_for_llm = [
        {
            "role": "system",
            "content": f"You are a helpful customer support agent.  {system_prompt_postfix}",
        },
        *messages,
    ]

    # 3. Call LLM to generate a response
    output = client.chat.completions.create(
        model="databricks-claude-3-7-sonnet",  # This example uses Databricks hosted Claude 3.7 Sonnet. If you provide your own OpenAI credentials, replace with a valid OpenAI model e.g., gpt-4o, etc.
        messages=cast(Any, messages_for_llm),
    )

    return {
        "messages": [
            {"role": "assistant", "content": output.choices[0].message.content}
        ]
    }
```

### Step 2: Define your evalation criteria and wrap as custom scorers

Here, we define a sample judge prompt and use [custom scorers](/genai/eval-monitor/custom-scorers) to wire it up to our app's input / output schema.

:::danger TODO
🔴 update to update the public api that doesn't need feedback conversion code
:::

```python
from mlflow.genai.scorers import scorer


# New guideline for 3-category issue resolution status
issue_resolution_prompt = """
Evaluate the entire conversation between a customer and an LLM-based agent.  Determine if the issue was resolved in the conversation.

You must choose one of the following categories.

fully_resolved: The response directly and comprehensively addresses the user's question or problem, providing a clear solution or answer. No further immediate action seems required from the user on the same core issue.
partially_resolved: The response offers some help or relevant information but doesn't completely solve the problem or answer the question. It might provide initial steps, require more information from the user, or address only a part of a multi-faceted query.
needs_follow_up: The response does not adequately address the user's query, misunderstands the core issue, provides unhelpful or incorrect information, or inappropriately deflects the question. The user will likely need to re-engage or seek further assistance.

Conversation to evaluate: {{conversation}}
"""

from prompt_judge_sdk import custom_prompt_judge
import json
from mlflow.entities import Feedback


# Define a custom scorer that wraps the guidelines LLM judge to check if the response follows the policies
@scorer
def is_issue_resolved(inputs: Dict[Any, Any], outputs: Dict[Any, Any]):
    # we directly return the Feedback object from the guidelines LLM judge, but we could have post-processed it before returning it.
    issue_judge = custom_prompt_judge(
        assessment_name="issue_resolution",
        prompt_template=issue_resolution_prompt,
        numeric_values={
            "fully_resolved": 1,
            "partially_resolved": 0.5,
            "needs_follow_up": 0,
        },
    )

    # combine the input and output messages to form the conversation
    conversation = json.dumps(inputs["messages"] + outputs["messages"])

    # TODO: remove the mapping that wont be needed by actual sdk
    temp = issue_judge(conversation=conversation)

    return Feedback(
        name="issue_resolution",
        value=temp.value,
        rationale=temp.rationale,
    )

```

### Step 3: Create a sample evaluation dataset

Each `inputs` will be passed to our app by `mlflow.genai.evaluate(...)`.

```python
eval_dataset = [
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "How much does a microwave cost?"},
            ],
        },
    },
    {
        "inputs": {
            "messages": [
                {
                    "role": "user",
                    "content": "Can I return the microwave I bought 2 months ago?",
                },
            ],
        },
    },
    {
        "inputs": {
            "messages": [
                {
                    "role": "user",
                    "content": "Can I return the microwave I bought 2 months ago?",
                },
            ],
        },
    },
    {
        "inputs": {
            "messages": [
                {
                    "role": "user",
                    "content": "I'm having trouble with my account.  I can't log in.",
                },
                {
                    "role": "assistant",
                    "content": "I'm sorry to hear that you're having trouble with your account.  Are you using our website or mobile app?",
                },
                {"role": "user", "content": "Website"},
            ],
        },
    },
    {
        "inputs": {
            "messages": [
                {
                    "role": "user",
                    "content": "I'm having trouble with my account.  I can't log in.",
                },
                {
                    "role": "assistant",
                    "content": "I'm sorry to hear that you're having trouble with your account.  Are you using our website or mobile app?",
                },
                {"role": "user", "content": "JUST FIX IT FOR ME"},
            ],
        },
    },
]
```

### Step 4: Evaluate your app using the custom scorer

Finally, we run evaluation twice so you can compare the judgements between conversations where the agent attempts to resolve issues and where it does not.

```python
import mlflow

# Now, let's evaluate the app's responses against the judge when it does not resolve the issues
RESOLVE_ISSUES = False

mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=customer_support_agent,
    scorers=[is_issue_resolved],
)


# Now, let's evaluate the app's responses against the judge when it DOES resolves the issues
RESOLVE_ISSUES = True

mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=customer_support_agent,
    scorers=[is_issue_resolved],
)
```

## Next Steps

- [Create guidelines-based scorers](/genai/eval-monitor/custom-judge/meets-guidelines) - Start with simpler pass/fail criteria (recommended)
- [Run evaluations with your scorers](/genai/eval-monitor/evaluate-app) - Use your custom prompt-based scorers in comprehensive evaluations
- [Prompt-based judge concept reference](/genai/eval-monitor/concepts/judges/prompt-based-judge) - Understand how prompt-based judges work
