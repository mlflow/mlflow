import CodeBlock from "@theme/CodeBlock";
import Link from "@docusaurus/Link";

const rows = {
  data_format: {
    formattedName: "Data Format",
  },
  example: {
    formattedName: "Example",
  },
  additional_notes: {
    formattedName: "Additional Notes",
  },
};

const format = { messages: [{ role: "user", content: "What is MLflow?" }] };
const columns = [
  {
    data_format: <span>A pandas DataFrame with a string column.</span>,
    example: (
      <CodeBlock language="python">
        {`pd.DataFrame(
  {
    "inputs": [
      "What is MLflow?",
      "What is Spark?",
    ]
  }
)`}
      </CodeBlock>
    ),
    additional_notes: (
      <span>
        For this input format, MLflow will construct the appropriate request
        payload to the model endpoint type. For example, if your model is a chat
        endpoint (`llm/v1/chat`), MLflow will wrap your input string with the
        chat messages format like <code>{JSON.stringify(format)}</code>. If you
        want to customize the request payload e.g. including system prompt,
        please use the next format.
      </span>
    ),
  },
  {
    data_format: <span>A pandas DataFrame with a dictionary column.</span>,
    example: (
      <CodeBlock language="python">
        {`pd.DataFrame(
    {
      "inputs": [
        {
          "messages": [
            {
              "role": "system", 
              "content": "Please answer."
            },
            {
              "role": "user", 
              "content": "What is MLflow?"
            },
          ],
          "max_tokens": 100,
        },
        # ... more dictionary records
      ]
    }
  )`}
      </CodeBlock>
    ),
    additional_notes: (
      <span>
        In this format, the dictionary should have the correct request format
        for your model endpoint. Please refer to the{" "}
        <Link to="../deployments/index.mdx#standard-query-parameters">
          MLflow Deployments documentation
        </Link>{" "}
        for more information about the request format for different model
        endpoint types.
      </span>
    ),
  },
  {
    data_format: <span>A list of input strings</span>,
    example: (
      <CodeBlock language="python">
        {`[
  "What is MLflow?",
  "What is Spark?",
]`}
      </CodeBlock>
    ),
    additional_notes: (
      <span>
        The{" "}
        <Link to="../python_api/mlflow.mdx#mlflow.evaluate()">
          mlflow.evaluate()
        </Link>{" "}
        also accepts a list input.
      </span>
    ),
  },
  {
    data_format: <span>A list of request payload (dictionary).</span>,
    example: (
      <CodeBlock language="python">
        {`[
  {
    "messages": [
      {
        "role": "system", 
        "content": "Please answer."
      },
      {
        "role": "user", 
        "content": "What is MLflow?"
      },
    ],
      "max_tokens": 100,
  },
  # ... more dictionary records
]`}
      </CodeBlock>
    ),
    additional_notes: (
      <span>
        Similarly to Pandas DataFrame input, the dictionary should have the
        correct request format for your model endpoint.
      </span>
    ),
  },
];

export { rows, columns };
