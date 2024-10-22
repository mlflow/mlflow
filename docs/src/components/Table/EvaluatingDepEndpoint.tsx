import CodeBlock from "@theme/CodeBlock";
import Link from "@docusaurus/Link";

export default function EvaluatingDepEndpoint() {
  const format = { messages: [{ role: "user", content: "What is MLflow?" }] };
  return (
    <div>
      <table>
        <colgroup>
          <col span={1} />
          <col span={1} />
          <col span={1} />
        </colgroup>
        <tr>
          <th>Data Format</th>
          <th>Example</th>
          <th>Additional Notes</th>
        </tr>
        <tr>
          <td>A pandas DataFrame with a string column. </td>
          <td>
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
          </td>
          <td>
            For this input format, MLflow will construct the appropriate request
            payload to the model endpoint type. For example, if your model is a
            chat endpoint (`llm/v1/chat`), MLflow will wrap your input string
            with the chat messages format like{""}
            <code>{JSON.stringify(format)}</code>. If you want to customize the
            request payload e.g. including system prompt, please use the next
            format.
          </td>
        </tr>
        <tr>
          <td>A pandas DataFrame with a dictionary column.</td>
          <td>
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
          </td>
          <td>
            In this format, the dictionary should have the correct request
            format for your model endpoint. Please refer to the{" "}
            <Link to="../deployments/index.mdx#standard-query-parameters">
              MLflow Deployments documentation
            </Link>{" "}
            for more information about the request format for different model
            endpoint types.
          </td>
        </tr>
        <tr>
          <td>A list of input strings</td>
          <td>
            <CodeBlock language="python">
              {`[
  "What is MLflow?",
  "What is Spark?",
]`}
            </CodeBlock>
          </td>
          <td>
            The{" "}
            <Link to="../python_api/mlflow.mdx#mlflow.evaluate()">
              mlflow.evaluate()
            </Link>
            also accepts a list input.
          </td>
        </tr>
        <tr>
          <td>A list of request payload (dictionary).</td>
          <td>
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
          </td>
          <td>
            Similarly to Pandas DataFrame input, the dictionary should have the
            correct request format for your model endpoint.
          </td>
        </tr>
      </table>
    </div>
  );
}
