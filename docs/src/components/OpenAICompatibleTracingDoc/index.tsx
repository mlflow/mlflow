import React from 'react';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Link from '@docusaurus/Link';
import TilesGrid from '@site/src/components/TilesGrid';
import TileCard from '@site/src/components/TileCard';
import ImageBox from '@site/src/components/ImageBox';
import StepHeader from '@site/src/components/StepHeader';
import ServerSetup from '@site/src/content/setup_server_slim.mdx';
import TabsWrapper from '@site/src/components/TabsWrapper';
import { Users, BookOpen, Scale } from 'lucide-react';
import CodeBlock from '@theme/CodeBlock';
import { getProvider, type OpenAICompatibleProvider } from '@site/src/components/OpenAICompatibleProviders/config';

interface Props {
  providerId: string;
}

export const OpenAICompatibleTracingDoc: React.FC<Props> = ({ providerId }) => {
  const provider = getProvider(providerId);

  if (!provider) {
    return <div>Provider "{providerId}" not found in configuration.</div>;
  }

  const displayName = provider.displayName || provider.name;
  const codeRefName = provider.codeRefName || provider.name;
  const tsBaseUrl = provider.tsBaseUrl || provider.baseUrl;
  const tsSampleModel = provider.tsSampleModel || provider.sampleModel;

  return (
    <>
      <p>
        <Link to="../../">MLflow Tracing</Link> provides automatic tracing capability for {displayName} models through
        the OpenAI SDK integration. Since {displayName} offers an OpenAI-compatible API format, you can use{' '}
        <code>mlflow.openai.autolog()</code> to trace interactions with {codeRefName} models.
      </p>

      <ImageBox src="/images/llms/tracing/openai-function-calling.png" alt="Tracing via autolog" />

      <p>MLflow trace automatically captures the following information about {codeRefName} calls:</p>

      <ul>
        <li>Prompts and completion responses</li>
        <li>Latencies</li>
        <li>Token usage</li>
        <li>Model name</li>
        <li>
          Additional metadata such as <code>temperature</code>, <code>max_completion_tokens</code>, if specified.
        </li>
        <li>Function calling if returned in the response</li>
        <li>Built-in tools such as web search, file search, computer use, etc.</li>
        <li>Any exception if raised</li>
      </ul>

      <h2>Getting Started</h2>

      <StepHeader number={1} title="Install dependencies" />

      <TabsWrapper>
        <Tabs>
          <TabItem value="python" label="Python" default>
            <CodeBlock language="bash">pip install mlflow openai</CodeBlock>
          </TabItem>
          <TabItem value="typescript" label="JS / TS">
            <CodeBlock language="bash">npm install mlflow-openai openai</CodeBlock>
          </TabItem>
        </Tabs>
      </TabsWrapper>

      <StepHeader number={2} title="Start MLflow server" />

      <ServerSetup />

      <StepHeader number={3} title={`Enable tracing and call ${codeRefName}`} />

      <TabsWrapper>
        <Tabs>
          <TabItem value="python" label="Python" default>
            <CodeBlock language="python">
              {`import openai
import mlflow

# Enable auto-tracing for OpenAI (works with ${codeRefName})
mlflow.openai.autolog()

# Optional: Set a tracking URI and an experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("${codeRefName}")

# Initialize the OpenAI client with ${codeRefName} API endpoint
client = openai.OpenAI(
    base_url="${provider.baseUrl}",
    api_key="${provider.apiKeyPlaceholder}",
)

response = client.chat.completions.create(
    model="${provider.sampleModel}",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
)`}
            </CodeBlock>
          </TabItem>
          <TabItem value="typescript" label="JS / TS">
            <CodeBlock language="typescript">
              {`import { OpenAI } from "openai";
import { tracedOpenAI } from "mlflow-openai";

// Wrap the OpenAI client and point to ${codeRefName} endpoint
const client = tracedOpenAI(
  new OpenAI({
    baseURL: "${tsBaseUrl}",
    apiKey: "${provider.apiKeyPlaceholder}",
  })
);

const response = await client.chat.completions.create({
  model: "${tsSampleModel}",
  messages: [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "What is the capital of France?" },
  ],
  temperature: 0.1,
  max_tokens: 100,
});`}
            </CodeBlock>
          </TabItem>
        </Tabs>
      </TabsWrapper>

      <StepHeader number={4} title="View traces in MLflow UI" />

      <p>
        Browse to your MLflow UI (for example, http://localhost:5000) and open the <code>{codeRefName}</code> experiment
        to see traces for the calls above.
      </p>

      <ImageBox src="/images/llms/tracing/basic-openai-trace.png" alt={`${codeRefName} Tracing`} />

      <p>
        -&gt; View{' '}
        <u>
          <a href="#next-steps">Next Steps</a>
        </u>{' '}
        for learning about more MLflow features like user feedback tracking, prompt management, and evaluation.
      </p>

      <h2>Streaming and Async Support</h2>

      <p>
        MLflow supports tracing for streaming and async {codeRefName} APIs. Visit the{' '}
        <Link to="../openai">OpenAI Tracing documentation</Link> for example code snippets for tracing streaming and
        async calls through OpenAI SDK.
      </p>

      <h2>Combine with frameworks or manual tracing</h2>

      <p>
        The automatic tracing capability in MLflow is designed to work seamlessly with the{' '}
        <Link to="/genai/tracing/app-instrumentation/manual-tracing">Manual Tracing SDK</Link> or multi-framework
        integrations. The examples below show Python (manual span) and JS/TS (manual span) at the same level of
        complexity.
      </p>

      <TabsWrapper>
        <Tabs>
          <TabItem value="python" label="Python" default>
            <CodeBlock language="python">
              {`import json
from openai import OpenAI
import mlflow
from mlflow.entities import SpanType

# Initialize the OpenAI client with ${codeRefName} API endpoint
client = OpenAI(
    base_url="${provider.baseUrl}",
    api_key="${provider.apiKeyPlaceholder}",
)


# Create a parent span for the ${codeRefName} call
@mlflow.trace(span_type=SpanType.CHAIN)
def answer_question(question: str):
    messages = [{"role": "user", "content": question}]
    response = client.chat.completions.create(
        model="${provider.sampleModel}",
        messages=messages,
    )

    # Attach session/user metadata to the trace
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.session": "session-12345",
            "mlflow.trace.user": "user-a",
        }
    )
    return response.choices[0].message.content


answer = answer_question("What is the capital of France?")`}
            </CodeBlock>
          </TabItem>
          <TabItem value="typescript" label="JS / TS">
            <CodeBlock language="typescript">
              {`import * as mlflow from "mlflow-tracing";
import { OpenAI } from "openai";
import { tracedOpenAI } from "mlflow-openai";

mlflow.init({
  trackingUri: "http://localhost:5000",
  experimentId: "<your_experiment_id>",
});

// Wrap the OpenAI client and point to ${codeRefName} endpoint
const client = tracedOpenAI(
  new OpenAI({
    baseURL: "${tsBaseUrl}",
    apiKey: "${provider.apiKeyPlaceholder}",
  })
);

// Create a traced function that wraps the ${codeRefName} call
const answerQuestion = mlflow.trace(
  async (question: string) => {
    const resp = await client.chat.completions.create({
      model: "${tsSampleModel}",
      messages: [{ role: "user", content: question }],
    });
    return resp.choices[0].message?.content;
  },
  { name: "answer-question" }
);

await answerQuestion("What is the capital of France?");`}
            </CodeBlock>
          </TabItem>
        </Tabs>
      </TabsWrapper>

      <p>
        Running either example will produce a trace that includes the {codeRefName} LLM span; the traced function
        creates the parent span automatically.
      </p>

      <ImageBox
        src="/images/llms/tracing/openai-trace-with-manual-span.png"
        alt={`${codeRefName} Tracing with Manual Tracing`}
      />

      <h2 id="next-steps">Next steps</h2>

      <TilesGrid>
        <TileCard
          icon={Users}
          iconSize={48}
          title="Track User Feedback"
          description="Record user feedback on traces for tracking user satisfaction."
          href="/genai/tracing/collect-user-feedback"
          linkText="Learn about feedback ->"
          containerHeight={64}
        />
        <TileCard
          icon={BookOpen}
          iconSize={48}
          title="Manage Prompts"
          description="Learn how to manage prompts with MLflow's prompt registry."
          href="/genai/prompt-registry"
          linkText="Manage prompts ->"
          containerHeight={64}
        />
        <TileCard
          icon={Scale}
          iconSize={48}
          title="Evaluate Traces"
          description="Evaluate traces with LLM judges to understand and improve your AI application's behavior."
          href="/genai/eval-monitor/running-evaluation/traces"
          linkText="Evaluate traces ->"
          containerHeight={64}
        />
      </TilesGrid>
    </>
  );
};

export default OpenAICompatibleTracingDoc;
