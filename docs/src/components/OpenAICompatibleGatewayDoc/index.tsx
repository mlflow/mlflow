import React from 'react';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Admonition from '@theme/Admonition';
import TilesGrid from '@site/src/components/TilesGrid';
import TileCard from '@site/src/components/TileCard';
import ImageBox from '@site/src/components/ImageBox';
import StepHeader from '@site/src/components/StepHeader';
import ServerSetup from '@site/src/content/setup_server_slim.mdx';
import TabsWrapper from '@site/src/components/TabsWrapper';
import { Users, BookOpen, Scale } from 'lucide-react';
import CodeBlock from '@theme/CodeBlock';
import { getGateway, type OpenAICompatibleGateway } from '@site/src/components/OpenAICompatibleGateways/config';

interface Props {
  gatewayId: string;
}

function formatPythonHeaders(headers: Record<string, string>, comment?: string): string {
  const entries = Object.entries(headers);
  if (entries.length === 0) return '';

  const formatted = entries.map(([key, value]) => {
    const commentStr = comment ? `  # ${comment}` : '';
    return `        "${key}": "${value}",${commentStr}`;
  });

  return `\n    default_headers={\n${formatted.join('\n')}\n    },`;
}

function formatPythonHeadersShort(headers: Record<string, string>): string {
  const entries = Object.entries(headers);
  if (entries.length === 0) return '';

  const formatted = entries.map(([key, value]) => `"${key}": "${value}"`).join(', ');
  return `, default_headers={${formatted}}`;
}

function formatTypeScriptHeaders(headers: Record<string, string>, comment?: string): string {
  const entries = Object.entries(headers);
  if (entries.length === 0) return '';

  const formatted = entries.map(([key, value]) => {
    const commentStr = comment ? ` // ${comment}` : '';
    return `      "${key}": "${value}",${commentStr}`;
  });

  return `\n    defaultHeaders: {\n${formatted.join('\n')}\n    },`;
}

function formatTypeScriptHeadersShort(headers: Record<string, string>): string {
  const entries = Object.entries(headers);
  if (entries.length === 0) return '';

  const formatted = entries.map(([key, value]) => `"${key}": "${value}"`).join(', ');
  return `, defaultHeaders: { ${formatted} }`;
}

export const OpenAICompatibleGatewayDoc: React.FC<Props> = ({ gatewayId }) => {
  const gateway = getGateway(gatewayId);

  if (!gateway) {
    return <div>Gateway "{gatewayId}" not found in configuration.</div>;
  }

  const displayName = gateway.displayName || gateway.name;
  const heroImage = gateway.heroImage || '/images/llms/tracing/basic-openai-trace.png';

  const pythonHeaders = gateway.defaultHeaders?.python || {};
  const tsHeaders = gateway.defaultHeaders?.typescript || {};
  const hasPythonHeaders = Object.keys(pythonHeaders).length > 0;
  const hasTsHeaders = Object.keys(tsHeaders).length > 0;

  return (
    <>
      <div dangerouslySetInnerHTML={{ __html: gateway.description }} />

      <ImageBox src={heroImage} alt={`${displayName} Tracing`} />

      <p>
        Since {displayName} exposes an OpenAI-compatible API, you can use MLflow's OpenAI autolog integration to
        automatically trace all your LLM calls through the gateway.
      </p>

      <h2>Getting Started</h2>

      <Admonition type="tip" title="Prerequisites">
        <div dangerouslySetInnerHTML={{ __html: gateway.prerequisite }} />
      </Admonition>

      <StepHeader number={1} title="Install Dependencies" />

      <TabsWrapper>
        <Tabs groupId="programming-language">
          <TabItem value="python" label="Python" default>
            <CodeBlock language="bash">pip install mlflow openai</CodeBlock>
          </TabItem>
          <TabItem value="typescript" label="TypeScript">
            <CodeBlock language="bash">npm install mlflow-openai openai</CodeBlock>
          </TabItem>
        </Tabs>
      </TabsWrapper>

      <StepHeader number={2} title="Start MLflow Server" />

      <ServerSetup />

      <StepHeader number={3} title="Enable Tracing and Make API Calls" />

      <TabsWrapper>
        <Tabs groupId="programming-language">
          <TabItem value="python" label="Python" default>
            <p>
              Enable tracing with <code>mlflow.openai.autolog()</code> and configure the OpenAI client to use{' '}
              {displayName}'s base URL.
            </p>
            <CodeBlock language="python">
              {`import mlflow
from openai import OpenAI

# Enable auto-tracing for OpenAI
mlflow.openai.autolog()

# Set tracking URI and experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("${gateway.name}")

# Create OpenAI client pointing to ${displayName}
client = OpenAI(
    base_url="${gateway.baseUrl}",
    api_key="${gateway.apiKeyPlaceholder}",${hasPythonHeaders ? formatPythonHeaders(pythonHeaders, gateway.headerComment) : ''}
)

# Make API calls - traces will be captured automatically
response = client.chat.completions.create(
    model="${gateway.sampleModel}",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
)
print(response.choices[0].message.content)`}
            </CodeBlock>
          </TabItem>
          <TabItem value="typescript" label="TypeScript">
            <p>
              Initialize MLflow tracing with <code>init()</code> and wrap the OpenAI client with the{' '}
              <code>tracedOpenAI</code> function.
            </p>
            <CodeBlock language="typescript">
              {`import { init } from "mlflow-tracing";
import { tracedOpenAI } from "mlflow-openai";
import { OpenAI } from "openai";

// Initialize MLflow tracing
init({
  trackingUri: "http://localhost:5000",
  experimentId: "<experiment-id>",
});

// Wrap the OpenAI client pointing to ${displayName}
const client = tracedOpenAI(
  new OpenAI({
    baseURL: "${gateway.baseUrl}",
    apiKey: "${gateway.apiKeyPlaceholder}",${hasTsHeaders ? formatTypeScriptHeaders(tsHeaders, gateway.headerComment) : ''}
  })
);

// Make API calls - traces will be captured automatically
const response = await client.chat.completions.create({
  model: "${gateway.sampleModel}",
  messages: [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "What is the capital of France?" },
  ],
});
console.log(response.choices[0].message.content);`}
            </CodeBlock>
          </TabItem>
        </Tabs>
      </TabsWrapper>

      <StepHeader number={4} title="View Traces in MLflow UI" />

      <p>Open the MLflow UI at http://localhost:5000 to see the traces from your {displayName} API calls.</p>

      <h2>Combining with Manual Tracing</h2>

      <p>
        You can combine auto-tracing with MLflow's manual tracing to create comprehensive traces that include your
        application logic:
      </p>

      <TabsWrapper>
        <Tabs groupId="programming-language">
          <TabItem value="python" label="Python" default>
            <CodeBlock language="python">
              {`import mlflow
from mlflow.entities import SpanType
from openai import OpenAI

mlflow.openai.autolog()

client = OpenAI(
    base_url="${gateway.baseUrl}",
    api_key="${gateway.apiKeyPlaceholder}"${hasPythonHeaders ? formatPythonHeadersShort(pythonHeaders) : ''},
)


@mlflow.trace(span_type=SpanType.CHAIN)
def ask_question(question: str) -> str:
    """A traced function that calls the LLM through ${displayName}."""
    response = client.chat.completions.create(
        model="${gateway.sampleModel}", messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content


# The entire function call and nested LLM call will be traced
answer = ask_question("What is machine learning?")
print(answer)`}
            </CodeBlock>
          </TabItem>
          <TabItem value="typescript" label="TypeScript">
            <CodeBlock language="typescript">
              {`import { init, trace, SpanType } from "mlflow-tracing";
import { tracedOpenAI } from "mlflow-openai";
import { OpenAI } from "openai";

init({
  trackingUri: "http://localhost:5000",
  experimentId: "<experiment-id>",
});

const client = tracedOpenAI(
  new OpenAI({
    baseURL: "${gateway.baseUrl}",
    apiKey: "${gateway.apiKeyPlaceholder}"${hasTsHeaders ? formatTypeScriptHeadersShort(tsHeaders) : ''},
  })
);

// Wrap your function with trace() to create a span
const askQuestion = trace(
  { name: "askQuestion", spanType: SpanType.CHAIN },
  async (question: string): Promise<string> => {
    const response = await client.chat.completions.create({
      model: "${gateway.sampleModel}",
      messages: [{ role: "user", content: question }],
    });
    return response.choices[0].message.content ?? "";
  }
);

// The entire function call and nested LLM call will be traced
const answer = await askQuestion("What is machine learning?");
console.log(answer);`}
            </CodeBlock>
          </TabItem>
        </Tabs>
      </TabsWrapper>

      <h2>Streaming Support</h2>

      <p>MLflow supports tracing streaming responses from {displayName}:</p>

      <TabsWrapper>
        <Tabs groupId="programming-language">
          <TabItem value="python" label="Python" default>
            <CodeBlock language="python">
              {`import mlflow
from openai import OpenAI

mlflow.openai.autolog()

client = OpenAI(
    base_url="${gateway.baseUrl}",
    api_key="${gateway.apiKeyPlaceholder}"${hasPythonHeaders ? formatPythonHeadersShort(pythonHeaders) : ''},
)

stream = client.chat.completions.create(
    model="${gateway.sampleModel}",
    messages=[{"role": "user", "content": "Write a haiku about machine learning."}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")`}
            </CodeBlock>
          </TabItem>
          <TabItem value="typescript" label="TypeScript">
            <CodeBlock language="typescript">
              {`import { init } from "mlflow-tracing";
import { tracedOpenAI } from "mlflow-openai";
import { OpenAI } from "openai";

init({
  trackingUri: "http://localhost:5000",
  experimentId: "<experiment-id>",
});

const client = tracedOpenAI(
  new OpenAI({
    baseURL: "${gateway.baseUrl}",
    apiKey: "${gateway.apiKeyPlaceholder}"${hasTsHeaders ? formatTypeScriptHeadersShort(tsHeaders) : ''},
  })
);

const stream = await client.chat.completions.create({
  model: "${gateway.sampleModel}",
  messages: [{ role: "user", content: "Write a haiku about machine learning." }],
  stream: true,
});

for await (const chunk of stream) {
  if (chunk.choices[0].delta.content) {
    process.stdout.write(chunk.choices[0].delta.content);
  }
}`}
            </CodeBlock>
          </TabItem>
        </Tabs>
      </TabsWrapper>

      <p>MLflow will automatically capture the complete streamed response in the trace.</p>

      <h2>Next Steps</h2>

      <TilesGrid>
        <TileCard
          icon={Users}
          iconSize={48}
          title="Track User Feedback"
          description="Record user feedback on traces for tracking user satisfaction."
          href="/genai/tracing/collect-user-feedback"
          linkText="Learn about feedback →"
          containerHeight={64}
        />
        <TileCard
          icon={BookOpen}
          iconSize={48}
          title="Manage Prompts"
          description="Learn how to manage prompts with MLflow's prompt registry."
          href="/genai/prompt-registry"
          linkText="Manage prompts →"
          containerHeight={64}
        />
        <TileCard
          icon={Scale}
          iconSize={48}
          title="Evaluate Traces"
          description="Evaluate traces with LLM judges to understand and improve your AI application's behavior."
          href="/genai/eval-monitor/running-evaluation/traces"
          linkText="Evaluate traces →"
          containerHeight={64}
        />
      </TilesGrid>
    </>
  );
};

export default OpenAICompatibleGatewayDoc;
