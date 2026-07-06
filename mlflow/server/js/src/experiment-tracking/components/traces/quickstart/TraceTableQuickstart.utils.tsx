import { Typography } from '@databricks/design-system';
import React from 'react';
import { FormattedMessage } from 'react-intl';

export type QUICKSTART_FLAVOR =
  | 'openai'
  | 'langchain'
  | 'langgraph'
  | 'llama_index'
  | 'dspy'
  | 'crewai'
  | 'autogen'
  | 'anthropic'
  | 'bedrock'
  | 'litellm'
  | 'gemini'
  | 'custom';

export const PYTHON_FRAMEWORK_OPTIONS: { key: QUICKSTART_FLAVOR; label: string }[] = [
  { key: 'openai', label: 'OpenAI' },
  { key: 'anthropic', label: 'Anthropic' },
  { key: 'langchain', label: 'LangChain' },
  { key: 'langgraph', label: 'LangGraph' },
  { key: 'dspy', label: 'DSPy' },
  { key: 'litellm', label: 'LiteLLM' },
  { key: 'gemini', label: 'Gemini' },
  { key: 'bedrock', label: 'Bedrock' },
  { key: 'crewai', label: 'CrewAI' },
  { key: 'custom', label: 'Custom' },
];

export const TS_FRAMEWORK_OPTIONS: { key: string; label: string }[] = [
  { key: 'openai', label: 'OpenAI' },
  { key: 'anthropic', label: 'Anthropic' },
  { key: 'gemini', label: 'Gemini' },
  { key: 'vercel', label: 'Vercel AI SDK' },
  { key: 'custom', label: 'Custom' },
];

export const getPythonConnectCode = (trackingUri: string, experimentName: string) =>
  `import mlflow

# Specify the tracking server URI, e.g. http://localhost:5000
mlflow.set_tracking_uri("${trackingUri}")
mlflow.set_experiment("${experimentName}")`;

export const TS_INSTALL_CODE = 'npm install @mlflow/core';

export const OTEL_INSTALL_CODE =
  'pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http';

export const getOtelEnvCode = (trackingUri: string, experimentId: string) =>
  `export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=${trackingUri}/v1/traces
export OTEL_EXPORTER_OTLP_TRACES_HEADERS=x-mlflow-experiment-id=${experimentId}`;

export const OTEL_INSTRUMENT_CODE = `from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Endpoint and headers are picked up from the env vars set above.
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

# Instrument your application using any OpenTelemetry SDK or auto-instrumentation.
tracer = trace.get_tracer(__name__)
with tracer.start_as_current_span("my-app") as span:
    span.set_attribute("input", "hello")
    # ... your application logic ...`;

export const getTsConnectCode = (trackingUri: string, experimentId: string) =>
  `import * as mlflow from '@mlflow/core';

mlflow.init({
  // Specify the tracking server URI, e.g. http://localhost:5000
  trackingUri: '${trackingUri}',
  experimentId: '${experimentId}',
});`;

export const getTsFrameworkCode = (trackingUri: string, experimentId: string) =>
  ({
    openai: {
      install: 'npm install @mlflow/openai openai',
      code: `import { OpenAI } from 'openai';
import { tracedOpenAI } from '@mlflow/openai';

// Wrap the OpenAI client with the tracedOpenAI function
const client = tracedOpenAI(new OpenAI());

// Invoke the client as usual
const response = await client.chat.completions.create({
  model: 'gpt-5-nano',
  messages: [
    { role: 'system', content: 'You are a helpful weather assistant.' },
    { role: 'user', content: "What's the weather like in Seattle?" },
  ],
});`,
    },
    anthropic: {
      install: 'npm install @mlflow/anthropic @anthropic-ai/sdk',
      code: `import Anthropic from '@anthropic-ai/sdk';
import { tracedAnthropic } from '@mlflow/anthropic';

// Wrap the Anthropic client with the tracedAnthropic function
const client = tracedAnthropic(new Anthropic());

// Invoke the client as usual
const message = await client.messages.create({
  model: 'claude-sonnet-4-5',
  max_tokens: 1024,
  messages: [{ role: 'user', content: 'Hello, Claude' }],
});`,
    },
    gemini: {
      install: 'npm install @mlflow/gemini @google/genai',
      code: `import { GoogleGenAI } from '@google/genai';
import { tracedGemini } from '@mlflow/gemini';

// Wrap the GoogleGenAI client with the tracedGemini function
const client = tracedGemini(new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY }));

// Invoke the client as usual
const response = await client.models.generateContent({
  model: 'gemini-2.5-flash',
  contents: 'What is the capital of France?',
});`,
    },
    vercel: {
      install:
        'npm install @mlflow/vercel ai @ai-sdk/openai @opentelemetry/exporter-trace-otlp-proto @opentelemetry/sdk-trace-node',
      code: `import { MLflowSpanProcessor } from '@mlflow/vercel';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';

const provider = new NodeTracerProvider({
  spanProcessors: [
    new MLflowSpanProcessor(
      new OTLPTraceExporter({
        url: '${trackingUri}/api/2.0/otel/v1/traces',
        headers: { 'x-mlflow-experiment-id': '${experimentId}' },
      }),
    ),
  ],
});
provider.register();

// Pass experimental_telemetry: { isEnabled: true } to record traces
const { text } = await generateText({
  model: openai('gpt-5-nano'),
  prompt: 'What is MLflow?',
  experimental_telemetry: { isEnabled: true },
});

console.log(text);`,
    },
    custom: {
      install: '',
      code: `// Wrap any async function with mlflow.trace for automatic tracing
const processRequest = mlflow.trace(
  async (userInput: string) => {
    // Your application logic here
    const response = await callYourAPI(userInput);

    // Log custom attributes
    mlflow.setAttributes({
      user_input_length: userInput.length,
      response_type: typeof response
    });

    return response;
  },
  { name: 'process-user-request' }
);

// Use the traced function
const result = await processRequest("Hello, MLflow!");
console.log("Processed:", result);`,
    },
  }) satisfies Record<string, { install: string; code: string }>;

export const QUICKSTART_CONTENT: Record<
  QUICKSTART_FLAVOR,
  {
    minVersion: string;
    getContent: (baseComponentId?: string) => React.ReactNode;
    getCodeSource: () => string;
  }
> = {
  openai: {
    minVersion: '2.15.1',
    getContent: () => (
      <FormattedMessage
        defaultMessage="Automatically log traces for OpenAI API calls by calling the {code} function. For example:"
        description="Description of how to log traces for the OpenAI package using MLflow autologging. This message is followed by a code example."
        values={{
          code: <code>mlflow.openai.autolog()</code>,
        }}
      />
    ),
    getCodeSource: () =>
      `from openai import OpenAI

mlflow.openai.autolog()

# Ensure that the "OPENAI_API_KEY" environment variable is set
client = OpenAI()

messages = [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Hello!"}
]

# Inputs and outputs of the API request will be logged in a trace
client.chat.completions.create(model="gpt-5-nano", messages=messages)`,
  },
  langchain: {
    // the autologging integration was really introduced in
    // 2.14.0, but it does not support newer versions of langchain
    // so effectively that version will not work with the code snippet
    minVersion: '2.17.2',
    getContent: () => (
      <FormattedMessage
        defaultMessage="Automatically log traces for LangChain or LangGraph invocations by calling the {code} function. For example:"
        description="Description of how to log traces for the LangChain/LangGraph package using MLflow autologging. This message is followed by a code example."
        values={{
          code: <code>mlflow.langchain.autolog()</code>,
        }}
      />
    ),
    getCodeSource: () =>
      `from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

mlflow.langchain.autolog()

# Ensure that the "OPENAI_API_KEY" environment variable is set
llm = OpenAI()
prompt = PromptTemplate.from_template("Answer the following question: {question}")
chain = prompt | llm

# Invoking the chain will cause a trace to be logged
chain.invoke("What is MLflow?")`,
  },
  langgraph: {
    minVersion: '2.19.0',
    getContent: () => (
      <FormattedMessage
        defaultMessage="Automatically log traces for LangGraph workflows by calling the {code} function. For example:"
        description="Description of how to log traces for the LangGraph package using MLflow autologging. This message is followed by a code example."
        values={{
          code: <code>mlflow.langchain.autolog()</code>,
        }}
      />
    ),
    getCodeSource: () =>
      `from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from typing import Annotated

mlflow.langchain.autolog()

# Ensure that the "OPENAI_API_KEY" environment variable is set
model = ChatOpenAI(model="gpt-5-nano")

# Define a minimal LangGraph workflow
class GraphState(dict):
    input: Annotated[str, "input"]

def call_model(state: GraphState) -> GraphState:
    response = model.invoke(state["input"])
    return {"input": state["input"], "response": response.content}

graph = StateGraph(GraphState)
graph.add_node("model", call_model)
graph.set_entry_point("model")
app = graph.compile()

# Executing the graph will log the steps as a trace
app.invoke({"input": "Say hello to MLflow."})`,
  },
  llama_index: {
    minVersion: '2.15.1',
    getContent: () => (
      <FormattedMessage
        defaultMessage="Automatically log traces for LlamaIndex queries by calling the {code} function. For example:"
        description="Description of how to log traces for the LlamaIndex package using MLflow autologging. This message is followed by a code example."
        values={{
          code: <code>mlflow.llama_index.autolog()</code>,
        }}
      />
    ),
    getCodeSource: () =>
      `from llama_index.core import Document, VectorStoreIndex

mlflow.llama_index.autolog()

# Ensure that the "OPENAI_API_KEY" environment variable is set
index = VectorStoreIndex.from_documents([Document.example()])
query_engine = index.as_query_engine()

# Querying the engine will cause a trace to be logged
query_engine.query("What is LlamaIndex?")`,
  },
  dspy: {
    minVersion: '2.18.0',
    getContent: () => (
      <FormattedMessage
        defaultMessage="Automatically log traces for DSPy executions by calling the {code} function. For example:"
        description="Description of how to log traces for the DSPy package using MLflow autologging. This message is followed by a code example."
        values={{
          code: <code>mlflow.dspy.autolog()</code>,
        }}
      />
    ),
    getCodeSource: () =>
      `import dspy

mlflow.dspy.autolog()

# Configure the LLM to use. Please ensure that
# the OPENAI_API_KEY environment variable is set
lm = dspy.LM("openai/gpt-5-nano")
dspy.configure(lm=lm)

# Define a simple chain-of-thought model and run it
math = dspy.ChainOfThought("question -> answer: float")
question = "Two dice are tossed. What is the probability that the sum equals two?"

# All intermediate outputs from the execution will be logged
math(question=question)`,
  },
  crewai: {
    minVersion: '2.19.0',
    getContent: () => (
      <FormattedMessage
        defaultMessage="Automatically log traces for CrewAI executions by calling the {code} function. For example:"
        description="Description of how to log traces for the CrewAI package using MLflow autologging. This message is followed by a code example."
        values={{
          code: <code>mlflow.crewai.autolog()</code>,
        }}
      />
    ),
    getCodeSource: () => `from crewai import Agent, Crew, Process, Task

mlflow.crewai.autolog()

city_selection_agent = Agent(
    role="City selection expert",
    goal="Select the best city based on weather, season, and prices",
    backstory="An expert in analyzing travel data to pick ideal destinations",
    allow_delegation=True,
    verbose=True,
)

local_expert = Agent(
    role="Local expert",
    goal="Provide the best insights about the selected city",
    backstory="A local guide with extensive information about the city",
    verbose=True,
)
  
plan_trip = Task(
    name="Plan a trip",
    description="""Plan a trip to a city based on weather, prices, and best local attractions. 
    Please consult with a local expert when researching things to do.""",
    expected_output="A short summary of the trip destination and key things to do",
    agent=city_selection_agent,
)

crew = Crew(
  agents=[
    city_selection_agent,
    local_expert,
  ],
  tasks=[plan_trip],
  process=Process.sequential
)

# Ensure the "OPENAI_API_KEY" environment variable is set
# before kicking off the crew. All intermediate agent outputs
# will be logged in the resulting trace.
crew.kickoff()`,
  },
  autogen: {
    minVersion: '2.16.2',
    getContent: () => (
      <FormattedMessage
        defaultMessage="Automatically log traces for AutoGen conversations by calling the {code} function. For example:"
        description="Description of how to log traces for the AutoGen package using MLflow autologging. This message is followed by a code example."
        values={{
          code: <code>mlflow.autogen.autolog()</code>,
        }}
      />
    ),
    getCodeSource: () =>
      `import os
from autogen import AssistantAgent, UserProxyAgent

mlflow.autogen.autolog()

# Ensure that the "OPENAI_API_KEY" environment variable is set
llm_config = { "model": "gpt-5-nano", "api_key": os.environ["OPENAI_API_KEY"] }
assistant = AssistantAgent("assistant", llm_config = llm_config)
user_proxy = UserProxyAgent("user_proxy", code_execution_config = False)

# All intermediate executions within the chat session will be logged
user_proxy.initiate_chat(assistant, message = "What is MLflow?", max_turns = 1)`,
  },
  anthropic: {
    minVersion: '2.19.0',
    getContent: () => (
      <FormattedMessage
        defaultMessage="Automatically log traces for Anthropic API calls by calling the {code} function. For example:"
        description="Description of how to log traces for the Anthropic package using MLflow autologging. This message is followed by a code example."
        values={{
          code: <code>mlflow.anthropic.autolog()</code>,
        }}
      />
    ),
    getCodeSource: () => `import os
import anthropic

# Enable auto-tracing for Anthropic
mlflow.anthropic.autolog()

# Configure your API key (please ensure that the "ANTHROPIC_API_KEY" environment variable is set)
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Inputs and outputs of API calls will be logged as a trace
message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"},
    ],
)`,
  },
  bedrock: {
    minVersion: '2.20.0',
    getContent: () => (
      <FormattedMessage
        defaultMessage="Automatically log traces for Bedrock conversations by calling the {code} function. For example:"
        description="Description of how to log traces for the Bedrock package using MLflow autologging. This message is followed by a code example."
        values={{
          code: <code>mlflow.bedrock.autolog()</code>,
        }}
      />
    ),
    getCodeSource: () => `import boto3

mlflow.bedrock.autolog()

# Ensure that your boto3 client has the necessary auth information
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="<REPLACE_WITH_YOUR_AWS_REGION>",
)

model = "anthropic.claude-sonnet-4-5-20250929-v1:0"
messages = [{ "role": "user", "content": [{"text": "Hello!"}]}]

# All intermediate executions within the chat session will be logged
bedrock.converse(modelId=model, messages=messages)`,
  },
  litellm: {
    minVersion: '2.18.0',
    getContent: () => (
      <FormattedMessage
        defaultMessage="Automatically log traces for LiteLLM API calls by calling the {code} function. For example:"
        description="Description of how to log traces for the LiteLLM package using MLflow autologging. This message is followed by a code example."
        values={{
          code: <code>mlflow.litellm.autolog()</code>,
        }}
      />
    ),
    getCodeSource: () => `import litellm

mlflow.litellm.autolog()

# Ensure that the "OPENAI_API_KEY" environment variable is set
messages = [{"role": "user", "content": "Hello!"}]

# Inputs and outputs of the API request will be logged in a trace
litellm.completion(model="gpt-5-nano", messages=messages)`,
  },
  gemini: {
    minVersion: '2.18.0',
    getContent: () => (
      <FormattedMessage
        defaultMessage="Automatically log traces for Gemini conversations by calling the {code} function. For example:"
        description="Description of how to log traces for API calls to Google's Gemini API using MLflow autologging. This message is followed by a code example."
        values={{
          code: <code>mlflow.gemini.autolog()</code>,
        }}
      />
    ),
    getCodeSource: () => `import google.genai as genai

mlflow.gemini.autolog()

# Replace "GEMINI_API_KEY" with your API key
client = genai.Client(api_key="GEMINI_API_KEY")

# Inputs and outputs of the API request will be logged in a trace
client.models.generate_content(model="gemini-2.5-flash", contents="Hello!")`,
  },
  custom: {
    minVersion: '2.14.3',
    getContent: (baseComponentId) => (
      <>
        <FormattedMessage
          defaultMessage="To manually instrument your own traces, the most convenient method is to use the {code} function decorator. This will cause the inputs and outputs of the function to be captured in the trace. For more information, please visit the <a>official documentation</a> for manual tracing."
          description="Description of how to log custom code traces using MLflow. This message is followed by a code example. The link leads to the MLflow documentation for the user to learn more."
          values={{
            code: <code>@mlflow.trace</code>,
            a: (text: string) => (
              <Typography.Link
                title="official documentation"
                componentId="codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_components_traces_quickstart_tracetablequickstart.utils_366"
                href="https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/manual-tracing/"
                openInNewTab
              >
                {text}
              </Typography.Link>
            ),
          }}
        />
        <br />
      </>
    ),
    getCodeSource: () =>
      `@mlflow.trace
def foo(a):
    return a + bar(a)

# Various attributes can be passed to the decorator
# to modify the information contained in the span
@mlflow.trace(name = "custom_name", attributes = { "key": "value" })
def bar(b):
    return b + 1

# Invoking the traced function will cause a trace to be logged
foo(1)`,
  },
};
