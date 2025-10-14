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
client.chat.completions.create(model="gpt-4o-mini", messages=messages)`,
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
          code: <code>mlflow.langgraph.autolog()</code>,
        }}
      />
    ),
    getCodeSource: () =>
      `from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from typing import Annotated

mlflow.langgraph.autolog()

# Ensure that the "OPENAI_API_KEY" environment variable is set
model = ChatOpenAI(model="gpt-4o-mini")

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
lm = dspy.LM("openai/gpt-4o-mini")
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
llm_config = { "model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"] }
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
    model="claude-3-5-sonnet-20241022",
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

model = "anthropic.claude-3-5-sonnet-20241022-v2:0"
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
litellm.completion(model="gpt-4o-mini", messages=messages)`,
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
client.models.generate_content(model="gemini-1.5-flash", contents="Hello!")`,
  },
  custom: {
    minVersion: '2.14.3',
    getContent: (baseComponentId) => (
      <>
        <Typography.Paragraph css={{ maxWidth: 800 }}>
          <FormattedMessage
            defaultMessage="To manually instrument your own traces, the most convenient method is to use the {code} function decorator. This will cause the inputs and outputs of the function to be captured in the trace."
            description="Description of how to log custom code traces using MLflow. This message is followed by a code example."
            values={{
              code: <code>@mlflow.trace</code>,
            }}
          />
        </Typography.Paragraph>
        <Typography.Paragraph css={{ maxWidth: 800 }}>
          <FormattedMessage
            defaultMessage="For more complex use cases, MLflow also provides granular APIs that can be used to control tracing behavior. For more information, please visit the <a>official documentation</a> on fluent and client APIs for MLflow Tracing."
            description="Explanation of alternative APIs for custom tracing in MLflow. The link leads to the MLflow documentation for the user to learn more."
            values={{
              a: (text: string) => (
                <Typography.Link
                  title="official documentation"
                  componentId={`${baseComponentId}.traces_table.custom_tracing_docs_link`}
                  href="https://mlflow.org/docs/latest/llms/tracing/index.html#tracing-fluent-apis"
                  openInNewTab
                >
                  {text}
                </Typography.Link>
              ),
            }}
          />
        </Typography.Paragraph>
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
