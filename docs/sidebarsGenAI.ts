import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const sidebarsGenAI: SidebarsConfig = {
  genAISidebar: [
    {
      type: 'doc',
      id: 'index',
      className: 'sidebar-top-level-category',
    },
    {
      type: 'category',
      label: 'MLflow 3.0',
      className: 'sidebar-top-level-category',
      items: [
        {
          type: 'doc',
          id: 'mlflow-3/deep-learning',
        },
        {
          type: 'doc',
          id: 'mlflow-3/genai-agent',
        },
        {
          type: 'doc',
          id: 'mlflow-3/breaking-changes',
        },
        {
          type: 'doc',
          id: 'mlflow-3/faqs',
        },
      ],
      link: {
        type: 'doc',
        id: 'mlflow-3/index',
      },
    },
    {
      type: 'category',
      label: 'Getting Started',
      className: 'sidebar-top-level-category',
      items: [
        {
          type: 'doc',
          id: 'getting-started/connect-environment',
          label: 'Set Up MLflow',
        },
        {
          type: 'link',
          href: '/genai/tracing/quickstart/python-openai',
          label: 'Tracing GenAI Apps',
        },
        {
          type: 'doc',
          id: 'eval-monitor/quickstart',
          label: 'Evaluate LLMs and Agents',
        },
        {
          type: 'doc',
          id: 'getting-started/databricks-trial/index',
        },
      ],
      link: {
        type: 'doc',
        id: 'getting-started/index',
      },
    },
    {
      type: 'category',
      label: 'Tracing (Observability)',
      className: 'sidebar-top-level-category',
      items: [
        {
          type: 'category',
          label: 'Quickstart',
          items: [
            {
              type: 'doc',
              id: 'tracing/quickstart/python-openai',
              label: 'Getting Started (Python)',
            },
            {
              type: 'doc',
              id: 'tracing/quickstart/typescript-openai',
              label: 'Getting Started (TS/JS)',
            },
          ],
        },
        {
          type: 'category',
          label: 'How to Trace Your App/Agents',
          items: [
            {
              type: 'doc',
              id: 'tracing/app-instrumentation/automatic',
            },
            {
              type: 'doc',
              id: 'tracing/app-instrumentation/manual-tracing',
            },
            {
              type: 'doc',
              id: 'tracing/app-instrumentation/typescript-sdk',
            },
            {
              type: 'doc',
              id: 'tracing/app-instrumentation/opentelemetry',
            },
          ],
          link: {
            type: 'doc',
            id: 'tracing/app-instrumentation/index',
          },
        },
        {
          type: 'category',
          label: 'Integrations',
          items: [
            {
              type: 'category',
              label: 'Frameworks',
              items: [
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/ag2',
                  label: 'AG2',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/agno',
                  label: 'Agno',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/autogen',
                  label: 'AutoGen',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/crewai',
                  label: 'CrewAI',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/dspy',
                  label: 'DSPy',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/google-adk',
                  label: 'Google ADK',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/haystack',
                  label: 'Haystack',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/langchain',
                  label: 'LangChain',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/langgraph',
                  label: 'LangGraph',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/llama_index',
                  label: 'LlamaIndex',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/mastra',
                  label: 'Mastra',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/microsoft-agent-framework',
                  label: 'Microsoft Agent Framework',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/openai-agent',
                  label: 'OpenAI Agent',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/pydantic_ai',
                  label: 'PydanticAI',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/semantic_kernel',
                  label: 'Semantic Kernel',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/smolagents',
                  label: 'Smolagents',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/strands',
                  label: 'Strands Agents SDK',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/txtai',
                  label: 'Txtai',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/vercelai',
                  label: 'Vercel AI SDK',
                },
              ],
            },
            {
              type: 'category',
              label: 'Model Providers',
              items: [
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/anthropic',
                  label: 'Anthropic',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/bedrock',
                  label: 'AWS Bedrock',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/deepseek',
                  label: 'DeepSeek',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/fireworksai',
                  label: 'FireworksAI',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/gemini',
                  label: 'Gemini',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/groq',
                  label: 'Groq',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/litellm',
                  label: 'LiteLLM',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/mistral',
                  label: 'Mistral',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/ollama',
                  label: 'Ollama',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/openai',
                  label: 'OpenAI',
                },
              ],
            },
            {
              type: 'category',
              label: 'Tools',
              items: [
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/claude_code',
                  label: 'Claude Code',
                },
                {
                  type: 'doc',
                  id: 'tracing/integrations/listing/instructor',
                  label: 'Instructor',
                },
              ],
            },
            {
              type: 'doc',
              id: 'tracing/integrations/contribute',
              label: 'Add New Integration',
            },
          ],
          link: {
            type: 'doc',
            id: 'tracing/integrations/index',
          },
        },
        {
          type: 'category',
          label: 'Guides',
          items: [
            {
              type: 'doc',
              id: 'tracing/track-users-sessions/index',
              label: 'Track Users and Sessions',
            },
            {
              type: 'doc',
              id: 'tracing/track-environments-context/index',
              label: 'Track App Versions and Environments',
            },
            {
              type: 'doc',
              id: 'tracing/collect-user-feedback/index',
              label: 'User Feedback Collection',
            },
            {
              type: 'doc',
              id: 'tracing/attach-tags/index',
              label: 'Trace Tagging',
            },
            {
              type: 'doc',
              id: 'tracing/observe-with-traces/delete-traces',
            },
            {
              type: 'doc',
              id: 'tracing/observe-with-traces/masking',
              label: 'Redacting Sensitive Data',
            },
            {
              type: 'doc',
              id: 'tracing/lightweight-sdk',
              label: 'Lightweight Tracing SDK Optimized for Production Usage',
            },
          ],
        },

        {
          type: 'category',
          label: 'Viewing & Searching Traces',
          items: [
            {
              type: 'doc',
              id: 'tracing/observe-with-traces/ui',
            },
            {
              type: 'doc',
              id: 'tracing/search-traces',
              label: 'Searching for Traces',
            },
          ],
        },
        {
          type: 'category',
          label: 'OpenTelemetry',
          items: [
            {
              type: 'doc',
              id: 'tracing/opentelemetry/index',
              label: 'Overview',
            },
            {
              type: 'doc',
              id: 'tracing/opentelemetry/ingest',
            },
            {
              type: 'doc',
              id: 'tracing/opentelemetry/export',
            },
          ],
        },
        {
          type: 'doc',
          id: 'tracing/prod-tracing',
          label: 'Production Tracing',
        },
        {
          type: 'doc',
          id: 'tracing/faq',
          label: 'Tracing FAQ',
        },
      ],
      link: {
        type: 'doc',
        id: 'tracing/index',
      },
    },
    {
      type: 'category',
      label: 'Evaluate & Monitor',
      className: 'sidebar-top-level-category',
      items: [
        {
          type: 'doc',
          id: 'eval-monitor/quickstart',
          label: 'Quickstart',
        },
        {
          type: 'category',
          label: 'Running Evaluations',
          items: [
            {
              type: 'doc',
              id: 'eval-monitor/running-evaluation/prompts',
              label: 'Evaluate Prompts',
            },
            {
              type: 'doc',
              id: 'eval-monitor/running-evaluation/agents',
              label: 'Evaluate Agents',
            },
            {
              type: 'doc',
              id: 'eval-monitor/running-evaluation/traces',
              label: 'Evaluate Traces',
            },
          ],
        },
        {
          type: 'category',
          label: 'Scorers',
          items: [
            {
              type: 'doc',
              id: 'eval-monitor/scorers/index',
              label: 'What is a Scorer?',
            },
            {
              type: 'category',
              label: 'Supported Scorers',
              items: [
                {
                  type: 'doc',
                  id: 'eval-monitor/scorers/llm-judge/predefined',
                  label: 'Predefined Scorers',
                },
                {
                  type: 'category',
                  label: 'LLM-as-a-Judge',
                  items: [
                    {
                      type: 'doc',
                      id: 'eval-monitor/scorers/llm-judge/index',
                      label: 'Overview',
                    },
                    {
                      type: 'doc',
                      id: 'eval-monitor/scorers/llm-judge/make-judge',
                      label: 'Template-based',
                    },
                    {
                      type: 'doc',
                      id: 'eval-monitor/scorers/llm-judge/guidelines',
                      label: 'Guidelines-based',
                    },
                  ],
                  collapsed: false,
                },
                {
                  type: 'doc',
                  id: 'eval-monitor/scorers/llm-judge/agentic-overview',
                  label: 'Agent-as-a-Judge',
                },
                {
                  type: 'doc',
                  id: 'eval-monitor/scorers/custom',
                  label: 'Code-based Scorers',
                },
              ],
            },
            {
              type: 'doc',
              id: 'eval-monitor/scorers/llm-judge/alignment',
              label: 'Align with Human Feedback',
            },
            {
              type: 'doc',
              id: 'eval-monitor/scorers/versioning',
              label: 'Versioning Scorers',
            },
          ],
        },
        {
          type: 'category',
          label: 'Evaluation Datasets',
          items: [
            {
              type: 'doc',
              id: 'datasets/end-to-end-workflow',
              label: 'End-to-End Workflow',
            },
            {
              type: 'doc',
              id: 'datasets/sdk-guide',
              label: 'SDK Guide',
            },
          ],
          link: {
            type: 'doc',
            id: 'datasets/index',
          },
        },
        {
          type: 'category',
          label: 'Annotation',
          items: [
            {
              type: 'doc',
              id: 'assessments/feedback',
              label: 'Human Feedback',
            },
            {
              type: 'doc',
              id: 'assessments/expectations',
              label: 'Annotating Ground Truth',
            },
          ],
        },
        {
          type: 'category',
          label: 'AI Insights',
          items: [
            {
              type: 'doc',
              id: 'eval-monitor/ai-insights/ai-issue-discovery',
              label: 'AI Issue Discovery',
            },
          ],
        },
        {
          type: 'doc',
          id: 'eval-monitor/legacy-llm-evaluation',
          label: 'Migrating from MLflow 2 LLM Evaluation',
        },
        {
          type: 'doc',
          id: 'eval-monitor/faq',
          label: 'FAQ',
        },
      ],
      link: {
        type: 'doc',
        id: 'eval-monitor/index',
      },
    },
    {
      type: 'category',
      label: 'Prompt Management',
      className: 'sidebar-top-level-category',
      items: [
        {
          type: 'doc',
          id: 'prompt-registry/create-and-edit-prompts',
        },
        {
          type: 'doc',
          id: 'prompt-registry/evaluate-prompts',
        },
        {
          type: 'doc',
          id: 'prompt-registry/manage-prompt-lifecycles-with-aliases',
        },
        {
          type: 'doc',
          id: 'prompt-registry/use-prompts-in-apps',
        },
        {
          type: 'doc',
          id: 'prompt-registry/log-with-model',
        },
        {
          type: 'doc',
          id: 'prompt-registry/structured-output',
        },
        {
          type: 'category',
          label: 'Optimize Prompts',
          link: {
            type: 'doc',
            id: 'prompt-registry/optimize-prompts',
          },
          items: [
            {
              type: 'doc',
              id: 'prompt-registry/optimize-prompts/openai-agent-optimization',
            },
          ],
        },
        {
          type: 'doc',
          id: 'prompt-registry/rewrite-prompts',
        },
        {
          type: 'doc',
          id: 'prompt-registry/prompt-engineering',
          label: 'Prompt Engineering UI',
        },
      ],
      link: {
        type: 'doc',
        id: 'prompt-registry/index',
      },
    },
    {
      type: 'category',
      label: 'Version Tracking',
      className: 'sidebar-top-level-category',
      items: [
        {
          type: 'doc',
          id: 'version-tracking/quickstart',
          label: 'Quickstart',
        },
        {
          type: 'category',
          label: 'Guides',
          items: [
            {
              type: 'doc',
              id: 'version-tracking/track-application-versions-with-mlflow',
            },
            {
              type: 'doc',
              id: 'version-tracking/compare-app-versions',
            },
            {
              type: 'category',
              label: 'App Packaging & Deployment',
              items: [
                {
                  type: 'category',
                  label: 'OpenAI',
                  items: [
                    {
                      type: 'doc',
                      id: 'flavors/openai/guide/index',
                      label: 'Guide',
                    },
                    {
                      type: 'doc',
                      id: 'flavors/openai/autologging/index',
                      label: 'Autologging Support',
                    },
                    {
                      type: 'category',
                      label: 'Tutorials',
                      items: [
                        {
                          type: 'doc',
                          id: 'flavors/openai/notebooks/openai-quickstart-ipynb',
                          label: 'OpenAI Quickstart',
                        },
                        {
                          type: 'doc',
                          id: 'flavors/openai/notebooks/openai-chat-completions-ipynb',
                          label: 'Chat Completions with OpenAI',
                        },
                        {
                          type: 'doc',
                          id: 'flavors/openai/notebooks/openai-code-helper-ipynb',
                          label: 'Building a Code Assistant with OpenAI & MLflow',
                        },
                        {
                          type: 'doc',
                          id: 'flavors/openai/notebooks/openai-embeddings-generation-ipynb',
                          label: 'Embeddings Support with OpenAI in MLflow',
                        },
                      ],
                      link: {
                        type: 'doc',
                        id: 'flavors/openai/notebooks/index',
                      },
                    },
                  ],
                  link: {
                    type: 'doc',
                    id: 'flavors/openai/index',
                  },
                },
                {
                  type: 'category',
                  label: 'DSPy',
                  items: [
                    {
                      type: 'doc',
                      id: 'flavors/dspy/notebooks/dspy_quickstart-ipynb',
                      label: 'DSPy Quickstart',
                    },
                    {
                      type: 'doc',
                      id: 'flavors/dspy/optimizer',
                      label: 'Using DSPy Optimizers',
                    },
                  ],
                  link: {
                    type: 'doc',
                    id: 'flavors/dspy/index',
                  },
                },
                {
                  type: 'category',
                  label: 'LangChain',
                  items: [
                    {
                      type: 'doc',
                      id: 'flavors/langchain/guide/index',
                      label: 'Guide to using LangChain with MLflow',
                    },
                    {
                      type: 'doc',
                      id: 'flavors/langchain/notebooks/langchain-quickstart-ipynb',
                      label: 'LangChain Quickstart',
                    },
                    {
                      type: 'doc',
                      id: 'flavors/langchain/notebooks/langchain-retriever-ipynb',
                      label: 'Retrievers with LangChain',
                    },
                  ],
                  link: {
                    type: 'doc',
                    id: 'flavors/langchain/index',
                  },
                },
                {
                  type: 'category',
                  label: 'LlamaIndex',
                  items: [
                    {
                      type: 'doc',
                      id: 'flavors/llama-index/notebooks/llama_index_quickstart-ipynb',
                      label: 'LlamaIndex Quickstart',
                    },
                    {
                      type: 'doc',
                      id: 'flavors/llama-index/notebooks/llama_index_workflow_tutorial-ipynb',
                      label: 'Agents with LlamaIndex',
                    },
                  ],
                  link: {
                    type: 'doc',
                    id: 'flavors/llama-index/index',
                  },
                },
                {
                  type: 'category',
                  label: 'Custom Applications',
                  items: [
                    {
                      type: 'doc',
                      id: 'flavors/custom-pyfunc-for-llms/notebooks/custom-pyfunc-advanced-llm-ipynb',
                      label: 'Custom App Development Guide',
                    },
                  ],
                  link: {
                    type: 'doc',
                    id: 'flavors/custom-pyfunc-for-llms/index',
                  },
                },
                {
                  type: 'doc',
                  id: 'flavors/chat-model-intro/index',
                  label: 'Intro to ChatModel',
                },
                {
                  type: 'category',
                  label: 'Building with ChatModel',
                  items: [
                    {
                      type: 'doc',
                      id: 'flavors/chat-model-guide/chat-model-tool-calling-ipynb',
                      label: 'ChatModel Tool Calling Example',
                    },
                  ],
                  link: {
                    type: 'doc',
                    id: 'flavors/chat-model-guide/index',
                  },
                },
                {
                  type: 'doc',
                  id: 'flavors/responses-agent-intro',
                  label: 'Building with ResponsesAgent',
                },
              ],
              link: {
                type: 'doc',
                id: 'flavors/index',
              },
            },
          ],
        },
      ],
      link: {
        type: 'doc',
        id: 'version-tracking/index',
      },
    },
    {
      type: 'category',
      label: 'MCP',
      className: 'sidebar-top-level-category',
      items: [
        {
          type: 'doc',
          id: 'mcp/index',
        },
      ],
    },
    {
      type: 'category',
      label: 'Model Serving',
      className: 'sidebar-top-level-category',
      items: [
        {
          type: 'doc',
          id: 'serving/agent-server',
          label: 'Agent Server',
        },
        {
          type: 'doc',
          id: 'serving/responses-agent',
          label: 'Responses Agent',
        },
        {
          type: 'doc',
          id: 'serving/custom-apps',
          label: 'Custom Apps',
        },
      ],
      link: {
        type: 'doc',
        id: 'serving/index',
      },
    },
    {
      type: 'category',
      label: 'Governance',
      className: 'sidebar-top-level-category',
      items: [
        {
          type: 'category',
          label: 'AI Gateway',
          items: [
            {
              type: 'doc',
              id: 'governance/ai-gateway/setup',
              label: 'Setup',
            },
            {
              type: 'doc',
              id: 'governance/ai-gateway/configuration',
              label: 'Configuration',
            },
            {
              type: 'doc',
              id: 'governance/ai-gateway/usage',
              label: 'Usage',
            },
            {
              type: 'doc',
              id: 'governance/ai-gateway/integration',
              label: 'Integration',
            },
            {
              type: 'category',
              label: 'Guides',
              items: [
                {
                  type: 'doc',
                  id: 'governance/ai-gateway/guides/step1-create-deployments/index',
                  label: 'Setup the AI Gateway',
                },
                {
                  type: 'doc',
                  id: 'governance/ai-gateway/guides/step2-query-deployments/index',
                  label: 'Use the AI Gateway',
                },
              ],
              link: {
                type: 'doc',
                id: 'governance/ai-gateway/guides/index',
              },
            },
          ],
          link: {
            type: 'doc',
            id: 'governance/ai-gateway/index',
          },
        },
        {
          type: 'doc',
          id: 'governance/unity-catalog',
          label: 'Unity Catalog',
        },
      ],
      link: {
        type: 'doc',
        id: 'governance/ai-gateway/index',
      },
    },
    {
      type: 'category',
      label: 'Concepts',
      className: 'sidebar-top-level-category',
      items: [
        {
          type: 'doc',
          id: 'concepts/trace',
          label: 'Trace',
        },
        {
          type: 'doc',
          id: 'concepts/span',
          label: 'Span',
        },
        {
          type: 'doc',
          id: 'concepts/feedback',
          label: 'Feedback',
        },
        {
          type: 'doc',
          id: 'concepts/expectations',
          label: 'Expectations',
        },
        {
          type: 'doc',
          id: 'concepts/scorers',
          label: 'Scorers',
        },
        {
          type: 'doc',
          id: 'concepts/evaluation-datasets',
          label: 'Evaluation Datasets',
        },
      ],
    },
  ],
};

export default sidebarsGenAI;
