import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const sidebarsGenAI: SidebarsConfig = {
  genAISidebar: [
    {
      type: 'doc',
      id: 'index',
      className: 'sidebar-top-level-category'
    },
    {
      type: 'category',
      label: 'Overview 🌟',
      className: 'sidebar-top-level-category',
      collapsed: false,
      items: [
        {
          type: 'doc',
          id: 'overview/key-challenges',
          label: 'Key Challenges',
        },
        {
          type: 'doc',
          id: 'overview/how-mlflow-helps',
          label: 'How MLflow helps',
        },
        {
          type: 'doc',
          id: 'overview/why-mlflow',
          label: 'Why use MLflow',
        },
        {
          type: 'doc',
          id: 'overview/oss-managed-diff',
          label: 'OSS vs Managed MLflow',
        },
      ],
      link: {
        type: 'doc',
        id: 'overview/index',
      }   
    },
    {
      type: 'category',
      label: '👉 Developer Workflow',
      className: 'sidebar-top-level-category',
      items: [
        {
          type: 'doc',
          id: 'developer-workflow/why-diff-workflow',
          label: 'Why a different workflow',
        },
        {
          type: 'doc',
          id: 'developer-workflow/eval-centric-development',
          label: 'MLflow\'s solution',
        },
        {
          type: 'doc',
          id: 'developer-workflow/get-started',
          label: 'Get Started!',
        },
      ],
      link: {
        type: 'doc',
        id: 'developer-workflow/index',
      }
    },
    {
      type: 'category',
      label: 'Getting Started 🚀',
      className: 'sidebar-top-level-category',
      items: [
        {
          type: 'doc',
          id: 'getting-started/hello-world-databricks',
          label: 'Getting started with Databricks MLflow',
        },
        {
          type: 'doc',
          id: 'getting-started/hello-world-oss',
          label: 'Getting started with Open-Source MLflow',
        },
        {
          type: 'doc',
          id: 'getting-started/hello-world-tracing',
          label: 'Getting started with tracing',
        },
        {
          type: 'doc',
          id: 'getting-started/hello-world-app',
          label: 'Viewing your first trace',
        },
        {
          type: 'category',
          label: 'Guides',
          items: [
            {
              type: 'doc',
              id: 'getting-started/guides/customer-support-app',
              label: 'Build your first GenAI application with MLflow'
            }
          ]
        }
      ],
      link: {
        type: 'doc',
        id: 'getting-started/index'
      }
    },
    {
      type: 'category',
      label: 'Tutorials 🎓',
      className: 'sidebar-top-level-category',
      items: [
        {
          type: 'doc',
          id: 'tutorials/debug-tracing',
          label: 'App Debugging with Tracing',
        },
        {
          type: 'doc',
          id: 'tutorials/dev-quality-iteration',
          label: 'Iterate on Quality during Development',
        },
        {
          type: 'doc',
          id: 'tutorials/enhance-quality',
          label: 'Using Traces for Quality Enhancement',
        },
        {
          type: 'doc',
          id: 'tutorials/prod-monitoring',
          label: 'Continuous Monitoring for Production Performance',
        },
        {
          type: 'doc',
          id: 'tutorials/jupyter-trace-demo-ipynb',
          label: 'Using Tracing in Jupyter Notebooks',
        }
      ],
      link: {
        type: 'doc',
        id: 'tutorials/index'
      }
    },
    {
      type: 'category',
      label: 'Tracing (Observability) 🔎',
      className: 'sidebar-top-level-category',
      collapsed: false,
      items: [
        {
          type: 'doc',
          id: 'tracing/tracing-101',
          label: 'Introduction to Tracing'
        },
        {
          type: 'doc',
          id: 'tracing/quickstart',
          label: 'Quickstart',
        },
        {
          type: 'category',
          label: 'Instrumenting GenAI Apps',
          items: [
            {
              type: 'doc',
              id: 'tracing/app-instrumentation/automatic'
            },
            {
              type: 'category',
              label: 'Manual Tracing',
              items: [
                {
                  type: 'doc',
                  id: 'tracing/app-instrumentation/manual-tracing/fluent-apis',
                },
                {
                  type: 'doc',
                  id: 'tracing/app-instrumentation/manual-tracing/low-level-api'
                }
              ],
              link: {
                type: 'doc',
                id: 'tracing/app-instrumentation/manual-tracing/index'
              },
            },
          ],
          link: {
            type: 'doc',
            id: 'tracing/app-instrumentation/index'
          },
        },
        {
          type: 'category',
          label: 'Tracing Integrations',
          items: [
            {
              type: 'autogenerated',
              dirName: 'tracing/integrations/listing'
            },
            {
              type: 'doc',
              id: 'tracing/integrations/contribute',
            },
          ],
          link: {
            type: 'doc',
            id: 'tracing/integrations/index',
          },
        },
        {
          type: 'category',
          label: 'Concepts',
          items: [
            {
              type: 'doc',
              id: 'tracing/concepts/trace-instrumentation'
            },
            {
              type: 'doc',
              id: 'tracing/data-model',
              label: 'Tracing Data Model',
            },
            {
              type: 'doc',
              id: 'tracing/concepts/trace/feedback',
            },
          ],
          link: {
            type: 'doc',
            id: 'tracing/concepts/trace-instrumentation'
          }
        },
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
          type: 'category',
          label: 'App Observation and Debugging',
          items: [
            {
              type: 'doc',
              id: 'tracing/observe-with-traces/delete-traces',
            },
            {
              type: 'doc',
              id: 'tracing/observe-with-traces/query-via-sdk',
            },
            {
              type: 'doc',
              id: 'tracing/observe-with-traces/ui',
            },
          ],
          link: {
            type: 'doc',
            id: 'tracing/observe-with-traces/index'
          },
        },
        {
          type: 'doc',
          id: 'tracing/search-traces',
          label: 'Searching for Traces',
        },
        {
          type: 'doc',
          id: 'tracing/attach-tags/index',
          label: 'Trace Tagging',
        },
        {
          type: 'doc',
          id: 'tracing/session',
          label: 'Tracing Chat Sessions',
        },
        {
          type: 'doc',
          id: 'tracing/lightweight-sdk',
          label: 'Lightweight Tracing SDK',
        },
        {
          type: 'doc',
          id: 'tracing/prod-tracing',
          label: 'Production Tracing',
        },
        {
          type: 'doc',
          id: 'tracing/quality-with-traces',
          label: 'Using Traces for Quality Improvement',
        },
        {
          type: 'doc',
          id: 'tracing/faq',
          label: 'Tracing FAQ'
        },
      ],
      link: {
        type: 'doc',
        id: 'tracing/index'
      }
    },
    {
      type: 'category',
      label: 'Evaluate & Monitor 📊',
      className: 'sidebar-top-level-category',
      items: [
        {
          type: 'doc',
          id: 'eval-monitor/build-eval-dataset',
          label: 'Build an Evaluation Dataset',
        },
        {
          type: 'doc',
          id: 'eval-monitor/define-quality-metrics',
          label: 'Define Quality Metrics',
        },
        {
          type: 'doc',
          id: 'eval-monitor/dev-testing',
          label: 'Testing during Development',
        },
        {
          type: 'doc',
          id: 'eval-monitor/ci-cd-testing',
          label: 'Evaluate within a CI/CD pipeline',
        },
        {
          type: 'doc',
          id: 'eval-monitor/prod-monitoring',
          label: 'Production Monitoring of Quality',
        }
      ],
      link: {
        type: 'doc',
        id: 'eval-monitor/index'
      }
    },
    {
      type: 'category',
      label: 'Human Feedback 📝',
      className: 'sidebar-top-level-category',
      items: [
        {
          type: 'doc',
          id: 'human-feedback/dev-annotations',
          label: 'Annotating Traces during Development',
        },
        {
          type: 'doc',
          id: 'human-feedback/user-feedback',
          label: 'Collecting User Feedback',
        },
        {
          type: 'doc',
          id: 'human-feedback/expert-feedback',
          label: 'Collecting Domain Expert Feedback',
        },
        {
          type: 'doc',
          id: 'human-feedback/feedback-sync-eval-dataset',
          label: 'Sync Feedback to Evaluation Datasets',
        }
      ],
      link: {
        type: 'doc',
        id: 'human-feedback/index'
      }
    },
    {
      type: 'category',
      label: 'Prompt and Version Management 🔨',
      className: 'sidebar-top-level-category',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'Version Tracking',
          items: [
            {
              type: 'doc',
              id: 'prompt-version-mgmt/version-tracking/quickstart',
              label: 'Quickstart',
            },
            {
              type: 'category',
              label: 'Guides',
              items: [
                {
                  type: 'doc',
                  id: 'prompt-version-mgmt/version-tracking/track-application-versions-with-mlflow',
                },
                {
                  type: 'doc',
                  id: 'prompt-version-mgmt/version-tracking/link-evaluation-results-and-traces-to-app-versions',
                },
                {
                  type: 'doc',
                  id: 'prompt-version-mgmt/version-tracking/compare-app-versions',
                },
                {
                  type: 'doc',
                  id: 'prompt-version-mgmt/version-tracking/optionally-package-app-code-and-files-for-databricks-model-serving',
                },
                {
                  type: 'doc',
                  id: 'prompt-version-mgmt/version-tracking/link-production-traces-to-app-versions'
                },
                {
                  type: 'doc',
                  id: 'prompt-version-mgmt/version-tracking/manual-tracking',
                }
              ]
            }
          ],
          link: {
            type: 'doc',
            id: 'prompt-version-mgmt/version-tracking/index'
          }
        },
        {
          type: 'category',
          label: 'Prompt Registry',
          items: [
            {
              type: 'doc',
              id: 'prompt-version-mgmt/prompt-registry/create-and-edit-prompts',
            },
            {
              type: 'doc',
              id: 'prompt-version-mgmt/prompt-registry/evaluate-prompts',
            },
            {
              type: 'doc',
              id: 'prompt-version-mgmt/prompt-registry/manage-prompt-lifecycles-with-aliases',
            },
            {
              type: 'doc',
              id: 'prompt-version-mgmt/prompt-registry/use-prompts-in-apps',
            },
            {
              type: 'doc',
              id: 'prompt-version-mgmt/prompt-registry/use-prompts-in-deployed-apps',
            },
          ],
          link: {
            type: 'doc',
            id: 'prompt-version-mgmt/prompt-registry/index'
          }
        },
        
      ]
    },
    {
      type: 'category',
      label: 'Application Serving ⛵',
      className: 'sidebar-top-level-category',
      items: [
        {
          type: 'doc',
          id: 'serving/responses-agent',
          label: 'Responses Agent',
        },
        {
          type: 'doc',
          id: 'serving/custom-apps',
          label: 'Custom Apps'
        },
        {
          type: 'doc',
          id: 'serving/endpoint-creation',
          label: 'Creating an Endpoint'
        }
      ],
      link: {
        type: 'doc',
        id: 'serving/index'
      }
    },
    {
      type: 'category',
      label: 'Governance 🛡️',
      className: 'sidebar-top-level-category',
      items: [
        {
          type: 'doc',
          id: 'governance/ai-gateway',
          label: 'AI Gateway',
        },
        {
          type: 'doc',
          id: 'governance/unity-catalog',
          label: 'Unity Catalog'
        }
      ],
      link: {
        type: 'doc',
        id: 'governance/index'
      }
    },
    {
      type: 'category',
      label: 'Data Model 🧩',
      className: 'sidebar-top-level-category',
      items: [
        {
          type: 'category',
          label: 'Experiments',
          items: [
            {
              type: 'doc',
              id: 'data-model/experiments/traces',
              label: 'Traces and Assessments'
            },
            {
              type: 'doc',
              id: 'data-model/experiments/prompts',
              label: 'Prompt Management',
            },
            {
              type: 'doc',
              id: 'data-model/experiments/evaluations',
              label: 'Evaluation',
            },
            {
              type: 'doc',
              id: 'data-model/experiments/eval-datasets',
              label: 'Evaluation Datasets',
            },
            {
              type: 'doc',
              id: 'data-model/experiments/app-versions',
              label: 'App Versions',
            },
            {
              type: 'doc',
              id: 'data-model/experiments/labeling',
              label: 'Trace Labeling',
            },
            {
              type: 'doc',
              id: 'data-model/experiments/monitors',
              label: 'Monitoring',
            },
          ],
          link: {
            type: 'doc',
            id: 'data-model/experiments/index'
          }
        }
      ]
    }
  ],
};

export default sidebarsGenAI;