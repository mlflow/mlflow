import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';
import { apiReferencePrefix } from "./docusaurusConfigUtils";

const sidebarsClassicML: SidebarsConfig = {
  classicMLSidebar: [
    {
      type: 'doc',
      id: 'index',
      className: 'sidebar-top-level-category',
    },
    {
      type: 'category',
      label: 'Getting Started 🚀',
      className: 'sidebar-top-level-category',
      collapsed: false,
      items: [
        {
          type: 'doc',
          id: 'getting-started/databricks-trial/index',
        },
        {
          type: 'doc',
          id: 'getting-started/intro-quickstart/index',
        },
        {
          type: 'doc',
          id: 'getting-started/running-notebooks/index',
        },
        {
          type: 'category',
          label: 'Your First MLflow Model: Complete Tutorial',
          items: [
            {
                type: 'doc',
                id: 'getting-started/logging-first-model/step1-tracking-server/index',
            },
            {
                type: 'doc',
                id: 'getting-started/logging-first-model/step2-mlflow-client/index',
            },
            {
                type: 'doc',
                id: 'getting-started/logging-first-model/step3-create-experiment/index',
            },
            {
                type: 'doc',
                id: 'getting-started/logging-first-model/step4-experiment-search/index',
            },
            {
                type: 'doc',
                id: 'getting-started/logging-first-model/step5-synthetic-data/index',
            },
            {
                type: 'doc',
                id: 'getting-started/logging-first-model/step6-logging-a-run/index',
            },
          ],
          link: {
            type: 'doc',
            id: 'getting-started/logging-first-model/index',
          }
        },
        {
          type: 'doc',
          id: 'getting-started/hyperparameter-tuning/index',
        },
        {
          type: 'category',
          label: 'Model Registry Quickstart',
          items: [
            {
                type: 'doc',
                id: 'getting-started/registering-first-model/step1-register-model/index',
            },
            {
                type: 'doc',
                id: 'getting-started/registering-first-model/step2-explore-registered-model/index',
            },
            {
                type: 'doc',
                id: 'getting-started/registering-first-model/step3-load-model/index'
            }
          ],
          link: {
            type: 'doc',
            id: 'getting-started/registering-first-model/index',
          },
        },
        {
          type: 'doc',
          id: 'getting-started/tracking-server-overview/index',
        }
      ],
      link: {
        type: 'doc',
        id: 'getting-started/index',
      }
    },
    {
      type: 'category',
      label: 'Machine Learning 🤖',
      className: 'sidebar-top-level-category',
      collapsed: false,
      items: [
        {
            type: 'category',
            label: 'Traditional ML',
            items: [
                {
                    type: 'category',
                    label: 'Hyperparameter Tuning with MLflow and Optuna',
                    items: [
                        {
                            type: 'doc',
                            id: 'traditional-ml/hyperparameter-tuning/part1-child-runs/index',
                        },
                        {
                            type: 'doc',
                            id: 'traditional-ml/hyperparameter-tuning/part2-logging-plots/index',
                        },
                        {
                            type: 'doc',
                            id: 'traditional-ml/hyperparameter-tuning/notebooks/index',
                            label: 'Notebooks',
                        },
                    ],
                    link: {
                        type: 'doc',
                        id: 'traditional-ml/hyperparameter-tuning/index',
                    },
                },
                {
                    type: 'category',
                    label: 'Building Custom Python Function Models with MLflow',
                    items: [
                        {
                            type: 'doc',
                            id: 'traditional-ml/creating-custom-pyfunc/part1-named-flavors/index',
                        },
                        {
                            type: 'doc',
                            id: 'traditional-ml/creating-custom-pyfunc/part2-pyfunc-components/index',
                        },
                        {
                            type: 'doc',
                            id: 'traditional-ml/creating-custom-pyfunc/notebooks/index',
                            label: 'Notebooks',
                        },
                    ],
                    link: {
                        type: 'doc',
                        id: 'traditional-ml/creating-custom-pyfunc/index',
                    }
                },
                {
                    type: 'category',
                    label: 'Serving Multiple Models on a Single Endpoint with a Custom PyFunc Model',
                    items: [
                        {
                            type: 'doc',
                            id: 'traditional-ml/serving-multiple-models-with-pyfunc/notebooks/index',
                            label: 'Notebooks',
                        },
                    ],
                    link: {
                        type: 'doc',
                        id: 'traditional-ml/serving-multiple-models-with-pyfunc/index',
                    },
                },
            ],
            link: {
                type: 'doc',
                id: 'traditional-ml/index'
            }
        },
        {
          type: 'category',
          label: 'Deep Learning 🕸️',
          items: [
            {
              type: 'category',
              label: 'Keras',
              items: [
                {
                  type: 'doc',
                  id: 'deep-learning/keras/quickstart/index',
                  label: 'Quickstart',
                },
                {
                  type: 'doc',
                  id: 'deep-learning/keras/guide/index',
                },
              ],
              link: {
                type: 'doc',
                id: 'deep-learning/keras/index',
              },
            },
            {
              type: 'category',
              label: 'PyTorch',
              items: [
                {
                    type: 'doc',
                    id: 'deep-learning/pytorch/quickstart/quickstart-pytorch-ipynb',
                    label: 'Quickstart',
                },
                {
                  type: 'doc',
                  id: 'deep-learning/pytorch/guide/index',
                },
              ],
              link: {
                type: 'doc',
                id: 'deep-learning/pytorch/index',
              },
            },
            {
              type: 'category',
              label: 'TensorFlow',
              items: [
                {
                    type: 'doc',
                    id: 'deep-learning/tensorflow/quickstart/quickstart-tensorflow-ipynb',
                    label: 'Quickstart',
                },
                {
                  type: 'doc',
                  id: 'deep-learning/tensorflow/guide/index',
                },
              ],
              link: {
                type: 'doc',
                id: 'deep-learning/tensorflow/index',
              },
            },
            {
              type: 'category',
              label: 'spaCy',
              items: [
                {
                  type: 'doc',
                  id: 'deep-learning/spacy/guide/index',
                },
              ],
              link: {
                type: 'doc',
                id: 'deep-learning/spacy/index',
              },
            },
          ],
          link: {
            type: 'doc',
            id: 'deep-learning/index',
          }
        }
      ]
    },
    {
      type: 'category',
      label: 'Build 🔨 ',
      className: 'sidebar-top-level-category',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'MLflow Tracking 📈',
          items: [
            {
              type: 'link',
              href: '/ml/getting-started/intro-quickstart/',
              label: 'Quickstart ⚡',
            },
            {
              type: 'doc',
              id: 'tracking/autolog/index',
              label: 'Auto Logging 🤖',
            },
            {
              type: 'category',
              label: 'Tracking Server 🖥️',
              items: [
                {
                  type: 'doc',
                  id: 'tracking/artifact-stores/index',
                  label: 'Artifact Store 📦',
                },
                {
                  type: 'doc',
                  id: 'tracking/backend-stores/index',
                  label: 'Backend Store 🗄️',
                },
                {
                  type: 'category',
                  label: 'Tutorials 🎓',
                  items: [
                    {
                      type: 'autogenerated',
                      dirName: 'tracking/tutorials',
                    }
                  ],
                }
              ],
              link: {
                type: 'doc',
                id: 'tracking/server/index',
              }
            },
            {
              type: 'category',
              label: 'Search 🔍',
              items: [
                {
                  type: 'doc',
                  id: 'search/search-models/index',
                },
                {
                  type: 'doc',
                  id: 'search/search-runs/index',
                },
                {
                  type: 'doc',
                  id: 'search/search-experiments/index',
                },
              ]
            },
            {
              type: 'doc',
              id: 'tracking/system-metrics/index',
              label: 'System Metrics 🌡️'
            },
            {
                type: 'doc',
                id: 'tracking/tracking-api/index',
                label: 'Tracking APIs 🛠️'
            }
          ],
          link: {
            type: 'doc',
            id: 'tracking/index',
          },
        },
        {
          type: 'category',
          label: 'MLflow Model 🧠',
          items: [
            {
              type: 'autogenerated',
              dirName: 'model',
            },
          ],
        },
      ],
    },
    {
        type: 'category',
        label: 'Evaluate 🎯',
        className: 'sidebar-top-level-category',
        items: [
            {
                type: 'doc',
                id: 'evaluation/function-eval/index',
            },
            {
                type: 'doc',
                id: 'evaluation/dataset-eval/index',
            },
            {
                type: 'doc',
                id: 'evaluation/model-eval/index',
            },
            {
                type: 'doc',
                id: 'evaluation/metrics-visualizations/index',
            },
            {
                type: 'doc',
                id: 'evaluation/shap/index',
            },
        ],
        link: {
            type: 'doc',
            id: 'evaluation/index'
        }
    },
    {
      type: 'category',
      label: 'Deploy 🚢',
      className: 'sidebar-top-level-category',
      items: [
        {
          type: 'doc',
          id: 'model-registry/index',
          label: 'MLflow Model Registry 📚'
        },
        {
          type: 'category',
          label: 'MLflow Serving ⚙️',
          items: [
            {
              type: 'doc',
              id: 'deployment/deploy-model-locally/index'
            },
            {
                type: 'doc',
                id: 'deployment/deploy-model-to-kubernetes/index',
            },
            {
                type: 'doc',
                id: 'deployment/deploy-model-to-sagemaker/index',
            },
          ],
          link: {
            type: 'doc',
            id: 'deployment/index'
          }
        },
      ],
    },
    {
      type: 'category',
      label: 'Team Collaboration 👥',
      className: 'sidebar-top-level-category',
      collapsed: true,
      items: [
        {
          type: 'link',
          href: 'tracking/#tracking-setup',
          label: 'Self-Hosting'
        },
        {
          type: 'link',
          href: '#running-mlflow-anywhere',
          label: 'Managed Services'
        },
        {
          type: 'doc',
          id: 'auth/index',
          label: 'Access Control 🔐',
        },
        {
          type: 'doc',
          id: 'projects/index',
          label: 'MLflow Projects 📦',
        },
      ]
    },
    {
        type: 'category',
        label: 'API References',
        className: 'sidebar-top-level-category',
        collapsed: true,
        items: [
        {
            type: 'link',
            label: 'Python API',
            href: `${apiReferencePrefix()}api_reference/python_api/index.html`,
        },
        {
            type: 'link',
            label: 'Java API',
            href: `${apiReferencePrefix()}api_reference/java_api/index.html`,
        },
        {
            type: 'link',
            label: 'R API',
            href: `${apiReferencePrefix()}api_reference/R-api.html`,
        },
        {
            type: 'link',
            label: 'REST API',
            href: `${apiReferencePrefix()}api_reference/rest-api.html`,
        },
        {
            type: 'link',
            label: 'CLI',
            href: `${apiReferencePrefix()}cli.html`,
        }
        ]
    },
    {
        type: 'category',
        label: 'More',
        collapsed: true,
        className:'sidebar-top-level-category',
        items: [
        {
            type: 'link',
            label: 'Contributing 🤝',
            href: 'https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.md',
        },
        {
            type: 'link',
            label: 'MLflow Blogs 📰',
            href: 'https://mlflow.org/blog/index.html',
        },
        {
            type: 'doc',
            id: 'plugins/index',
            label: 'MLflow Plugins 🔌'
        },
        ]
    },
  ],
};

export default sidebarsClassicML;