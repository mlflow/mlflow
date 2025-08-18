import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';
import { apiReferencePrefix } from './docusaurusConfigUtils';

const sidebarsClassicML: SidebarsConfig = {
  classicMLSidebar: [
    {
      type: 'doc',
      id: 'index',
      className: 'sidebar-top-level-category',
      label: 'MLflow',
    },
    {
      type: 'link',
      label: 'MLflow 3.0',
      href: 'https://mlflow.org/docs/latest/genai/mlflow-3',
    },
    {
      type: 'category',
      label: 'Getting Started',
      className: 'sidebar-top-level-category',
      items: [
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
              id: 'getting-started/logging-first-model/step1-tracking-server',
            },
            {
              type: 'doc',
              id: 'getting-started/logging-first-model/step2-mlflow-client',
            },
            {
              type: 'doc',
              id: 'getting-started/logging-first-model/step3-create-experiment',
            },
            {
              type: 'doc',
              id: 'getting-started/logging-first-model/step4-experiment-search',
            },
            {
              type: 'doc',
              id: 'getting-started/logging-first-model/step5-synthetic-data',
            },
            {
              type: 'doc',
              id: 'getting-started/logging-first-model/step6-logging-a-run',
            },
            {
              type: 'category',
              label: 'Notebooks',
              items: [
                {
                  type: 'doc',
                  id: 'getting-started/logging-first-model/notebooks/logging-first-model-ipynb',
                  label: 'Logging First Model',
                },
              ],
              link: {
                type: 'doc',
                id: 'getting-started/logging-first-model/notebooks/index',
              },
            },
          ],
          link: {
            type: 'doc',
            id: 'getting-started/logging-first-model/index',
          },
        },
        {
          type: 'doc',
          id: 'getting-started/hyperparameter-tuning/index',
        },
        {
          type: 'doc',
          id: 'getting-started/deep-learning',
        },
        {
          type: 'category',
          label: 'Model Registry Quickstart',
          items: [
            {
              type: 'doc',
              id: 'getting-started/registering-first-model/step1-register-model',
            },
            {
              type: 'doc',
              id: 'getting-started/registering-first-model/step2-explore-registered-model',
            },
            {
              type: 'doc',
              id: 'getting-started/registering-first-model/step3-load-model',
            },
          ],
          link: {
            type: 'doc',
            id: 'getting-started/registering-first-model/index',
          },
        },
        {
          type: 'doc',
          id: 'getting-started/tracking-server-overview/index',
        },
      ],
      link: {
        type: 'doc',
        id: 'getting-started/index',
      },
    },
    {
      type: 'category',
      label: 'Machine Learning',
      className: 'sidebar-top-level-category',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'Traditional ML',
          items: [
            {
              type: 'category',
              label: 'Tutorials',
              items: [
                {
                  type: 'category',
                  label: 'Hyperparameter Tuning with MLflow and Optuna',
                  items: [
                    {
                      type: 'doc',
                      id: 'traditional-ml/tutorials/hyperparameter-tuning/part1-child-runs/index',
                    },
                    {
                      type: 'doc',
                      id: 'traditional-ml/tutorials/hyperparameter-tuning/part2-logging-plots/index',
                    },
                    {
                      type: 'category',
                      label: 'Notebooks',
                      items: [
                        {
                          type: 'doc',
                          id: 'traditional-ml/tutorials/hyperparameter-tuning/notebooks/hyperparameter-tuning-with-child-runs-ipynb',
                        },
                        {
                          type: 'doc',
                          id: 'traditional-ml/tutorials/hyperparameter-tuning/notebooks/logging-plots-in-mlflow-ipynb',
                        },
                        {
                          type: 'doc',
                          id: 'traditional-ml/tutorials/hyperparameter-tuning/notebooks/parent-child-runs-ipynb',
                        },
                      ],
                      link: {
                        type: 'doc',
                        id: 'traditional-ml/tutorials/hyperparameter-tuning/notebooks/index',
                      },
                    },
                  ],
                  link: {
                    type: 'doc',
                    id: 'traditional-ml/tutorials/hyperparameter-tuning/index',
                  },
                },
                {
                  type: 'category',
                  label: 'Building Custom Python Function Models with MLflow',
                  items: [
                    {
                      type: 'doc',
                      id: 'traditional-ml/tutorials/creating-custom-pyfunc/part1-named-flavors/index',
                    },
                    {
                      type: 'doc',
                      id: 'traditional-ml/tutorials/creating-custom-pyfunc/part2-pyfunc-components/index',
                    },
                    {
                      type: 'category',
                      label: 'Notebooks',
                      items: [
                        {
                          type: 'doc',
                          id: 'traditional-ml/tutorials/creating-custom-pyfunc/notebooks/basic-pyfunc-ipynb',
                          label: 'Introduction to PythonModel',
                        },
                        {
                          type: 'doc',
                          id: 'traditional-ml/tutorials/creating-custom-pyfunc/notebooks/introduction-ipynb',
                          label: 'Custom Model Basics',
                        },
                        {
                          type: 'doc',
                          id: 'traditional-ml/tutorials/creating-custom-pyfunc/notebooks/override-predict-ipynb',
                          label: 'Customizing the `predict` method',
                        },
                      ],
                      link: {
                        type: 'doc',
                        id: 'traditional-ml/tutorials/creating-custom-pyfunc/notebooks/index',
                      },
                    },
                  ],
                  link: {
                    type: 'doc',
                    id: 'traditional-ml/tutorials/creating-custom-pyfunc/index',
                  },
                },
                {
                  type: 'category',
                  label: 'Serving Multiple Models on a Single Endpoint with a Custom PyFunc Model',
                  items: [
                    {
                      type: 'doc',
                      id: 'traditional-ml/tutorials/serving-multiple-models-with-pyfunc/notebooks/MME_Tutorial-ipynb',
                      label: 'Notebooks',
                    },
                  ],
                  link: {
                    type: 'doc',
                    id: 'traditional-ml/tutorials/serving-multiple-models-with-pyfunc/index',
                  },
                },
              ],
            },
            {
              type: 'category',
              label: 'Scikit Learn',
              items: [
                {
                  type: 'doc',
                  id: 'traditional-ml/sklearn/quickstart/quickstart-sklearn-ipynb',
                  label: 'Quickstart',
                },
                {
                  type: 'doc',
                  id: 'traditional-ml/sklearn/guide/index',
                  label: 'Scikit Learn within MLflow',
                },
              ],
              link: {
                type: 'doc',
                id: 'traditional-ml/sklearn/index',
              },
            },
            {
              type: 'category',
              label: 'XGBoost',
              items: [
                {
                  type: 'doc',
                  id: 'traditional-ml/xgboost/quickstart/quickstart-xgboost-ipynb',
                  label: 'Quickstart',
                },
                {
                  type: 'doc',
                  id: 'traditional-ml/xgboost/guide/index',
                  label: 'XGBoost within MLflow',
                },
              ],
              link: {
                type: 'doc',
                id: 'traditional-ml/xgboost/index',
              },
            },
            {
              type: 'category',
              label: 'SparkML',
              items: [
                {
                  type: 'doc',
                  id: 'traditional-ml/sparkml/guide/index',
                  label: 'SparkML within MLflow',
                },
              ],
              link: {
                type: 'doc',
                id: 'traditional-ml/sparkml/index',
              },
            },
            {
              type: 'category',
              label: 'Prophet',
              items: [
                {
                  type: 'doc',
                  id: 'traditional-ml/prophet/guide/index',
                  label: 'Prophet within MLflow',
                },
              ],
              link: {
                type: 'doc',
                id: 'traditional-ml/prophet/index',
              },
            },
          ],
          link: {
            type: 'doc',
            id: 'traditional-ml/index',
          },
        },
        {
          type: 'category',
          label: 'Deep Learning',
          items: [
            {
              type: 'category',
              label: 'Keras',
              items: [
                {
                  type: 'doc',
                  id: 'deep-learning/keras/quickstart/quickstart-keras-ipynb',
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
              label: 'Transformers',
              items: [
                {
                  type: 'doc',
                  id: 'deep-learning/transformers/guide/index',
                },
                {
                  type: 'doc',
                  id: 'deep-learning/transformers/large-models/index',
                  label: 'Working with Large Transformers Models',
                },
                {
                  type: 'doc',
                  id: 'deep-learning/transformers/task/index',
                  label: 'Transformers Task Types',
                },
                {
                  type: 'category',
                  label: 'Tutorials',
                  items: [
                    {
                      type: 'doc',
                      id: 'deep-learning/transformers/tutorials/conversational/conversational-model-ipynb',
                      label: 'Introduction to Conversational Models',
                    },
                    {
                      type: 'doc',
                      id: 'deep-learning/transformers/tutorials/conversational/pyfunc-chat-model-ipynb',
                      label: 'Custom Conversational Models',
                    },
                    {
                      type: 'doc',
                      id: 'deep-learning/transformers/tutorials/fine-tuning/transformers-fine-tuning-ipynb',
                      label: 'Introduction to Fine Tuning',
                    },
                    {
                      type: 'doc',
                      id: 'deep-learning/transformers/tutorials/fine-tuning/transformers-peft-ipynb',
                      label: 'Leveraging PEFT for Fine Tuning',
                    },
                    {
                      type: 'doc',
                      id: 'deep-learning/transformers/tutorials/audio-transcription/whisper-ipynb',
                      label: 'Introduction to Audio Transcription',
                    },
                    {
                      type: 'doc',
                      id: 'deep-learning/transformers/tutorials/prompt-templating/prompt-templating-ipynb',
                      label: 'Introduction to Prompt Templating',
                    },
                    {
                      type: 'doc',
                      id: 'deep-learning/transformers/tutorials/text-generation/text-generation-ipynb',
                      label: 'Text Generation Models',
                    },
                    {
                      type: 'doc',
                      id: 'deep-learning/transformers/tutorials/translation/component-translation-ipynb',
                      label: 'Translation Models',
                    },
                  ],
                  link: {
                    type: 'doc',
                    id: 'deep-learning/transformers/tutorials/index',
                  },
                },
              ],
              link: {
                type: 'doc',
                id: 'deep-learning/transformers/index',
              },
            },
            {
              type: 'category',
              label: 'Sentence Transformers',
              items: [
                {
                  type: 'doc',
                  id: 'deep-learning/sentence-transformers/guide/index',
                },
                {
                  type: 'category',
                  label: 'Tutorials',
                  items: [
                    {
                      type: 'doc',
                      id: 'deep-learning/sentence-transformers/tutorials/quickstart/sentence-transformers-quickstart-ipynb',
                    },
                    {
                      type: 'doc',
                      id: 'deep-learning/sentence-transformers/tutorials/paraphrase-mining/paraphrase-mining-sentence-transformers-ipynb',
                    },
                    {
                      type: 'doc',
                      id: 'deep-learning/sentence-transformers/tutorials/semantic-search/semantic-search-sentence-transformers-ipynb',
                    },
                    {
                      type: 'doc',
                      id: 'deep-learning/sentence-transformers/tutorials/semantic-similarity/semantic-similarity-sentence-transformers-ipynb',
                    },
                  ],
                  link: {
                    type: 'doc',
                    id: 'deep-learning/sentence-transformers/tutorials/index',
                  },
                },
              ],
              link: {
                type: 'doc',
                id: 'deep-learning/sentence-transformers/index',
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
          },
        },
      ],
    },
    {
      type: 'category',
      label: 'Build ',
      className: 'sidebar-top-level-category',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'MLflow Tracking',
          items: [
            {
              type: 'category',
              label: 'Quickstart ⚡',
              items: [
                {
                  type: 'doc',
                  id: 'tracking/quickstart/notebooks/tracking_quickstart-ipynb',
                },
              ],
              link: {
                type: 'doc',
                id: 'tracking/quickstart/index',
              },
            },
            {
              type: 'doc',
              id: 'tracking/autolog/index',
              label: 'Auto Logging',
            },
            {
              type: 'category',
              label: 'Tracking Server',
              items: [
                {
                  type: 'doc',
                  id: 'tracking/artifact-stores/index',
                  label: 'Artifact Store',
                },
                {
                  type: 'doc',
                  id: 'tracking/backend-stores/index',
                  label: 'Backend Store',
                },
                {
                  type: 'category',
                  label: 'Tutorials',
                  items: [
                    {
                      type: 'autogenerated',
                      dirName: 'tracking/tutorials',
                    },
                  ],
                },
              ],
              link: {
                type: 'doc',
                id: 'tracking/server/index',
              },
            },
            {
              type: 'category',
              label: 'Search',
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
              ],
            },
            {
              type: 'doc',
              id: 'tracking/system-metrics/index',
              label: 'System Metrics',
            },
            {
              type: 'doc',
              id: 'tracking/tracking-api/index',
              label: 'Tracking APIs',
            },
          ],
          link: {
            type: 'doc',
            id: 'tracking/index',
          },
        },
        {
          type: 'category',
          label: 'MLflow Model',
          items: [
            {
              type: 'autogenerated',
              dirName: 'model',
            },
            {
              type: 'doc',
              id: 'community-model-flavors/index',
              label: 'Community-Managed Model Integrations',
            },
          ],
        },
        {
          type: 'doc',
          id: 'dataset/index',
          label: 'MLflow Datasets',
        },
      ],
    },
    {
      type: 'category',
      label: 'Evaluate',
      className: 'sidebar-top-level-category',
      items: [
        {
          type: 'doc',
          id: 'evaluation/function-eval',
        },
        {
          type: 'doc',
          id: 'evaluation/dataset-eval',
        },
        {
          type: 'doc',
          id: 'evaluation/model-eval',
        },
        {
          type: 'doc',
          id: 'evaluation/metrics-visualizations',
        },
        {
          type: 'doc',
          id: 'evaluation/shap',
        },
        {
          type: 'doc',
          id: 'evaluation/plugin-evaluators',
        },
      ],
      link: {
        type: 'doc',
        id: 'evaluation/index',
      },
    },
    {
      type: 'category',
      label: 'Deploy',
      className: 'sidebar-top-level-category',
      items: [
        {
          type: 'category',
          label: 'MLflow Model Registry',
          items: [
            {
              type: 'autogenerated',
              dirName: 'model-registry',
            },
          ],
          link: {
            type: 'doc',
            id: 'model-registry/index',
          },
        },
        {
          type: 'category',
          label: 'MLflow Serving',
          items: [
            {
              type: 'doc',
              id: 'deployment/deploy-model-locally/index',
            },
            {
              type: 'category',
              label: 'Deploy MLflow Model to Kubernetes',
              items: [
                {
                  type: 'doc',
                  id: 'deployment/deploy-model-to-kubernetes/tutorial',
                },
              ],
              link: {
                type: 'doc',
                id: 'deployment/deploy-model-to-kubernetes/index',
              },
            },
            {
              type: 'doc',
              id: 'deployment/deploy-model-to-sagemaker/index',
            },
          ],
          link: {
            type: 'doc',
            id: 'deployment/index',
          },
        },
        {
          type: 'doc',
          id: 'docker/index',
          label: 'Docker',
        },
      ],
    },
    {
      type: 'doc',
      id: 'webhooks/index',
      label: 'Webhooks',
    },
    {
      type: 'category',
      label: 'Team Collaboration',
      className: 'sidebar-top-level-category',
      collapsed: true,
      items: [
        {
          type: 'link',
          href: '/ml/tracking/#tracking-setup',
          label: 'Self-Hosting',
        },
        {
          type: 'link',
          href: '/ml/#running-mlflow-anywhere',
          label: 'Managed Services',
        },
        {
          type: 'doc',
          id: 'auth/index',
          label: 'Access Control',
        },
        {
          type: 'doc',
          id: 'projects/index',
          label: 'MLflow Projects',
        },
      ],
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
        },
      ],
    },
    {
      type: 'category',
      label: 'More',
      collapsed: true,
      className: 'sidebar-top-level-category',
      items: [
        {
          type: 'link',
          label: 'Contributing',
          href: 'https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.md',
        },
        {
          type: 'link',
          label: 'MLflow Blogs',
          href: 'https://mlflow.org/blog/index.html',
        },
        {
          type: 'doc',
          id: 'plugins/index',
          label: 'MLflow Plugins',
        },
        {
          type: 'doc',
          id: 'tutorials-and-examples/index',
          label: 'External Tutorials',
        },
      ],
    },
  ],
};

export default sidebarsClassicML;
