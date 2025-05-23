import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

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
          type: 'doc',
          id: 'getting-started/logging-first-model/index',
        },
        {
          type: 'doc',
          id: 'getting-started/hyperparameter-tuning/index',
        },
        {
          type: 'doc',
          id: 'getting-started/registering-first-model/index',
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
                    label: 'Hyperparameter Tuning Guide',
                    items: [
                        {
                            type: 'category',
                            label: 'Notebooks',  //TODO: add tutorials
                            items: [
                                {
                                    type: 'doc',
                                    id: 'classic-ml/traditional-ml/hyperparameter-tuning/notebooks/hyperparameter-tuning-with-child-runs',
                                    label: 'Hyperparameter Tuning with Optuna',
                                }
                            ]
                        }
                    ],
                    link: {
                        type: 'doc',
                        id: 'traditional-ml/hyperparameter-tuning/index'
                    }
                }
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
              label: 'Keras',  //TODO: add quickstart notebook content
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
              label: 'PyTorch', //TODO: add quickstart notebook content
              items: [
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
              label: 'TensorFlow', //TODO: add quickstart notebook content
              items: [
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
                },
                {
                  type: 'doc',
                  id: 'tracking/backend-stores/index',
                },
                {
                  type: 'category',
                  label: 'Tutorials',
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
            }
          ],
          link: {
            type: 'doc',
            id: 'deployment/index'
          }
        }
      ],
    }

  ],
};

export default sidebarsClassicML;