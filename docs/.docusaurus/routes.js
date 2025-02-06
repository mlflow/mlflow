import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '5ff'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '5ba'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'a2b'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', 'c3c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '156'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '88c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '000'),
    exact: true
  },
  {
    path: '/search',
    component: ComponentCreator('/search', '5de'),
    exact: true
  },
  {
    path: '/',
    component: ComponentCreator('/', 'c9e'),
    routes: [
      {
        path: '/',
        component: ComponentCreator('/', '450'),
        routes: [
          {
            path: '/',
            component: ComponentCreator('/', '795'),
            routes: [
              {
                path: '/auth/',
                component: ComponentCreator('/auth/', 'b65'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/community-model-flavors/',
                component: ComponentCreator('/community-model-flavors/', 'f12'),
                exact: true
              },
              {
                path: '/dataset/',
                component: ComponentCreator('/dataset/', 'fc6'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/deep-learning/',
                component: ComponentCreator('/deep-learning/', 'a08'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/deep-learning/keras/',
                component: ComponentCreator('/deep-learning/keras/', 'a0b'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/deep-learning/pytorch/',
                component: ComponentCreator('/deep-learning/pytorch/', '42b'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/deep-learning/pytorch/guide/',
                component: ComponentCreator('/deep-learning/pytorch/guide/', '829'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/deep-learning/tensorflow/',
                component: ComponentCreator('/deep-learning/tensorflow/', 'eb1'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/deep-learning/tensorflow/guide/',
                component: ComponentCreator('/deep-learning/tensorflow/guide/', 'd42'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/deployment/',
                component: ComponentCreator('/deployment/', '613'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/deployment/deploy-model-locally/',
                component: ComponentCreator('/deployment/deploy-model-locally/', 'ab0'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/deployment/deploy-model-to-kubernetes/',
                component: ComponentCreator('/deployment/deploy-model-to-kubernetes/', '9a2'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/deployment/deploy-model-to-kubernetes/tutorial/',
                component: ComponentCreator('/deployment/deploy-model-to-kubernetes/tutorial/', '779'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/deployment/deploy-model-to-sagemaker/',
                component: ComponentCreator('/deployment/deploy-model-to-sagemaker/', 'b9d'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/docker/',
                component: ComponentCreator('/docker/', '65f'),
                exact: true
              },
              {
                path: '/getting-started/',
                component: ComponentCreator('/getting-started/', 'ff4'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/getting-started/community-edition/',
                component: ComponentCreator('/getting-started/community-edition/', 'd2d'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/getting-started/intro-quickstart/',
                component: ComponentCreator('/getting-started/intro-quickstart/', 'a2a'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/getting-started/intro-quickstart/notebooks/',
                component: ComponentCreator('/getting-started/intro-quickstart/notebooks/', '047'),
                exact: true
              },
              {
                path: '/getting-started/logging-first-model/',
                component: ComponentCreator('/getting-started/logging-first-model/', 'f31'),
                exact: true
              },
              {
                path: '/getting-started/logging-first-model/notebooks/',
                component: ComponentCreator('/getting-started/logging-first-model/notebooks/', '174'),
                exact: true
              },
              {
                path: '/getting-started/logging-first-model/step1-tracking-server/',
                component: ComponentCreator('/getting-started/logging-first-model/step1-tracking-server/', '381'),
                exact: true
              },
              {
                path: '/getting-started/logging-first-model/step2-mlflow-client/',
                component: ComponentCreator('/getting-started/logging-first-model/step2-mlflow-client/', '8b1'),
                exact: true
              },
              {
                path: '/getting-started/logging-first-model/step3-create-experiment/',
                component: ComponentCreator('/getting-started/logging-first-model/step3-create-experiment/', '69c'),
                exact: true
              },
              {
                path: '/getting-started/logging-first-model/step4-experiment-search/',
                component: ComponentCreator('/getting-started/logging-first-model/step4-experiment-search/', 'c43'),
                exact: true
              },
              {
                path: '/getting-started/logging-first-model/step5-synthetic-data/',
                component: ComponentCreator('/getting-started/logging-first-model/step5-synthetic-data/', 'd63'),
                exact: true
              },
              {
                path: '/getting-started/logging-first-model/step6-logging-a-run/',
                component: ComponentCreator('/getting-started/logging-first-model/step6-logging-a-run/', 'da1'),
                exact: true
              },
              {
                path: '/getting-started/quickstart-2/',
                component: ComponentCreator('/getting-started/quickstart-2/', '764'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/getting-started/registering-first-model/',
                component: ComponentCreator('/getting-started/registering-first-model/', 'ff2'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/getting-started/registering-first-model/step1-register-model/',
                component: ComponentCreator('/getting-started/registering-first-model/step1-register-model/', 'cfe'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/getting-started/registering-first-model/step2-explore-registered-model/',
                component: ComponentCreator('/getting-started/registering-first-model/step2-explore-registered-model/', '7b7'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/getting-started/registering-first-model/step3-load-model/',
                component: ComponentCreator('/getting-started/registering-first-model/step3-load-model/', '909'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/getting-started/running-notebooks/',
                component: ComponentCreator('/getting-started/running-notebooks/', '6f2'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/getting-started/tracking-server-overview/',
                component: ComponentCreator('/getting-started/tracking-server-overview/', '6a4'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/introduction/',
                component: ComponentCreator('/introduction/', '7d6'),
                exact: true
              },
              {
                path: '/llms/',
                component: ComponentCreator('/llms/', '835'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/chat-model-guide/',
                component: ComponentCreator('/llms/chat-model-guide/', '480'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/chat-model-intro/',
                component: ComponentCreator('/llms/chat-model-intro/', 'cd4'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/custom-pyfunc-for-llms/',
                component: ComponentCreator('/llms/custom-pyfunc-for-llms/', 'e26'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/custom-pyfunc-for-llms/notebooks/',
                component: ComponentCreator('/llms/custom-pyfunc-for-llms/notebooks/', '303'),
                exact: true
              },
              {
                path: '/llms/deployments/',
                component: ComponentCreator('/llms/deployments/', '1c8'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/deployments/guides/',
                component: ComponentCreator('/llms/deployments/guides/', 'bd9'),
                exact: true
              },
              {
                path: '/llms/deployments/guides/step1-create-deployments/',
                component: ComponentCreator('/llms/deployments/guides/step1-create-deployments/', 'e06'),
                exact: true
              },
              {
                path: '/llms/deployments/guides/step2-query-deployments/',
                component: ComponentCreator('/llms/deployments/guides/step2-query-deployments/', '40b'),
                exact: true
              },
              {
                path: '/llms/deployments/uc_integration/',
                component: ComponentCreator('/llms/deployments/uc_integration/', 'aa5'),
                exact: true
              },
              {
                path: '/llms/dspy/',
                component: ComponentCreator('/llms/dspy/', 'dda'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/langchain/',
                component: ComponentCreator('/llms/langchain/', '90a'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/langchain/autologging',
                component: ComponentCreator('/llms/langchain/autologging', '6af'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/langchain/guide/',
                component: ComponentCreator('/llms/langchain/guide/', 'df3'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/llama-index/',
                component: ComponentCreator('/llms/llama-index/', '595'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/llm-evaluate/',
                component: ComponentCreator('/llms/llm-evaluate/', '8f0'),
                exact: true
              },
              {
                path: '/llms/llm-evaluate/notebooks/',
                component: ComponentCreator('/llms/llm-evaluate/notebooks/', '053'),
                exact: true
              },
              {
                path: '/llms/llm-tracking/',
                component: ComponentCreator('/llms/llm-tracking/', '66f'),
                exact: true
              },
              {
                path: '/llms/openai/',
                component: ComponentCreator('/llms/openai/', '358'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/openai/autologging/',
                component: ComponentCreator('/llms/openai/autologging/', 'd20'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/openai/guide/',
                component: ComponentCreator('/llms/openai/guide/', '96e'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/openai/notebooks/',
                component: ComponentCreator('/llms/openai/notebooks/', '128'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/prompt-engineering/',
                component: ComponentCreator('/llms/prompt-engineering/', 'b56'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/rag/',
                component: ComponentCreator('/llms/rag/', '6eb'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/rag/notebooks/',
                component: ComponentCreator('/llms/rag/notebooks/', 'f7d'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/sentence-transformers/',
                component: ComponentCreator('/llms/sentence-transformers/', 'b97'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/sentence-transformers/guide/',
                component: ComponentCreator('/llms/sentence-transformers/guide/', '80e'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/sentence-transformers/tutorials/',
                component: ComponentCreator('/llms/sentence-transformers/tutorials/', '33a'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/transformers/',
                component: ComponentCreator('/llms/transformers/', '882'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/transformers/guide/',
                component: ComponentCreator('/llms/transformers/guide/', '0b0'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/transformers/large-models/',
                component: ComponentCreator('/llms/transformers/large-models/', '04e'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/transformers/task/',
                component: ComponentCreator('/llms/transformers/task/', '350'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/llms/transformers/tutorials/',
                component: ComponentCreator('/llms/transformers/tutorials/', '340'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/model-evaluation/',
                component: ComponentCreator('/model-evaluation/', '833'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/model-registry/',
                component: ComponentCreator('/model-registry/', '0a7'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/model/',
                component: ComponentCreator('/model/', 'b69'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/model/dependencies/',
                component: ComponentCreator('/model/dependencies/', 'f16'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/model/models-from-code/',
                component: ComponentCreator('/model/models-from-code/', 'e7a'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/model/python_model',
                component: ComponentCreator('/model/python_model', '211'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/model/signatures/',
                component: ComponentCreator('/model/signatures/', '9b1'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/new-features/',
                component: ComponentCreator('/new-features/', 'cb6'),
                exact: true
              },
              {
                path: '/plugins/',
                component: ComponentCreator('/plugins/', '1f4'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/projects/',
                component: ComponentCreator('/projects/', 'b79'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/quickstart_drilldown/',
                component: ComponentCreator('/quickstart_drilldown/', '864'),
                exact: true
              },
              {
                path: '/recipes/',
                component: ComponentCreator('/recipes/', 'ea9'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/search-experiments/',
                component: ComponentCreator('/search-experiments/', '755'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/search-runs/',
                component: ComponentCreator('/search-runs/', '15b'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/system-metrics/',
                component: ComponentCreator('/system-metrics/', '380'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/',
                component: ComponentCreator('/tracing/', '920'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/api/',
                component: ComponentCreator('/tracing/api/', '204'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/api/client',
                component: ComponentCreator('/tracing/api/client', '0f3'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/api/how-to',
                component: ComponentCreator('/tracing/api/how-to', 'dfc'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/api/manual-instrumentation',
                component: ComponentCreator('/tracing/api/manual-instrumentation', '83e'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/api/search',
                component: ComponentCreator('/tracing/api/search', 'b3e'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/faq',
                component: ComponentCreator('/tracing/faq', '963'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/integrations/',
                component: ComponentCreator('/tracing/integrations/', '3ac'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/integrations/anthropic',
                component: ComponentCreator('/tracing/integrations/anthropic', 'bc3'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/integrations/autogen',
                component: ComponentCreator('/tracing/integrations/autogen', 'c98'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/integrations/bedrock',
                component: ComponentCreator('/tracing/integrations/bedrock', 'f80'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/integrations/contribute',
                component: ComponentCreator('/tracing/integrations/contribute', '951'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/integrations/crewai',
                component: ComponentCreator('/tracing/integrations/crewai', 'f39'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/integrations/dspy',
                component: ComponentCreator('/tracing/integrations/dspy', '310'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/integrations/gemini',
                component: ComponentCreator('/tracing/integrations/gemini', 'a09'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/integrations/groq',
                component: ComponentCreator('/tracing/integrations/groq', 'fcf'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/integrations/instructor',
                component: ComponentCreator('/tracing/integrations/instructor', '755'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/integrations/langchain',
                component: ComponentCreator('/tracing/integrations/langchain', '68d'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/integrations/langgraph',
                component: ComponentCreator('/tracing/integrations/langgraph', '660'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/integrations/litellm',
                component: ComponentCreator('/tracing/integrations/litellm', '128'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/integrations/llama_index',
                component: ComponentCreator('/tracing/integrations/llama_index', '9f4'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/integrations/mistral',
                component: ComponentCreator('/tracing/integrations/mistral', 'fa8'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/integrations/ollama',
                component: ComponentCreator('/tracing/integrations/ollama', '46b'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/integrations/openai',
                component: ComponentCreator('/tracing/integrations/openai', '960'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/integrations/swarm',
                component: ComponentCreator('/tracing/integrations/swarm', '86c'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/production',
                component: ComponentCreator('/tracing/production', '4e0'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/session',
                component: ComponentCreator('/tracing/session', 'eaf'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/tracing-schema',
                component: ComponentCreator('/tracing/tracing-schema', 'ba2'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/tutorials/',
                component: ComponentCreator('/tracing/tutorials/', 'd74'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/tutorials/concept',
                component: ComponentCreator('/tracing/tutorials/concept', '3d6'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracing/ui',
                component: ComponentCreator('/tracing/ui', '22d'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracking/',
                component: ComponentCreator('/tracking/', '046'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracking/artifacts-stores/',
                component: ComponentCreator('/tracking/artifacts-stores/', '3fe'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracking/autolog/',
                component: ComponentCreator('/tracking/autolog/', '53f'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracking/backend-stores/',
                component: ComponentCreator('/tracking/backend-stores/', 'b98'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracking/server/',
                component: ComponentCreator('/tracking/server/', '32c'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracking/tracking-api/',
                component: ComponentCreator('/tracking/tracking-api/', 'af0'),
                exact: true
              },
              {
                path: '/tracking/tutorials/local-database/',
                component: ComponentCreator('/tracking/tutorials/local-database/', '3df'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tracking/tutorials/remote-server/',
                component: ComponentCreator('/tracking/tutorials/remote-server/', '365'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/traditional-ml/',
                component: ComponentCreator('/traditional-ml/', '666'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/traditional-ml/creating-custom-pyfunc/',
                component: ComponentCreator('/traditional-ml/creating-custom-pyfunc/', 'ae9'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/traditional-ml/creating-custom-pyfunc/notebooks/',
                component: ComponentCreator('/traditional-ml/creating-custom-pyfunc/notebooks/', 'c80'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/traditional-ml/creating-custom-pyfunc/part1-named-flavors/',
                component: ComponentCreator('/traditional-ml/creating-custom-pyfunc/part1-named-flavors/', '710'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/traditional-ml/creating-custom-pyfunc/part2-pyfunc-components/',
                component: ComponentCreator('/traditional-ml/creating-custom-pyfunc/part2-pyfunc-components/', 'f71'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/traditional-ml/hyperparameter-tuning-with-child-runs/',
                component: ComponentCreator('/traditional-ml/hyperparameter-tuning-with-child-runs/', 'cc0'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/',
                component: ComponentCreator('/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/', '4ae'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/traditional-ml/hyperparameter-tuning-with-child-runs/part1-child-runs/',
                component: ComponentCreator('/traditional-ml/hyperparameter-tuning-with-child-runs/part1-child-runs/', 'fef'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/traditional-ml/hyperparameter-tuning-with-child-runs/part2-logging-plots/',
                component: ComponentCreator('/traditional-ml/hyperparameter-tuning-with-child-runs/part2-logging-plots/', 'ca4'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/traditional-ml/serving-multiple-models-with-pyfunc/',
                component: ComponentCreator('/traditional-ml/serving-multiple-models-with-pyfunc/', 'a83'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/tutorials-and-examples/',
                component: ComponentCreator('/tutorials-and-examples/', '426'),
                exact: true
              },
              {
                path: '/',
                component: ComponentCreator('/', '36e'),
                exact: true,
                sidebar: "docsSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
