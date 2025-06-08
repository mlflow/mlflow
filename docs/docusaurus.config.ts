import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";
import {
  postProcessSidebar,
  apiReferencePrefix,
} from "./docusaurusConfigUtils";
import tailwindPlugin from "./src/plugins/tailwind-config.cjs";

// ensure baseUrl always ends in `/`
const baseUrl = (process.env.DOCS_BASE_URL ?? "/").replace(/\/?$/, "/");

// Helper functions for creating redirects
const createMLRedirects = (paths: string[]) => {
  return paths.map(path => ({
    to: `/ml${path}`,
    from: path
  }));
};

const createGenAIRedirects = (paths: string[]) => {
  return paths.map(path => ({
    to: `/genai${path}`,
    from: path
  }));
};

// URLs that moved from root to /ml
const mlPaths = [
  '/auth/',
  '/community-model-flavors/',
  '/dataset/',
  '/deep-learning/',
  '/deep-learning/keras/',
  '/deep-learning/keras/quickstart/quickstart_keras',
  '/deep-learning/pytorch/',
  '/deep-learning/pytorch/guide/',
  '/deep-learning/pytorch/quickstart/pytorch_quickstart',
  '/deep-learning/tensorflow/',
  '/deep-learning/tensorflow/guide/',
  '/deep-learning/tensorflow/quickstart/quickstart_tensorflow',
  '/deployment/',
  '/deployment/deploy-model-locally/',
  '/deployment/deploy-model-to-kubernetes/',
  '/deployment/deploy-model-to-kubernetes/tutorial/',
  '/deployment/deploy-model-to-sagemaker/',
  '/docker/',
  '/getting-started/',
  '/getting-started/databricks-trial/',
  '/getting-started/intro-quickstart/',
  '/getting-started/intro-quickstart/notebooks/',
  '/getting-started/intro-quickstart/notebooks/tracking_quickstart',
  '/getting-started/logging-first-model/',
  '/getting-started/logging-first-model/notebooks/',
  '/getting-started/logging-first-model/notebooks/logging-first-model',
  '/getting-started/logging-first-model/step1-tracking-server/',
  '/getting-started/logging-first-model/step2-mlflow-client/',
  '/getting-started/logging-first-model/step3-create-experiment/',
  '/getting-started/logging-first-model/step4-experiment-search/',
  '/getting-started/logging-first-model/step5-synthetic-data/',
  '/getting-started/logging-first-model/step6-logging-a-run/',
  '/getting-started/quickstart-2/',
  '/getting-started/registering-first-model/',
  '/getting-started/registering-first-model/step1-register-model/',
  '/getting-started/registering-first-model/step2-explore-registered-model/',
  '/getting-started/registering-first-model/step3-load-model/',
  '/getting-started/running-notebooks/',
  '/getting-started/tracking-server-overview/',
  '/introduction/',
  '/model-evaluation/',
  '/model-registry/',
  '/model/',
  '/model/dependencies/',
  '/model/models-from-code/',
  '/model/notebooks/signature_examples',
  '/model/python_model',
  '/model/signatures/',
  '/new-features/',
  '/plugins/',
  '/projects/',
  '/quickstart_drilldown/',
  '/recipes/',
  '/search-experiments/',
  '/search-runs/',
  '/system-metrics/',
  '/tracking/',
  '/tracking/artifacts-stores/',
  '/tracking/autolog/',
  '/tracking/backend-stores/',
  '/tracking/server/',
  '/tracking/tracking-api/',
  '/tracking/tutorials/local-database/',
  '/tracking/tutorials/remote-server/',
  '/traditional-ml/',
  '/traditional-ml/creating-custom-pyfunc/',
  '/traditional-ml/creating-custom-pyfunc/notebooks/',
  '/traditional-ml/creating-custom-pyfunc/notebooks/basic-pyfunc',
  '/traditional-ml/creating-custom-pyfunc/notebooks/introduction',
  '/traditional-ml/creating-custom-pyfunc/notebooks/override-predict',
  '/traditional-ml/creating-custom-pyfunc/part1-named-flavors/',
  '/traditional-ml/creating-custom-pyfunc/part2-pyfunc-components/',
  '/traditional-ml/hyperparameter-tuning-with-child-runs/',
  '/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/',
  '/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/hyperparameter-tuning-with-child-runs',
  '/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/logging-plots-in-mlflow',
  '/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/parent-child-runs',
  '/traditional-ml/hyperparameter-tuning-with-child-runs/part1-child-runs/',
  '/traditional-ml/hyperparameter-tuning-with-child-runs/part2-logging-plots/',
  '/traditional-ml/serving-multiple-models-with-pyfunc/',
  '/traditional-ml/serving-multiple-models-with-pyfunc/notebooks/MME_Tutorial',
  '/tutorials-and-examples/',
];

// URLs that moved from root to /genai (LLM-related content)
const genAIPaths = [
  '/llms/',
  '/llms/chat-model-guide/',
  '/llms/chat-model-intro/',
  '/llms/custom-pyfunc-for-llms/',
  '/llms/custom-pyfunc-for-llms/notebooks/',
  '/llms/custom-pyfunc-for-llms/notebooks/custom-pyfunc-advanced-llm',
  '/llms/deployments/',
  '/llms/deployments/guides/',
  '/llms/deployments/guides/step1-create-deployments/',
  '/llms/deployments/guides/step2-query-deployments/',
  '/llms/deployments/uc_integration/',
  '/llms/dspy/',
  '/llms/dspy/notebooks/dspy_quickstart',
  '/llms/dspy/optimizer',
  '/llms/langchain/',
  '/llms/langchain/autologging',
  '/llms/langchain/guide/',
  '/llms/langchain/notebooks/langchain-quickstart',
  '/llms/langchain/notebooks/langchain-retriever',
  '/llms/llama-index/',
  '/llms/llama-index/notebooks/llama_index_quickstart',
  '/llms/llama-index/notebooks/llama_index_workflow_tutorial',
  '/llms/llm-evaluate/',
  '/llms/llm-evaluate/notebooks/',
  '/llms/llm-evaluate/notebooks/huggingface-evaluation',
  '/llms/llm-evaluate/notebooks/question-answering-evaluation',
  '/llms/llm-evaluate/notebooks/rag-evaluation',
  '/llms/llm-evaluate/notebooks/rag-evaluation-llama2',
  '/llms/llm-tracking/',
  '/llms/notebooks/chat-model-tool-calling',
  '/llms/openai/',
  '/llms/openai/autologging/',
  '/llms/openai/guide/',
  '/llms/openai/notebooks/',
  '/llms/openai/notebooks/openai-chat-completions',
  '/llms/openai/notebooks/openai-code-helper',
  '/llms/openai/notebooks/openai-embeddings-generation',
  '/llms/openai/notebooks/openai-quickstart',
  '/llms/prompt-engineering/',
  '/llms/rag/',
  '/llms/rag/notebooks/',
  '/llms/rag/notebooks/mlflow-e2e-evaluation',
  '/llms/rag/notebooks/question-generation-retrieval-evaluation',
  '/llms/rag/notebooks/retriever-evaluation-tutorial',
  '/llms/sentence-transformers/',
  '/llms/sentence-transformers/guide/',
  '/llms/sentence-transformers/tutorials/',
  '/llms/sentence-transformers/tutorials/paraphrase-mining/paraphrase-mining-sentence-transformers',
  '/llms/sentence-transformers/tutorials/quickstart/sentence-transformers-quickstart',
  '/llms/sentence-transformers/tutorials/semantic-search/semantic-search-sentence-transformers',
  '/llms/sentence-transformers/tutorials/semantic-similarity/semantic-similarity-sentence-transformers',
  '/llms/transformers/',
  '/llms/transformers/guide/',
  '/llms/transformers/large-models/',
  '/llms/transformers/task/',
  '/llms/transformers/tutorials/',
  '/llms/transformers/tutorials/audio-transcription/whisper',
  '/llms/transformers/tutorials/conversational/conversational-model',
  '/llms/transformers/tutorials/conversational/pyfunc-chat-model',
  '/llms/transformers/tutorials/fine-tuning/transformers-fine-tuning',
  '/llms/transformers/tutorials/fine-tuning/transformers-peft',
  '/llms/transformers/tutorials/prompt-templating/prompt-templating',
  '/llms/transformers/tutorials/text-generation/text-generation',
  '/llms/transformers/tutorials/translation/component-translation',
  '/prompts/',
  '/prompts/cm',
  '/prompts/evaluate',
  '/prompts/run-and-model',
  '/tracing/',
  '/tracing/api/',
  '/tracing/api/client',
  '/tracing/api/how-to',
  '/tracing/api/manual-instrumentation',
  '/tracing/api/search',
  '/tracing/faq',
  '/tracing/integrations/',
  '/tracing/integrations/anthropic',
  '/tracing/integrations/autogen',
  '/tracing/integrations/bedrock',
  '/tracing/integrations/contribute',
  '/tracing/integrations/crewai',
  '/tracing/integrations/deepseek',
  '/tracing/integrations/dspy',
  '/tracing/integrations/gemini',
  '/tracing/integrations/groq',
  '/tracing/integrations/instructor',
  '/tracing/integrations/langchain',
  '/tracing/integrations/langgraph',
  '/tracing/integrations/litellm',
  '/tracing/integrations/llama_index',
  '/tracing/integrations/mistral',
  '/tracing/integrations/ollama',
  '/tracing/integrations/openai',
  '/tracing/integrations/openai-agent',
  '/tracing/integrations/swarm',
  '/tracing/integrations/txtai',
  '/tracing/production',
  '/tracing/session',
  '/tracing/tracing-schema',
  '/tracing/tutorials/',
  '/tracing/tutorials/concept',
  '/tracing/tutorials/jupyter-trace-demo',
  '/tracing/ui',
];

// Special redirects that don't follow the simple pattern
const specialRedirects = [
  // MLflow 3 related
  {
    to: "/ml/mlflow-3/",
    from: "/mlflow-3/"
  },
  {
    to: "/ml/mlflow-3/deep-learning",
    from: "/mlflow-3/deep-learning"
  },
  {
    to: "/ml/mlflow-3/genai-agent",
    from: "/mlflow-3/genai-agent"
  },
  
  // Legacy redirects that should map to specific new locations
  {
    to: "/genai/tracing",
    from: ["/llms/tracing", "/tracing"]
  },
  {
    to: "/ml/tracking",
    from: ["/tracking/data-api/index", "/tracking/data-api", "/dataset"]
  },
  {
    to: "/ml/model-registry",
    from: ["/models", "/model", "/registry", "/model-registry"]
  },
  {
    to: "/genai/governance/ai-gateway",
    from: [
      "/llms/gateway/index", 
      "/llms/gateway", 
      "/llms/deployments",
      "/llms/gateway/guides/index", 
      "/llms/gateway/guide", 
      "/llms/deployments/guides"
    ]
  },
  {
    to: "/genai/governance/ai-gateway/guides/step1-create-deployments/",
    from: [
      "/llms/gateway/guides/step1-create-gateway", 
      "/llms/deployments/guides/step1-create-deployments"
    ]
  },
  {
    to: "/genai/governance/ai-gateway/guides/step2-query-deployments/",
    from: [
      "/llms/gateway/guides/step2-query-gateway", 
      "/llms/deployments/guides/step2-query-deployments"
    ]
  },
  {
    to: "/genai/getting-started",
    from: [
      "/getting-started/community-edition", 
      "/getting-started/databricks-trial"
    ]
  },
  {
    to: "/genai/tracing/integrations/contribute",
    from: [
      "/llms/tracing/contribute", 
      "/tracing/integrations/contribute"
    ]
  },
  {
    to: "/genai/tutorials/jupyter-trace-demo",
    from: [
      "/llms/tracing/notebooks/jupyter-trace-demo", 
      "/tracing/tutorials/jupyter-trace-demo"
    ]
  },
  {
    to: "/genai/tracing/tracing-101",
    from: [
      "/llms/tracing/overview", 
      "/tracing/tutorials/concept"
    ]
  },
  {
    to: "/genai/tracing/search-traces",
    from: [
      "/llms/tracing/search-traces", 
      "/tracing/api/search"
    ]
  },
  {
    to: "/genai/tracing/data-model",
    from: [
      "/llms/tracing/tracing-schema", 
      "/tracing/tracing-schema"
    ]
  },
];

// Combine all redirects
const redirectsConfig = [
  ...createMLRedirects(mlPaths),
  ...createGenAIRedirects(genAIPaths),
  ...specialRedirects,
];

const config: Config = {
  title: "MLflow",
  tagline: "MLflow Documentation",
  favicon: "images/favicon.ico",

  // Set the production url of your site here
  url: "https://mlflow.org",

  // when building for production, check this environment
  // variable to determine the correct base URL
  baseUrl: baseUrl,

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: "mlflow", // Usually your GitHub org/user name.
  projectName: "mlflow", // Usually your repo name.

  staticDirectories: ["static"],

  // change to throw when migration is done
  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "throw",
  onBrokenAnchors: "throw",

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  // NB: docusaurus does not automatically prepend the
  // baseURL to relative paths in script srcs. please
  // make sure to append this yourself.
  scripts: [
    {
      // runllm
      src: baseUrl + "js/runllm.js",
      defer: true,
    },
  ],

  themes: ['@docusaurus/theme-mermaid'],
  markdown: {
    mermaid: true,
  },
  
  presets: [
    [
      "classic",
      {
        docs: false,
        theme: {
          customCss: "./src/css/custom.css",
        },
        googleTagManager: {
          containerId: process.env.GTM_ID || "GTM-TEST",
        },
        gtag: {
          trackingID: [process.env.GTM_ID || "GTM-TEST", "AW-16857946923"],
          anonymizeIP: true,
        },
      } satisfies Preset.Options,
    ],
  ],

  clientModules: [
    require.resolve('./src/docusaurus.theme.js'),
  ],

  themeConfig: {
    mermaid: {
      theme: { light: 'neutral', dark: 'dark' },
      options: {
        fontFamily: 'inherit',
        fontSize: 16,
      },
    },
    ...(process.env.PR_PREVIEW
      ? {
          announcementBar: {
            id: "pr_preview",
            content:
              "<strong>⚠️ Reloading the page causes a 404 error. Add /index.html to the URL to avoid it ⚠️</strong>",
            backgroundColor: "#0194e2",
            textColor: "#ffffff",
            isCloseable: true,
          },
        }
      : {}),
    navbar: {
      logo: {
        alt: "MLflow Logo",
        src: "images/logo-light.svg",
        srcDark: "images/logo-dark.svg",
      },
      items: [
        {
          type: "dropdown",
          label: "Documentation",
          items: [
            // Classic ML docs
            {
              type: "docSidebar",
              sidebarId: "classicMLSidebar",
              label: "ML Docs",
              docsPluginId: "classic-ml",
              className: "ml-docs-link",
            },
            // GenAI docs
            {
              type: "docSidebar",
              sidebarId: "genAISidebar",
              label: "GenAI Docs",
              docsPluginId: "genai",
              className: "genai-docs-link",
            },
          ]
        },
        {
          to: `${apiReferencePrefix()}api_reference/index.html`,
          position: "left",
          label: "API Reference",
        },
        {
          href: "https://github.com/mlflow/mlflow",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      links: [
        {
          title: "Community",
          items: [
            {
              label: "Stack Overflow",
              href: "https://stackoverflow.com/questions/tagged/mlflow",
            },
            {
              label: "LinkedIn",
              href: "https://www.linkedin.com/company/mlflow-org",
            },
            {
              label: "X",
              href: "https://x.com/mlflow",
            },
          ],
        },
        {
          title: "Resources",
          items: [
            {
              label: "Docs",
              to: "/",
            },
            {
              label: "Releases",
              to: "https://mlflow.org/releases",
            },
            {
              label: "Blog",
              to: "https://mlflow.org/blog",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} MLflow Project, a Series of LF Projects, LLC.`,
    },
    prism: {
      // There is an array of languages enabled by default.
      // @see https://github.com/FormidableLabs/prism-react-renderer/blob/master/packages/generate-prism-languages/index.ts#L9-L26
      additionalLanguages: [
        "bash",
        "diff",
        "ini",
        "java",
        "nginx",
        "r",
        "scala",
        "sql",
        "toml",
      ],
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
    algolia: {
      // The application ID provided by Algolia
      appId: "XKVLO8P882",

      // Public API key: it is safe to commit it
      apiKey: "e78ac2f60eba60a4c4a353c0702cc7bb",

      indexName: "d1voxqa8i8phxc-cloudfront",

      // Optional: see doc section below
      contextualSearch: true,

      // Optional: Specify domains where the navigation should occur through window.location instead on history.push.
      // Useful when our Algolia config crawls multiple documentation sites and we want to navigate with window.location.href to them.
      externalUrlRegex: "/api_reference",
    },
  } satisfies Preset.ThemeConfig,

  plugins: [
    tailwindPlugin,
    // Classic ML docs plugin
    [
      "@docusaurus/plugin-content-docs",
      {
        id: "classic-ml",
        path: "docs/classic-ml",
        routeBasePath: "ml",
        sidebarPath: "./sidebarsClassicML.ts",
        async sidebarItemsGenerator({
          defaultSidebarItemsGenerator,
          ...args
        }) {
          const sidebarItems = await defaultSidebarItemsGenerator(args);
          return postProcessSidebar(sidebarItems);
        },
      },
    ],
    // GenAI docs plugin
    [
      "@docusaurus/plugin-content-docs",
      {
        id: "genai",
        path: "docs/genai",
        routeBasePath: "genai",
        sidebarPath: "./sidebarsGenAI.ts",
        async sidebarItemsGenerator({
          defaultSidebarItemsGenerator,
          ...args
        }) {
          const sidebarItems = await defaultSidebarItemsGenerator(args);
          return postProcessSidebar(sidebarItems);
        },
      },
    ],
    [
      "@docusaurus/plugin-client-redirects",
      {
        redirects: redirectsConfig,
      },
    ],
  ],
};

export default config;