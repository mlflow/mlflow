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
  '/deep-learning/keras/guide/',
  '/deep-learning/keras/quickstart/',
  '/deep-learning/pytorch/',
  '/deep-learning/pytorch/guide/',
  '/deep-learning/pytorch/quickstart/',
  '/deep-learning/sentence-transformers/',
  '/deep-learning/sentence-transformers/guide/',
  '/deep-learning/spacy/',
  '/deep-learning/spacy/guide/',
  '/deep-learning/tensorflow/',
  '/deep-learning/tensorflow/guide/',
  '/deep-learning/tensorflow/quickstart/',
  '/deep-learning/transformers/',
  '/deep-learning/transformers/guide/',
  '/deep-learning/transformers/large-models/',
  '/deep-learning/transformers/task/',
  '/deep-learning/transformers/tutorials/',
  '/deployment/',
  '/deployment/deploy-model-locally/',
  '/deployment/deploy-model-to-kubernetes/',
  '/deployment/deploy-model-to-kubernetes/tutorial/',
  '/deployment/deploy-model-to-sagemaker/',
  '/deployment/tutorial/',
  '/docker/',
  '/evaluation/',
  '/getting-started/',
  '/getting-started/databricks-trial/',
  '/getting-started/hyperparameter-tuning/',
  '/getting-started/logging-first-model/',
  '/getting-started/logging-first-model/notebooks/',
  '/getting-started/registering-first-model/',
  '/getting-started/running-notebooks/',
  '/getting-started/tracking-server-overview/',
  '/model-registry/',
  '/model/',
  '/model/dependencies/',
  '/model/models-from-code/',
  '/model/signatures/',
  '/plugins/',
  '/projects/',
  '/search/search-experiments/',
  '/search/search-models/',
  '/search/search-runs/',
  '/tracking/',
  '/tracking/artifact-stores/',
  '/tracking/autolog/',
  '/tracking/backend-stores/',
  '/tracking/quickstart/',
  '/tracking/server/',
  '/tracking/system-metrics/',
  '/tracking/tracking-api/',
  '/tracking/tutorials/local-database/',
  '/tracking/tutorials/remote-server/',
  '/traditional-ml/',
  '/traditional-ml/prophet/',
  '/traditional-ml/prophet/guide/',
  '/traditional-ml/sklearn/',
  '/traditional-ml/sklearn/guide/',
  '/traditional-ml/sklearn/quickstart/',
  '/traditional-ml/sparkml/',
  '/traditional-ml/sparkml/guide/',
  '/traditional-ml/tutorials/creating-custom-pyfunc/',
  '/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/',
  '/traditional-ml/tutorials/creating-custom-pyfunc/part1-named-flavors/',
  '/traditional-ml/tutorials/creating-custom-pyfunc/part2-pyfunc-components/',
  '/traditional-ml/tutorials/hyperparameter-tuning/',
  '/traditional-ml/tutorials/hyperparameter-tuning/notebooks/',
  '/traditional-ml/tutorials/hyperparameter-tuning/part1-child-runs/',
  '/traditional-ml/tutorials/hyperparameter-tuning/part2-logging-plots/',
  '/traditional-ml/tutorials/serving-multiple-models-with-pyfunc/',
  '/traditional-ml/xgboost/',
  '/traditional-ml/xgboost/guide/',
  '/traditional-ml/xgboost/quickstart/',
  '/tutorials-and-examples/',
];

const genAIPaths = [
  // Note: old /llms/ paths now map to /genai/flavors/
  // old /tracing/ paths now map to /genai/tracing/
  // old /prompts/ paths now map to /genai/prompt-version-mgmt/
];

// More specific redirects for LLM/GenAI content since the structure changed significantly
const llmToGenAIRedirects = [
  {
    to: "/genai/flavors/",
    from: "/llms/"
  },
  {
    to: "/genai/flavors/chat-model-guide/",
    from: "/llms/chat-model-guide/"
  },
  {
    to: "/genai/flavors/chat-model-intro/",
    from: "/llms/chat-model-intro/"
  },
  {
    to: "/genai/flavors/custom-pyfunc-for-llms/",
    from: "/llms/custom-pyfunc-for-llms/"
  },
  {
    to: "/genai/flavors/dspy/",
    from: "/llms/dspy/"
  },
  {
    to: "/genai/flavors/langchain/",
    from: "/llms/langchain/"
  },
  {
    to: "/genai/flavors/langchain/autologging",
    from: "/llms/langchain/autologging"
  },
  {
    to: "/genai/flavors/langchain/guide/",
    from: "/llms/langchain/guide/"
  },
  {
    to: "/genai/flavors/llama-index/",
    from: "/llms/llama-index/"
  },
  {
    to: "/genai/flavors/openai/",
    from: "/llms/openai/"
  },
  {
    to: "/genai/flavors/openai/autologging/",
    from: "/llms/openai/autologging/"
  },
  {
    to: "/genai/flavors/openai/guide/",
    from: "/llms/openai/guide/"
  },
  {
    to: "/genai/flavors/openai/notebooks/",
    from: "/llms/openai/notebooks/"
  },
  {
    to: "/genai/tracing/",
    from: "/tracing/"
  },
  {
    to: "/genai/tracing/faq",
    from: "/tracing/faq"
  },
  {
    to: "/genai/tracing/integrations/",
    from: "/tracing/integrations/"
  },
  {
    to: "/genai/tracing/integrations/contribute",
    from: "/tracing/integrations/contribute"
  },
  {
    to: "/genai/tracing/integrations/listing/anthropic",
    from: "/tracing/integrations/anthropic"
  },
  {
    to: "/genai/tracing/integrations/listing/autogen",
    from: "/tracing/integrations/autogen"
  },
  {
    to: "/genai/tracing/integrations/listing/bedrock",
    from: "/tracing/integrations/bedrock"
  },
  {
    to: "/genai/tracing/integrations/listing/crewai",
    from: "/tracing/integrations/crewai"
  },
  {
    to: "/genai/tracing/integrations/listing/deepseek",
    from: "/tracing/integrations/deepseek"
  },
  {
    to: "/genai/tracing/integrations/listing/dspy",
    from: "/tracing/integrations/dspy"
  },
  {
    to: "/genai/tracing/integrations/listing/gemini",
    from: "/tracing/integrations/gemini"
  },
  {
    to: "/genai/tracing/integrations/listing/groq",
    from: "/tracing/integrations/groq"
  },
  {
    to: "/genai/tracing/integrations/listing/instructor",
    from: "/tracing/integrations/instructor"
  },
  {
    to: "/genai/tracing/integrations/listing/langchain",
    from: "/tracing/integrations/langchain"
  },
  {
    to: "/genai/tracing/integrations/listing/langgraph",
    from: "/tracing/integrations/langgraph"
  },
  {
    to: "/genai/tracing/integrations/listing/litellm",
    from: "/tracing/integrations/litellm"
  },
  {
    to: "/genai/tracing/integrations/listing/llama_index",
    from: "/tracing/integrations/llama_index"
  },
  {
    to: "/genai/tracing/integrations/listing/mistral",
    from: "/tracing/integrations/mistral"
  },
  {
    to: "/genai/tracing/integrations/listing/ollama",
    from: "/tracing/integrations/ollama"
  },
  {
    to: "/genai/tracing/integrations/listing/openai",
    from: "/tracing/integrations/openai"
  },
  {
    to: "/genai/tracing/integrations/listing/openai-agent",
    from: "/tracing/integrations/openai-agent"
  },
  {
    to: "/genai/tracing/integrations/listing/swarm",
    from: "/tracing/integrations/swarm"
  },
  {
    to: "/genai/tracing/integrations/listing/txtai",
    from: "/tracing/integrations/txtai"
  },
  {
    to: "/genai/tracing/prod-tracing",
    from: "/tracing/production"
  },
  {
    to: "/genai/tracing/session",
    from: "/tracing/session"
  },
  {
    to: "/genai/tracing/data-model",
    from: "/tracing/tracing-schema"
  },
  {
    to: "/genai/tracing/tracing-101",
    from: "/tracing/tutorials/concept"
  },
  {
    to: "/genai/tutorials/jupyter-trace-demo",
    from: "/tracing/tutorials/jupyter-trace-demo"
  },
  {
    to: "/genai/tracing/observe-with-traces/ui",
    from: "/tracing/ui"
  },
  {
    to: "/genai/prompt-version-mgmt/",
    from: "/prompts/"
  },
  {
    to: "/genai/eval-monitor/",
    from: "/llms/llm-evaluate/"
  },
  {
    to: "/genai/governance/ai-gateway/",
    from: "/llms/deployments/"
  },
  {
    to: "/genai/governance/ai-gateway/guides/",
    from: "/llms/deployments/guides/"
  },
];

const specialRedirects = [
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
  {
    to: "/ml/tracking/",
    from: ["/tracking/data-api/index", "/tracking/data-api"]
  },
  {
    to: "/ml/model-registry/",
    from: ["/models", "/registry"]
  },
  {
    to: "/genai/getting-started/",
    from: ["/getting-started/community-edition"]
  },
];

const redirectsConfig = [
  ...createMLRedirects(mlPaths),
  ...createGenAIRedirects(genAIPaths),
  ...llmToGenAIRedirects,
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