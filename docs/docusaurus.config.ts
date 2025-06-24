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
          type: "custom-docsDropdown",
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
          className: "github-link",
        },
      ],
    },
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: true,
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
        redirects: [
          {
            to: "/",
            from: ["/new-features"],
          },
          // Redirect mlflow 3 pages
          {
            to: "/genai/mlflow-3",
            from: ["/mlflow-3", "/ml/mlflow-3"],
          },
          {
            to: "/genai/mlflow-3/deep-learning",
            from: ["/mlflow-3/deep-learning"],
          },
          {
            to: "/genai/mlflow-3/genai-agent",
            from: ["/mlflow-3/genai-agent"],
          },
          // GenAI/LLM Related Redirects
          {
            to: "/genai/tracing",
            from: ["/llms/llm-tracking", "/tracing", "/llms/tracing", "/tracing/api/how-to"],
          },
          {
            to: "/genai/tracing/faq",
            from: ["/tracing/faq"],
          },
          {
            to: "/genai/tracing/data-model",
            from: ["/tracing/tracing-schema", "/llms/tracing/tracing-schema"],
          },
          {
            to: "/genai/tracing/prod-tracing",
            from: ["/tracing/production"],
          },
          {
            to: "/genai/tracing/search-traces",
            from: ["/tracing/api/search", "/llms/tracing/search-traces"],
          },
          {
            to: "/genai/tracing/app-instrumentation/manual-tracing/low-level-api",
            from: ["/tracing/api/client"],
          },
          {
            to: "/genai/tracing/app-instrumentation/manual-tracing",
            from: ["/tracing/api/manual-instrumentation"],
          },
          {
            to: "/genai/tracing/app-instrumentation",
            from: ["/tracing/api"],
          },
          {
            to: "/genai/tracing/observe-with-traces/ui",
            from: ["/tracing/ui"],
          },
          {
            to: "/genai/tracing/integrations",
            from: ["/tracing/integrations"],
          },
          {
            to: "/genai/tracing/integrations/contribute",
            from: ["/tracing/integrations/contribute", "/llms/tracing/contribute"],
          },
          {
            to: "/genai/tracing/integrations/listing/anthropic",
            from: ["/tracing/integrations/anthropic"],
          },
          {
            to: "/genai/tracing/integrations/listing/autogen",
            from: ["/tracing/integrations/autogen"],
          },
          {
            to: "/genai/tracing/integrations/listing/bedrock",
            from: ["/tracing/integrations/bedrock", "/llms/bedrock/autologging"],
          },
          {
            to: "/genai/tracing/integrations/listing/crewai",
            from: ["/tracing/integrations/crewai"],
          },
          {
            to: "/genai/tracing/integrations/listing/deepseek",
            from: ["/tracing/integrations/deepseek"],
          },
          {
            to: "/genai/tracing/integrations/listing/dspy",
            from: ["/tracing/integrations/dspy"],
          },
          {
            to: "/genai/tracing/integrations/listing/gemini",
            from: ["/tracing/integrations/gemini"],
          },
          {
            to: "/genai/tracing/integrations/listing/groq",
            from: ["/tracing/integrations/groq"],
          },
          {
            to: "/genai/tracing/integrations/listing/instructor",
            from: ["/tracing/integrations/instructor"],
          },
          {
            to: "/genai/tracing/integrations/listing/langchain",
            from: ["/tracing/integrations/langchain"],
          },
          {
            to: "/genai/tracing/integrations/listing/langgraph",
            from: ["/tracing/integrations/langgraph"],
          },
          {
            to: "/genai/tracing/integrations/listing/litellm",
            from: ["/tracing/integrations/litellm"],
          },
          {
            to: "/genai/tracing/integrations/listing/llama_index",
            from: ["/tracing/integrations/llama_index"],
          },
          {
            to: "/genai/tracing/integrations/listing/mistral",
            from: ["/tracing/integrations/mistral"],
          },
          {
            to: "/genai/tracing/integrations/listing/ollama",
            from: ["/tracing/integrations/ollama"],
          },
          {
            to: "/genai/tracing/integrations/listing/openai",
            from: ["/tracing/integrations/openai"],
          },
          {
            to: "/genai/tracing/integrations/listing/openai-agent",
            from: ["/tracing/integrations/openai-agent"],
          },
          {
            to: "/genai/tracing/integrations/listing/swarm",
            from: ["/tracing/integrations/swarm"],
          },
          {
            to: "/genai/tracing/integrations/listing/txtai",
            from: ["/tracing/integrations/txtai"],
          },
          {
            to: "/genai",
            from: ["/tracing/tutorials/jupyter-trace-demo", "/llms/tracing/notebooks/jupyter-trace-demo"],
          },
          {
            to: "/genai",
            from: ["/tracing/tutorials"],
          },

          // LLM Flavors Redirects
          {
            to: "/genai/overview",
            from: [
              "/llms/rag",
              "/llms/rag/notebooks",
              "/llms/rag/notebooks/mlflow-e2e-evaluation",
              "/llms/rag/notebooks/question-generation-retrieval-evaluation",
              "/llms/rag/notebooks/retriever-evaluation-tutorial"
            ]
          },
          {
            to: "/genai",
            from: ["/llms", "/llms/index"],
          },
          {
            to: "/genai/flavors/chat-model-guide",
            from: ["/llms/chat-model-guide"],
          },
          {
            to: "/genai/flavors/chat-model-guide/chat-model-tool-calling",
            from: ["/llms/notebooks/chat-model-tool-calling"],
          },
          {
            to: "/genai/flavors/chat-model-intro",
            from: ["/llms/chat-model-intro"],
          },
          {
            to: "/genai/flavors/custom-pyfunc-for-llms",
            from: ["/llms/custom-pyfunc-for-llms", "/llms/custom-pyfunc-for-llms/notebooks"],
          },
          {
            to: "/genai/flavors/custom-pyfunc-for-llms/notebooks/custom-pyfunc-advanced-llm",
            from: ["/llms/custom-pyfunc-for-llms/notebooks/custom-pyfunc-advanced-llm"],
          },
          {
            to: "/genai/flavors/dspy",
            from: ["/llms/dspy"],
          },
          {
            to: "/genai/flavors/dspy/notebooks/dspy_quickstart",
            from: ["/llms/dspy/notebooks/dspy_quickstart"],
          },
          {
            to: "/genai/flavors/dspy/optimizer",
            from: ["/llms/dspy/optimizer"],
          },
          {
            to: "/genai/flavors/langchain",
            from: ["/llms/langchain"],
          },
          {
            to: "/genai/flavors/langchain/autologging",
            from: ["/llms/langchain/autologging"],
          },
          {
            to: "/genai/flavors/langchain/guide",
            from: ["/llms/langchain/guide"],
          },
          {
            to: "/genai/flavors/langchain/notebooks/langchain-quickstart",
            from: ["/llms/langchain/notebooks/langchain-quickstart"],
          },
          {
            to: "/genai/flavors/langchain/notebooks/langchain-retriever",
            from: ["/llms/langchain/notebooks/langchain-retriever"],
          },
          {
            to: "/genai/flavors/llama-index",
            from: ["/llms/llama-index"],
          },
          {
            to: "/genai/flavors/llama-index/notebooks/llama_index_quickstart",
            from: ["/llms/llama-index/notebooks/llama_index_quickstart"],
          },
          {
            to: "/genai/flavors/llama-index/notebooks/llama_index_workflow_tutorial",
            from: ["/llms/llama-index/notebooks/llama_index_workflow_tutorial"],
          },
          {
            to: "/genai/flavors/openai",
            from: ["/llms/openai"],
          },
          {
            to: "/genai/flavors/openai/autologging",
            from: ["/llms/openai/autologging"],
          },
          {
            to: "/genai/flavors/openai/guide",
            from: ["/llms/openai/guide"],
          },
          {
            to: "/genai/flavors/openai/notebooks",
            from: ["/llms/openai/notebooks"],
          },
          {
            to: "/genai/flavors/openai/notebooks/openai-chat-completions",
            from: ["/llms/openai/notebooks/openai-chat-completions"],
          },
          {
            to: "/genai/flavors/openai/notebooks/openai-code-helper",
            from: ["/llms/openai/notebooks/openai-code-helper"],
          },
          {
            to: "/genai/flavors/openai/notebooks/openai-embeddings-generation",
            from: ["/llms/openai/notebooks/openai-embeddings-generation"],
          },
          {
            to: "/genai/flavors/openai/notebooks/openai-quickstart",
            from: ["/llms/openai/notebooks/openai-quickstart"],
          },

          // Evaluation and Monitoring Redirects
          {
            to: "/genai/eval-monitor/llm-evaluation",
            from: ["/llms/llm-evaluate"],
          },
          {
            to: "/genai/eval-monitor/notebooks",
            from: [
              "/llms/llm-evaluate/notebooks",
              "/llms/llm-evaluate/notebooks/huggingface-evaluation",
              "/llms/llm-evaluate/notebooks/question-answering-evaluation",
              "/llms/llm-evaluate/notebooks/rag-evaluation",
              "/llms/llm-evaluate/notebooks/rag-evaluation-llama2",
            ],
          },

          // Prompt Management Redirects
          {
            to: "/genai/prompt-version-mgmt/prompt-engineering",
            from: ["/llms/prompt-engineering"],
          },
          {
            to: "/genai/prompt-version-mgmt/prompt-registry",
            from: ["/prompts"],
          },
          {
            to: "/genai/prompt-version-mgmt/prompt-registry/manage-prompt-lifecycles-with-aliases",
            from: ["/prompts/cm"],
          },
          {
            to: "/genai/prompt-version-mgmt/prompt-registry/evaluate-prompts",
            from: ["/prompts/evaluate"],
          },
          {
            to: "/genai/prompt-version-mgmt/prompt-registry/log-with-model",
            from: ["/prompts/run-and-model"],
          },
          {
            to: "/genai/prompt-version-mgmt/optimize-prompts",
            from: ["/genai/prompt-version-mgmt/prompt-registry/optimize-prompts"],
          },

          // Governance and Deployments Redirects
          {
            to: "/genai/governance/ai-gateway",
            from: ["/llms/deployments/", "/llms/gateway/index", "/llms/gateway"],
          },
          {
            to: "/genai/governance/ai-gateway/guides",
            from: ["/llms/deployments/guides", "/llms/gateway/guides/index", "/llms/gateway/guide"],
          },
          {
            to: "/genai/governance/ai-gateway/guides/step1-create-deployments",
            from: ["/llms/deployments/guides/step1-create-deployments", "/llms/gateway/guides/step1-create-gateway"],
          },
          {
            to: "/genai/governance/ai-gateway/guides/step2-query-deployments",
            from: ["/llms/deployments/guides/step2-query-deployments", "/llms/gateway/guides/step2-query-gateway"],
          },
          {
            to: "/genai/governance/unity-catalog",
            from: ["/llms/deployments/uc_integration"],
          },

          // Traditional ML and Core MLflow Redirects
          {
            to: "/ml",
            from: ["/introduction"],
          },
          {
            to: "/ml/auth",
            from: ["/auth"],
          },
          {
            to: "/ml/community-model-flavors",
            from: ["/community-model-flavors"],
          },
          {
            to: "/ml/dataset",
            from: ["/dataset", "/tracking/data-api/index", "/tracking/data-api"],
          },
          {
            to: "/ml/deep-learning",
            from: ["/deep-learning"],
          },
          {
            to: "/ml/deep-learning/keras",
            from: ["/deep-learning/keras"],
          },
          {
            to: "/ml/deep-learning/keras/quickstart/quickstart-keras",
            from: ["/deep-learning/keras/quickstart/quickstart_keras"],
          },
          {
            to: "/ml/deep-learning/pytorch",
            from: ["/deep-learning/pytorch"],
          },
          {
            to: "/ml/deep-learning/pytorch/guide",
            from: ["/deep-learning/pytorch/guide"],
          },
          {
            to: "/ml/deep-learning/pytorch/quickstart/quickstart-pytorch",
            from: ["/deep-learning/pytorch/quickstart/pytorch_quickstart"],
          },
          {
            to: "/ml/deep-learning/sentence-transformers",
            from: ["/llms/sentence-transformers"],
          },
          {
            to: "/ml/deep-learning/sentence-transformers/tutorials/",
            from: ["/llms/sentence-transformers/tutorials"],
          },
          {
            to: "/ml/deep-learning/sentence-transformers/tutorials/paraphrase-mining/paraphrase-mining-sentence-transformers",
            from: [
              "/llms/sentence-transformers/tutorials/paraphrase-mining/paraphrase-mining-sentence-transformers"
            ],
          },
          {
            to: "/ml/deep-learning/sentence-transformers/tutorials/quickstart/sentence-transformers-quickstart",
            from: ["/llms/sentence-transformers/tutorials/quickstart/sentence-transformers-quickstart"],
          },
          {
            to: "/ml/deep-learning/sentence-transformers/tutorials/semantic-search/semantic-search-sentence-transformers",
            from: ["/llms/sentence-transformers/tutorials/semantic-search/semantic-search-sentence-transformers"],
          },
          {
            to: "/ml/deep-learning/sentence-transformers/tutorials/semantic-similarity/semantic-similarity-sentence-transformers",
            from: ["/llms/sentence-transformers/tutorials/semantic-similarity/semantic-similarity-sentence-transformers"],
          },
          {
            to: "/ml/deep-learning/sentence-transformers/guide",
            from: ["/llms/sentence-transformers/guide"],
          },
          {
            to: "/ml/deep-learning/tensorflow",
            from: ["/deep-learning/tensorflow"],
          },
          {
            to: "/ml/deep-learning/tensorflow/guide",
            from: ["/deep-learning/tensorflow/guide"],
          },
          {
            to: "/ml/deep-learning/tensorflow/quickstart/quickstart-tensorflow",
            from: ["/deep-learning/tensorflow/quickstart/quickstart_tensorflow"],
          },
          {
            to: "/ml/deep-learning/transformers",
            from: ["/llms/transformers"],
          },
          {
            to: "/ml/deep-learning/transformers/guide",
            from: ["/llms/transformers/guide"],
          },
          {
            to: "/ml/deep-learning/transformers/large-models",
            from: ["/llms/transformers/large-models"],
          },
          {
            to: "/ml/deep-learning/transformers/task",
            from: ["/llms/transformers/task"],
          },
          {
            to: "/ml/deep-learning/transformers/tutorials",
            from: ["/llms/transformers/tutorials"],
          },
          {
            to: "/ml/deep-learning/transformers/tutorials/audio-transcription/whisper",
            from: ["/llms/transformers/tutorials/audio-transcription/whisper"],
          },
          {
            to: "/ml/deep-learning/transformers/tutorials/conversational/conversational-model",
            from: ["/llms/transformers/tutorials/conversational/conversational-model"],
          },
          {
            to: "/ml/deep-learning/transformers/tutorials/conversational/pyfunc-chat-model",
            from: ["/llms/transformers/tutorials/conversational/pyfunc-chat-model"],
          },
          {
            to: "/ml/deep-learning/transformers/tutorials/fine-tuning/transformers-fine-tuning",
            from: ["/llms/transformers/tutorials/fine-tuning/transformers-fine-tuning"],
          },
          {
            to: "/ml/deep-learning/transformers/tutorials/fine-tuning/transformers-peft",
            from: ["/llms/transformers/tutorials/fine-tuning/transformers-peft"],
          },
          {
            to: "/ml/deep-learning/transformers/tutorials/prompt-templating/prompt-templating",
            from: ["/llms/transformers/tutorials/prompt-templating/prompt-templating"],
          },
          {
            to: "/ml/deep-learning/transformers/tutorials/text-generation/text-generation",
            from: ["/llms/transformers/tutorials/text-generation/text-generation"],
          },
          {
            to: "/ml/deep-learning/transformers/tutorials/translation/component-translation",
            from: ["/llms/transformers/tutorials/translation/component-translation"],
          },
          {
            to: "/ml/deployment",
            from: ["/deployment"],
          },
          {
            to: "/ml/deployment/deploy-model-locally",
            from: ["/deployment/deploy-model-locally"],
          },
          {
            to: "/ml/deployment/deploy-model-to-kubernetes",
            from: ["/deployment/deploy-model-to-kubernetes"],
          },
          {
            to: "/ml/deployment/deploy-model-to-kubernetes/tutorial",
            from: ["/deployment/deploy-model-to-kubernetes/tutorial"],
          },
          {
            to: "/ml/deployment/deploy-model-to-sagemaker",
            from: ["/deployment/deploy-model-to-sagemaker"],
          },
          {
            to: "/ml/docker",
            from: ["/docker"],
          },
          {
            to: "/ml/evaluation",
            from: ["/model-evaluation"],
          },
          {
            to: "/ml/getting-started",
            from: ["/getting-started"],
          },
          {
            to: "/ml/getting-started/databricks-trial",
            from: ["/getting-started/databricks-trial", "/getting-started/community-edition"],
          },
          {
            to: "/ml/getting-started/hyperparameter-tuning",
            from: ["/getting-started/quickstart-2"],
          },
          {
            to: "/ml/getting-started/logging-first-model",
            from: ["/getting-started/logging-first-model"],
          },
          {
            to: "/ml/getting-started/logging-first-model/notebooks",
            from: ["/getting-started/logging-first-model/notebooks"],
          },
          {
            to: "/ml/getting-started/logging-first-model/notebooks/logging-first-model",
            from: ["/getting-started/logging-first-model/notebooks/logging-first-model"],
          },
          {
            to: "/ml/getting-started/logging-first-model/step1-tracking-server",
            from: ["/getting-started/logging-first-model/step1-tracking-server"],
          },
          {
            to: "/ml/getting-started/logging-first-model/step2-mlflow-client",
            from: ["/getting-started/logging-first-model/step2-mlflow-client"],
          },
          {
            to: "/ml/getting-started/logging-first-model/step3-create-experiment",
            from: ["/getting-started/logging-first-model/step3-create-experiment"],
          },
          {
            to: "/ml/getting-started/logging-first-model/step4-experiment-search",
            from: ["/getting-started/logging-first-model/step4-experiment-search"],
          },
          {
            to: "/ml/getting-started/logging-first-model/step5-synthetic-data",
            from: ["/getting-started/logging-first-model/step5-synthetic-data"],
          },
          {
            to: "/ml/getting-started/logging-first-model/step6-logging-a-run",
            from: ["/getting-started/logging-first-model/step6-logging-a-run"],
          },
          {
            to: "/ml/getting-started/registering-first-model",
            from: ["/getting-started/registering-first-model"],
          },
          {
            to: "/ml/getting-started/registering-first-model/step1-register-model",
            from: ["/getting-started/registering-first-model/step1-register-model"],
          },
          {
            to: "/ml/getting-started/registering-first-model/step2-explore-registered-model",
            from: ["/getting-started/registering-first-model/step2-explore-registered-model"],
          },
          {
            to: "/ml/getting-started/registering-first-model/step3-load-model",
            from: ["/getting-started/registering-first-model/step3-load-model"],
          },
          {
            to: "/ml/getting-started/running-notebooks",
            from: ["/getting-started/running-notebooks"],
          },
          {
            to: "/ml/getting-started/tracking-server-overview",
            from: ["/getting-started/tracking-server-overview"],
          },
          {
            to: "/ml/model-registry",
            from: ["/model-registry", "/registry"],
          },
          {
            to: "/ml/model",
            from: ["/model", "/models"],
          },
          {
            to: "/ml/model/dependencies",
            from: ["/model/dependencies"],
          },
          {
            to: "/ml/model/models-from-code",
            from: ["/model/models-from-code"],
          },
          {
            to: "/ml/model/notebooks/signature_examples",
            from: ["/model/notebooks/signature_examples"],
          },
          {
            to: "/ml/model/python_model",
            from: ["/model/python_model"],
          },
          {
            to: "/ml/model/signatures",
            from: ["/model/signatures"],
          },
          {
            to: "/ml/plugins",
            from: ["/plugins"],
          },
          {
            to: "/ml/projects",
            from: ["/projects"],
          },
          {
            to: "/ml/search/search-experiments",
            from: ["/search-experiments"],
          },
          {
            to: "/ml/search/search-runs",
            from: ["/search-runs"],
          },
          {
            to: "/ml/tracking",
            from: ["/tracking"],
          },
          {
            to: "/ml/tracking/artifact-stores",
            from: ["/tracking/artifacts-stores"],
          },
          {
            to: "/ml/tracking/autolog",
            from: ["/tracking/autolog"],
          },
          {
            to: "/ml/tracking/backend-stores",
            from: ["/tracking/backend-stores"],
          },
          {
            to: "/ml/tracking/quickstart",
            from: ["/getting-started/intro-quickstart", "/getting-started/intro-quickstart/notebooks", "/quickstart_drilldown"],
          },
          {
            to: "/ml/tracking/quickstart/notebooks/tracking_quickstart",
            from: ["/getting-started/intro-quickstart/notebooks/tracking_quickstart"],
          },
          {
            to: "/ml/tracking/server",
            from: ["/tracking/server"],
          },
          {
            to: "/ml/tracking/system-metrics",
            from: ["/system-metrics"],
          },
          {
            to: "/ml/tracking/tracking-api",
            from: ["/tracking/tracking-api"],
          },
          {
            to: "/ml/tracking/tutorials/local-database",
            from: ["/tracking/tutorials/local-database"],
          },
          {
            to: "/ml/tracking/tutorials/remote-server",
            from: ["/tracking/tutorials/remote-server"],
          },
          {
            to: "/ml/traditional-ml",
            from: ["/traditional-ml"],
          },
          {
            to: "/ml/traditional-ml/tutorials/creating-custom-pyfunc",
            from: ["/traditional-ml/creating-custom-pyfunc"],
          },
          {
            to: "/ml/traditional-ml/tutorials/creating-custom-pyfunc/notebooks",
            from: ["/traditional-ml/creating-custom-pyfunc/notebooks"],
          },
          {
            to: "/ml/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/basic-pyfunc",
            from: ["/traditional-ml/creating-custom-pyfunc/notebooks/basic-pyfunc"],
          },
          {
            to: "/ml/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/introduction",
            from: ["/traditional-ml/creating-custom-pyfunc/notebooks/introduction"],
          },
          {
            to: "/ml/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/override-predict",
            from: ["/traditional-ml/creating-custom-pyfunc/notebooks/override-predict"],
          },
          {
            to: "/ml/traditional-ml/tutorials/creating-custom-pyfunc/part1-named-flavors",
            from: ["/traditional-ml/creating-custom-pyfunc/part1-named-flavors"],
          },
          {
            to: "/ml/traditional-ml/tutorials/creating-custom-pyfunc/part2-pyfunc-components",
            from: ["/traditional-ml/creating-custom-pyfunc/part2-pyfunc-components"],
          },
          {
            to: "/ml/traditional-ml/tutorials/hyperparameter-tuning",
            from: ["/traditional-ml/hyperparameter-tuning-with-child-runs"],
          },
          {
            to: "/ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks",
            from: ["/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks"],
          },
          {
            to: "/ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/hyperparameter-tuning-with-child-runs",
            from: ["/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/hyperparameter-tuning-with-child-runs"],
          },
          {
            to: "/ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/logging-plots-in-mlflow",
            from: ["/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/logging-plots-in-mlflow"],
          },
          {
            to: "/ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/parent-child-runs",
            from: ["/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/parent-child-runs"],
          },
          {
            to: "/ml/traditional-ml/tutorials/hyperparameter-tuning/part1-child-runs",
            from: ["/traditional-ml/hyperparameter-tuning-with-child-runs/part1-child-runs"],
          },
          {
            to: "/ml/traditional-ml/tutorials/hyperparameter-tuning/part2-logging-plots",
            from: ["/traditional-ml/hyperparameter-tuning-with-child-runs/part2-logging-plots"],
          },
          {
            to: "/ml/traditional-ml/tutorials/serving-multiple-models-with-pyfunc",
            from: ["/traditional-ml/serving-multiple-models-with-pyfunc"],
          },
          {
            to: "/ml/traditional-ml/tutorials/serving-multiple-models-with-pyfunc/notebooks/MME_Tutorial",
            from: ["/traditional-ml/serving-multiple-models-with-pyfunc/notebooks/MME_Tutorial"],
          },
          {
            to: "/ml/tutorials-and-examples",
            from: ["/tutorials-and-examples"],
          },
        ],
      },
    ],
  ],
};

export default config;
