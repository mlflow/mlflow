import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";
import {
  postProcessSidebar,
  apiReferencePrefix,
} from "./docusaurusConfigUtils";
import { onRouteDidUpdate } from './src/docusaurus.theme';


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

  presets: [
    [
      "classic",
      {
        docs: false, // Disable default docs plugin
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
        // Classic ML docs
        {
          type: "docSidebar",
          sidebarId: "classicMLSidebar",
          position: "left",
          label: "ML Docs",
          docsPluginId: "classic-ml",
          className: "ml-docs-link",
        },
        // GenAI docs
        {
          type: "docSidebar",
          sidebarId: "genAISidebar",
          position: "left",
          label: "GenAI Docs",
          docsPluginId: "genai",
          className: "genai-docs-link",
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
            to: "/genai/tracing",
            from: ["/llms/tracing", "/tracing"],
          },
          {
            to: "/ml", //TODO: update
            from: ["/tracking/data-api/index", "/tracking/data-api", "/dataset"],
          },
          {
            to: "/ml", //TODO: update
            from: ["/models", "/model"],
          },
          {
            to: "/genai/tracing", //TODO: update
            from: ["/llms/bedrock/autologging", "/tracing/integrations/bedrock"],
          },
          {
            to: "/genai/getting-started", //TODO: verify location
            from: ["/getting-started/community-edition", "/getting-started/databricks-trial"],
          },
          {
            to: "/genai/tracing", //TODO: update
            from: ["/llms/tracing/contribute", "/tracing/integrations/contribute"],
          },
          {
            to: "/genai/tracing", //TODO: update
            from: ["/llms/tracing/notebooks/jupyter-trace-demo", "/tracing/tutorials/jupyter-trace-demo"],
          },
          {
            to: "/genai/tracing/features", //TODO: verify location
            from: ["/llms/tracing/overview", "/tracing/tutorials/concept"],
          },
          {
            to: "/genai/tracing", //TODO: verify location
            from: ["/llms/tracing/search-traces", "/tracing/api/search"],
          },
          {
            to: "/genai/tracing/features", //TODO: update
            from: ["/llms/tracing/tracing-schema", "/tracing/tracing-schema"],
          },
          {
            to: "/ml", //TODO: update
            from: ["/registry", "/model-registry"],
          },
          {
            to: "/genai/governance/ai-gateway",
            from: ["/llms/gateway/index", "/llms/gateway", "/llms/deployments"],
          },
          {
            to: "/genai/governance/ai-gateway", //TODO: verify location
            from: ["/llms/gateway/guides/index", "/llms/gateway/guide", "/llms/deployments/guides"],
          },
          {
            to: "/genai/governance/ai-gateway", //TODO: update
            from: ["/llms/gateway/guides/step1-create-gateway", "/llms/deployments/guides/step1-create-deployments"],
          },
          {
            to: "/genai/governance/ai-gateway", //TODO: update
            from: ["/llms/gateway/guides/step2-query-gateway", "/llms/deployments/guides/step2-query-deployments"],
          },
        ],
      },
    ],
  ],
};

export default config;
