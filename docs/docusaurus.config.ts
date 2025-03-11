import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";
import {
  postProcessSidebar,
  apiReferencePrefix,
} from "./docusaurusConfigUtils";

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
        docs: {
          routeBasePath: "/",
          sidebarPath: "./sidebars.ts",
          async sidebarItemsGenerator({
            defaultSidebarItemsGenerator,
            ...args
          }) {
            const sidebarItems = await defaultSidebarItemsGenerator(args);
            return postProcessSidebar(sidebarItems);
          },
        },
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
        {
          type: "docSidebar",
          sidebarId: "docsSidebar",
          position: "left",
          label: "Docs",
        },
        {
          to: `${apiReferencePrefix()}api_reference/`,
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
    [
      "@docusaurus/plugin-client-redirects",
      {
        redirects: [
          {
            to: "/tracing",
            from: ["/llms/tracing"],
          },
          {
            to: "/dataset",
            from: ["/tracking/data-api/index", "/tracking/data-api"],
          },
          {
            to: "/model",
            from: ["/models"],
          },
          {
            to: "/tracing/integrations/bedrock",
            from: ["/llms/bedrock/autologging"],
          },
          {
            to: "/getting-started/databricks-trial",
            from: ["/getting-started/community-edition"],
          },
          {
            to: "/tracing/integrations/contribute",
            from: ["/llms/tracing/contribute"],
          },
          {
            to: "/tracing/tutorials/jupyter-trace-demo",
            from: ["/llms/tracing/notebooks/jupyter-trace-demo"],
          },
          {
            to: "/tracing/tutorials/concept",
            from: ["/llms/tracing/overview"],
          },
          {
            to: "/tracing/api/search",
            from: ["/llms/tracing/search-traces"],
          },
          {
            to: "/tracing/tracing-schema",
            from: ["/llms/tracing/tracing-schema"],
          },
          {
            to: "/model-registry",
            from: ["/registry"],
          },
          {
            to: "/llms/deployments",
            from: ["/llms/gateway/index", "/llms/gateway"],
          },
          {
            to: "/llms/deployments/guides",
            from: ["/llms/gateway/guides/index", "/llms/gateway/guide"],
          },
          {
            to: "/llms/deployments/guides/step1-create-deployments",
            from: ["/llms/gateway/guides/step1-create-gateway"],
          },
          {
            to: "/llms/deployments/guides/step2-query-deployments",
            from: ["/llms/gateway/guides/step2-query-gateway"],
          },
        ],
      },
    ],
  ],
};

export default config;
