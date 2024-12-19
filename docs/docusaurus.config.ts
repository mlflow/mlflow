import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";
import { postProcessSidebar } from "./postProcessSidebar";

const config: Config = {
  title: "MLflow",
  tagline: "MLflow Documentation",
  favicon: "img/favicon.ico",

  // Set the production url of your site here
  url: "https://mlflow.org",

  // when building for production, check this environment
  // variable to determine the correct base URL
  baseUrl: process.env.DOCS_BASE_URL ?? "/",

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: "mlflow", // Usually your GitHub org/user name.
  projectName: "mlflow", // Usually your repo name.

  staticDirectories: ["static"],

  // change to throw when migration is done
  onBrokenLinks: "warn",
  onBrokenMarkdownLinks: "warn",

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

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
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: "img/docusaurus-social-card.jpg",
    navbar: {
      logo: {
        alt: "MLflow Logo",
        src: "images/logo.png",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "docsSidebar",
          position: "left",
          label: "Docs",
        },
        {
          href: "https://github.com/facebook/docusaurus",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Community",
          items: [
            {
              label: "Stack Overflow",
              href: "https://stackoverflow.com/questions/tagged/docusaurus",
            },
            {
              label: "Discord",
              href: "https://discordapp.com/invite/docusaurus",
            },
            {
              label: "Twitter",
              href: "https://twitter.com/docusaurus",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} My Project, Inc. Built with Docusaurus.`,
    },
    prism: {
      // There is an array of languages enabled by default.
      // @see https://github.com/FormidableLabs/prism-react-renderer/blob/master/packages/generate-prism-languages/index.ts#L9-L26
      additionalLanguages: ["bash", "diff", "ini", "java", "r", "scala", "sql"],
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
        ],
      },
    ],
  ],
};

export default config;
