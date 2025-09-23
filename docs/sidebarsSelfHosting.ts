import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const sidebarsSelfHosting: SidebarsConfig = {
  selfHostingSidebar: [
    {
      type: 'doc',
      id: 'index',
      label: 'Overview',
      className: 'sidebar-top-level-category',
    },
    {
      type: 'category',
      label: 'Architecture',
      items: [
        {
          type: 'doc',
          id: 'architecture/tracking-server',
          label: 'Tracking Server',
        },
        {
          type: 'doc',
          id: 'architecture/backend-store',
          label: 'Backend Store',
        },
        {
          type: 'doc',
          id: 'architecture/artifact-store',
          label: 'Artifact Store',
        },
      ],
      link: {
        type: 'doc',
        id: 'architecture/index',
      },
    },
    {
      type: 'category',
      label: 'Authentication',
      items: [
        {
          type: 'doc',
          id: 'authentication/basic-http-auth',
          label: 'Username and Password',
        },
        {
          type: 'doc',
          id: 'authentication/sso',
          label: 'SSO (Single Sign-On)',
        },
        {
          type: 'doc',
          id: 'authentication/custom',
          label: 'Custom Authentication',
        }
      ],
    },
    {
      type: 'doc',
      id: 'migration',
      label: 'Upgrade',
    },
    {
      type: 'doc',
      id: 'troubleshooting',
      label: 'Troubleshooting & FAQs',
    },
  ],
};

export default sidebarsSelfHosting;
