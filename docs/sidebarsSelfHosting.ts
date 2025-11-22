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
          id: 'architecture/overview',
          label: 'Overview',
        },
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
    },
    {
      type: 'category',
      label: 'Workspaces (Multi-Tenancy)',
      items: [
        {
          type: 'doc',
          id: 'workspaces/index',
          label: 'Overview',
        },
        {
          type: 'doc',
          id: 'workspaces/getting-started',
          label: 'Getting Started',
        },
        {
          type: 'doc',
          id: 'workspaces/configuration',
          label: 'Configuration',
        },
        {
          type: 'doc',
          id: 'workspaces/workspace-providers',
          label: 'Workspace Providers',
        },
        {
          type: 'doc',
          id: 'workspaces/permissions',
          label: 'Permissions',
        },
      ],
    },
    {
      type: 'category',
      label: 'Security',
      items: [
        {
          type: 'doc',
          id: 'security/network',
          label: ' Network Protection',
        },
        {
          type: 'doc',
          id: 'security/basic-http-auth',
          label: 'Username and Password',
        },
        {
          type: 'doc',
          id: 'security/sso',
          label: 'SSO (Single Sign-On)',
        },
        {
          type: 'doc',
          id: 'security/custom',
          label: 'Custom Authentication',
        },
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
