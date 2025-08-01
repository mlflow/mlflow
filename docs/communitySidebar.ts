import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const communitySidebar: SidebarsConfig = {
  communitySidebar: [
    {
      type: 'doc',
      id: 'usage-tracking',
      label: 'Usage Tracking',
    },
    {
      type: 'category',
      label: 'Community Resources',
      collapsed: false,
      items: [
        {
          type: 'link',
          label: 'GitHub Repository',
          href: 'https://github.com/mlflow/mlflow',
        },
        {
          type: 'link',
          label: 'Slack Community',
          href: 'https://mlflow.org/slack',
        },
        {
          type: 'link',
          label: 'Twitter/X',
          href: 'https://twitter.com/mlflow',
        },
        {
          type: 'link',
          label: 'LinkedIn',
          href: 'https://www.linkedin.com/company/mlflow-org',
        },
        {
          type: 'link',
          label: 'Stack Overflow',
          href: 'https://stackoverflow.com/questions/tagged/mlflow',
        },
      ],
    },
    {
      type: 'category',
      label: 'Get Involved',
      collapsed: false,
      items: [
        {
          type: 'link',
          label: 'Contributing Guide',
          href: 'https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.md',
        },
        {
          type: 'link',
          label: 'MLflow Blog',
          href: 'https://mlflow.org/blog/index.html',
        },
        {
          type: 'link',
          label: 'MLflow Ambassador Program',
          href: 'https://mlflow.org/ambassadors',
        },
        {
          type: 'link',
          label: 'Report Issues',
          href: 'https://github.com/mlflow/mlflow/issues/new/choose',
        },
      ],
    },
  ],
};

export default communitySidebar;