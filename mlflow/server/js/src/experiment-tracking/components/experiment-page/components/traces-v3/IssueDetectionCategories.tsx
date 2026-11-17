import React from 'react';
import {
  DollarIcon,
  SearchIcon,
  ShieldIcon,
  LightningIcon,
  CheckCircleIcon,
  ClipboardIcon,
  StarIcon,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

export type IssueCategory = 'correctness' | 'latency' | 'execution' | 'adherence' | 'relevance' | 'safety' | 'cost';

interface IssueCategoryDefinition {
  id: IssueCategory;
  icon: React.ReactNode;
  title: React.ReactNode;
  description: React.ReactNode;
}

export const ISSUE_CATEGORY_DEFINITIONS: IssueCategoryDefinition[] = [
  {
    id: 'correctness',
    icon: <SearchIcon />,
    title: <FormattedMessage defaultMessage="Correctness" description="Issue category title for correctness" />,
    description: (
      <FormattedMessage
        defaultMessage="The output is factually inaccurate or grounded incorrectly."
        description="Issue category description for correctness"
      />
    ),
  },
  {
    id: 'latency',
    icon: <LightningIcon />,
    title: <FormattedMessage defaultMessage="Latency" description="Issue category title for latency" />,
    description: (
      <FormattedMessage
        defaultMessage="The response is unusually slow, overall or in a single span."
        description="Issue category description for latency"
      />
    ),
  },
  {
    id: 'execution',
    icon: <CheckCircleIcon />,
    title: <FormattedMessage defaultMessage="Execution" description="Issue category title for execution" />,
    description: (
      <FormattedMessage
        defaultMessage="The agent errors, a tool call fails, or an action does not complete."
        description="Issue category description for execution"
      />
    ),
  },
  {
    id: 'adherence',
    icon: <ClipboardIcon />,
    title: <FormattedMessage defaultMessage="Adherence" description="Issue category title for adherence" />,
    description: (
      <FormattedMessage
        defaultMessage="The response violates instructions, constraints, or formatting."
        description="Issue category description for adherence"
      />
    ),
  },
  {
    id: 'relevance',
    icon: <StarIcon />,
    title: <FormattedMessage defaultMessage="Relevance" description="Issue category title for relevance" />,
    description: (
      <FormattedMessage
        defaultMessage="The response does not address the user's actual request."
        description="Issue category description for relevance"
      />
    ),
  },
  {
    id: 'safety',
    icon: <ShieldIcon />,
    title: <FormattedMessage defaultMessage="Safety" description="Issue category title for safety" />,
    description: (
      <FormattedMessage
        defaultMessage="The output is harmful, biased, or policy-violating."
        description="Issue category description for safety"
      />
    ),
  },
  {
    id: 'cost',
    icon: <DollarIcon />,
    title: <FormattedMessage defaultMessage="Cost" description="Issue category title for cost" />,
    description: (
      <FormattedMessage
        defaultMessage="The agent's token or tool-call spend is anomalous against its own baseline."
        description="Issue category description for cost"
      />
    ),
  },
];

export const ALL_ISSUE_CATEGORIES: IssueCategory[] = ISSUE_CATEGORY_DEFINITIONS.map((def) => def.id);
