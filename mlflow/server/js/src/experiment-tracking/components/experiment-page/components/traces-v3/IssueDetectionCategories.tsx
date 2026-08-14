import React from 'react';
import {
  SearchIcon,
  ShieldIcon,
  LightningIcon,
  CheckCircleIcon,
  ClipboardIcon,
  StarIcon,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

export type IssueCategory = 'correctness' | 'latency' | 'execution' | 'adherence' | 'relevance' | 'safety';

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
        defaultMessage="Output is factually accurate and grounded in provided data"
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
        defaultMessage="Agent responds within acceptable time bounds"
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
        defaultMessage="Agent successfully completes actions (tool calls, API steps)"
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
        defaultMessage="Response follows instructions, constraints, policies, and formatting"
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
        defaultMessage="Output is useful, directly addresses the user's request, and leaves the user satisfied with the interaction"
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
        defaultMessage="Response avoids harmful, sensitive, or inappropriate content"
        description="Issue category description for safety"
      />
    ),
  },
];

export const ALL_ISSUE_CATEGORIES: IssueCategory[] = ISSUE_CATEGORY_DEFINITIONS.map((def) => def.id);
