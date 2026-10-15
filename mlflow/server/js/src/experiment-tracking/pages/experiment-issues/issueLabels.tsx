import type { TagColors } from '@databricks/design-system';
import { defineMessage, type MessageDescriptor } from 'react-intl';
import {
  ISSUE_CATEGORY_DEFINITIONS,
  type IssueCategory,
} from '../../components/experiment-page/components/traces-v3/IssueDetectionCategories';
import type { IssueStatus } from '../../components/run-page/hooks/useSearchIssuesQuery';

export const STATUS_LABELS: Record<IssueStatus, MessageDescriptor> = {
  pending: defineMessage({ defaultMessage: 'Open', description: 'Experiment Issues open status label' }),
  resolved: defineMessage({ defaultMessage: 'Resolved', description: 'Experiment Issues resolved status label' }),
  rejected: defineMessage({ defaultMessage: 'Not an issue', description: 'Experiment Issues rejected status label' }),
};

const CATEGORY_TAG_COLORS: Record<IssueCategory, TagColors> = {
  correctness: 'indigo',
  latency: 'teal',
  execution: 'coral',
  adherence: 'purple',
  relevance: 'turquoise',
  safety: 'lime',
};

export const getCategoryDefinition = (category: string) =>
  ISSUE_CATEGORY_DEFINITIONS.find(({ id }) => id === category.replace(/^\[|\]$/g, ''));

export const categoryLabel = (category: string) => getCategoryDefinition(category)?.title ?? category;

export const categoryColor = (category: string) => {
  const definition = getCategoryDefinition(category);
  return definition ? CATEGORY_TAG_COLORS[definition.id] : undefined;
};
