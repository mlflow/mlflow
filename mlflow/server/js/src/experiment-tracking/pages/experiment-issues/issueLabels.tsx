import type { TagColors } from '@databricks/design-system';
import { defineMessage } from 'react-intl';
import type { MessageDescriptor } from 'react-intl';
import { ISSUE_CATEGORY_DEFINITIONS } from '../../components/experiment-page/components/traces-v3/IssueDetectionCategories';
import type { IssueSeverity, IssueStatus } from '../../components/run-page/hooks/useSearchIssuesQuery';
import type { IssueCategory } from '../../components/experiment-page/components/traces-v3/IssueDetectionCategories';

/**
 * Shared by the list row and the detail pane so one severity reads the same in both.
 *
 * Deliberately local to this tab rather than shared with IssueCard: that component serves the
 * per-run detection view, which this tab replaces, so coupling the two would be building
 * against a component scheduled for deletion.
 */
export const SEVERITY_LABELS: Record<IssueSeverity, MessageDescriptor> = {
  not_an_issue: defineMessage({
    defaultMessage: 'Not an issue',
    description: 'Experiment Issues legacy severity label',
  }),
  high: defineMessage({
    defaultMessage: 'High',
    description: 'Experiment issues tab > High severity label',
  }),
  medium: defineMessage({
    defaultMessage: 'Medium',
    description: 'Experiment issues tab > Medium severity label',
  }),
  low: defineMessage({
    defaultMessage: 'Low',
    description: 'Experiment issues tab > Low severity label',
  }),
};

export const STATUS_LABELS: Record<IssueStatus, MessageDescriptor> = {
  pending: defineMessage({
    defaultMessage: 'Open',
    description: 'Experiment issues tab > Open issue status label',
  }),
  resolved: defineMessage({
    defaultMessage: 'Resolved',
    description: 'Experiment issues tab > Resolved issue status label',
  }),
  rejected: defineMessage({
    defaultMessage: 'Not an issue',
    description: 'Experiment issues tab > Rejected issue status label',
  }),
};

/**
 * The same severities as whole phrases, for the list row's glyph, which has no visible text and
 * so needs the accessible name to say what the bars mean.
 *
 * Two message sets rather than composing "{severity} severity" from SEVERITY_LABELS at runtime:
 * assembling a phrase from fragments gives translators no control over word order or adjective
 * agreement, which differ by language. They live next to each other here so a change to one is
 * an obvious prompt to check the other.
 */
export const SEVERITY_GLYPH_LABELS: Record<IssueSeverity, MessageDescriptor> = {
  not_an_issue: defineMessage({
    defaultMessage: 'Not an issue',
    description: 'Experiment Issues legacy severity accessible label',
  }),
  high: defineMessage({
    defaultMessage: 'High severity',
    description: 'Experiment issues tab > Accessible label of the high severity glyph on a list row',
  }),
  medium: defineMessage({
    defaultMessage: 'Medium severity',
    description: 'Experiment issues tab > Accessible label of the medium severity glyph on a list row',
  }),
  low: defineMessage({
    defaultMessage: 'Low severity',
    description: 'Experiment issues tab > Accessible label of the low severity glyph on a list row',
  }),
};

// Tag's wrapper carries an 8px marginRight that flex `gap` adds to rather than replaces, which
// double-spaces any tag row. Reset it so the container's gap is the whole distance.
export const TAG_MARGIN_RESET = { marginRight: 0 };

// A total Record, so adding a category to ISSUE_DETECTION_CATEGORIES without a colour here is a
// type error. Declared in that list's order so the two read side by side.
export const CATEGORY_TAG_COLORS: Record<IssueCategory, TagColors> = {
  correctness: 'indigo',
  latency: 'teal',
  execution: 'coral',
  adherence: 'purple',
  relevance: 'turquoise',
  safety: 'lime',
  cost: 'brown',
};

// Built-in categories have shared labels and definitions; any other value is rendered as a
// user-defined category by IssueCategoryTag.
export const getCategoryDefinition = (category: string) =>
  ISSUE_CATEGORY_DEFINITIONS.find((definition) => definition.id === category.replace(/^\[|\]$/g, ''));

export const categoryLabel = (category: string) => getCategoryDefinition(category)?.title ?? category;

// What the category means, for the chips' hover tooltips. Undefined rather than the id when a
// category has no definition: a chip with no tooltip beats one quoting a raw id back at the user.
export const categoryDescription = (category: string) => getCategoryDefinition(category)?.description;

export const categoryColor = (category: string) => {
  const definition = getCategoryDefinition(category);
  return definition ? CATEGORY_TAG_COLORS[definition.id] : undefined;
};
