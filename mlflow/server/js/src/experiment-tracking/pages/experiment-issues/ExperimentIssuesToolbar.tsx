import { useState } from 'react';
import {
  Button,
  ChevronDownIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSearch,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxTrigger,
  DropdownMenu,
  FilterIcon,
  Popover,
  SortUnsortedIcon,
  Tag,
  type TagColors,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { defineMessage, FormattedMessage, useIntl } from 'react-intl';
import type { MessageDescriptor } from 'react-intl';
import type { IssueSeverity, IssueStatus } from '../../components/run-page/hooks/useSearchIssuesQuery';
import type { IssueCategory } from '../../components/experiment-page/components/traces-v3/IssueDetectionCategories';
import { categoryColor, categoryDescription, STATUS_LABELS, TAG_MARGIN_RESET } from './issueLabels';
import {
  ISSUE_DETECTION_CATEGORIES,
  ISSUE_DETECTION_SEVERITIES,
  ISSUE_DETECTION_TRIAGE_STATUSES,
  BACKEND_ISSUE_SORT_KEYS,
  isIssueSortKey,
  type ExperimentIssuesToolbarControls,
  type IssueSortKey,
} from './useExperimentIssuesListControls';

const COMPONENT_ID = 'mlflow.experiment-issues.toolbar';

const ALL_LABEL = defineMessage({
  defaultMessage: 'All',
  description: 'Experiment issues tab > Filter option that matches every value in a dimension',
});

const SEVERITY_LABELS: Record<IssueSeverity, MessageDescriptor> = {
  not_an_issue: defineMessage({
    defaultMessage: 'Not an issue',
    description: 'Experiment Issues legacy severity filter option',
  }),
  high: defineMessage({
    defaultMessage: 'High',
    description: 'Experiment issues tab > High severity filter option',
  }),
  medium: defineMessage({
    defaultMessage: 'Medium',
    description: 'Experiment issues tab > Medium severity filter option',
  }),
  low: defineMessage({
    defaultMessage: 'Low',
    description: 'Experiment issues tab > Low severity filter option',
  }),
};

// Combobox values need strings; the shared category titles are React elements.
const CATEGORY_LABELS: Record<IssueCategory, MessageDescriptor> = {
  correctness: defineMessage({
    defaultMessage: 'Correctness',
    description: 'Experiment issues tab > Correctness category filter option',
  }),
  latency: defineMessage({
    defaultMessage: 'Latency',
    description: 'Experiment issues tab > Latency category filter option',
  }),
  execution: defineMessage({
    defaultMessage: 'Execution',
    description: 'Experiment issues tab > Execution category filter option',
  }),
  adherence: defineMessage({
    defaultMessage: 'Adherence',
    description: 'Experiment issues tab > Adherence category filter option',
  }),
  relevance: defineMessage({
    defaultMessage: 'Relevance',
    description: 'Experiment issues tab > Relevance category filter option',
  }),
  safety: defineMessage({
    defaultMessage: 'Safety',
    description: 'Experiment issues tab > Safety category filter option',
  }),
};

const SORT_LABELS: Record<IssueSortKey, MessageDescriptor> = {
  severity: defineMessage({
    defaultMessage: 'Severity',
    description: 'Experiment issues tab > Sort by severity option',
  }),
  mostRecent: defineMessage({
    defaultMessage: 'Most recent',
    description: 'Experiment issues tab > Sort by most recently detected option',
  }),
  lastSeen: defineMessage({
    defaultMessage: 'Last seen',
    description: 'Experiment issues tab > Sort by the most recent trace the issue affected option',
  }),
  lastUpdated: defineMessage({
    defaultMessage: 'Last updated',
    description: 'Experiment issues tab > Sort by when the issue last changed option',
  }),
};

const SEVERITY_DIMENSION_LABEL = defineMessage({
  defaultMessage: 'Severity',
  description: 'Experiment issues tab > Severity dimension name in an active filter chip',
});
const CATEGORY_DIMENSION_LABEL = defineMessage({
  defaultMessage: 'Category',
  description: 'Experiment issues tab > Category dimension name in an active filter chip',
});
const STATUS_DIMENSION_LABEL = defineMessage({
  defaultMessage: 'Status',
  description: 'Experiment issues tab > Status dimension name in an active filter chip',
});

/** The filter button and its popover of severity, category and status selections. */
export const ExperimentIssuesFilterControl = ({
  controls,
  align = 'end',
}: {
  controls: ExperimentIssuesToolbarControls;
  /** Which edge of the button the popover lines up with. */
  align?: 'start' | 'end';
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const {
    severities,
    setSeverities,
    categories,
    setCategories,
    statuses,
    setStatuses,
    hasActiveFilters,
    clearFilters,
  } = controls;

  const severityDisplayValue = (value: string) => {
    const severity = ISSUE_DETECTION_SEVERITIES.find((candidate) => candidate === value);
    return severity ? intl.formatMessage(SEVERITY_LABELS[severity]) : value;
  };
  const categoryDisplayValue = (value: string) => {
    const category = ISSUE_DETECTION_CATEGORIES.find((candidate) => candidate === value);
    return category ? intl.formatMessage(CATEGORY_LABELS[category]) : value;
  };
  const statusDisplayValue = (value: string) => {
    const status = ISSUE_DETECTION_TRIAGE_STATUSES.find((candidate) => candidate === value);
    return status ? intl.formatMessage(STATUS_LABELS[status]) : value;
  };

  const toggleSeverity = (value: IssueSeverity) => {
    setSeverities(
      severities.includes(value) ? severities.filter((severity) => severity !== value) : [...severities, value],
    );
  };
  const toggleCategory = (value: string) => {
    setCategories(
      categories.includes(value) ? categories.filter((category) => category !== value) : [...categories, value],
    );
  };
  const toggleStatus = (value: IssueStatus) => {
    setStatuses(statuses.includes(value) ? statuses.filter((status) => status !== value) : [...statuses, value]);
  };

  const [categorySearch, setCategorySearch] = useState('');
  const query = categorySearch.trim();
  const matchesCategory = (value: string) => categoryDisplayValue(value).toLowerCase().includes(query.toLowerCase());
  const customCategories = Array.from(
    new Set([
      ...categories.filter((value) => !ISSUE_DETECTION_CATEGORIES.some((id) => id === value)),
      ...(query &&
      !ISSUE_DETECTION_CATEGORIES.some(
        (id) =>
          id === query.toLowerCase() || intl.formatMessage(CATEGORY_LABELS[id]).toLowerCase() === query.toLowerCase(),
      )
        ? [query]
        : []),
    ]),
  ).filter(matchesCategory);

  const severityRowLabel = intl.formatMessage({
    defaultMessage: 'Severity',
    description: 'Experiment issues tab > Filter popover row label for the severity dimension',
  });
  const categoryRowLabel = intl.formatMessage({
    defaultMessage: 'Category',
    description: 'Experiment issues tab > Filter popover row label for the category dimension',
  });
  const statusRowLabel = intl.formatMessage({
    defaultMessage: 'Status',
    description: 'Experiment issues tab > Filter popover row label for the status dimension',
  });

  return (
    <Popover.Root componentId={`${COMPONENT_ID}.filter`}>
      <Popover.Trigger asChild>
        <Button
          componentId={`${COMPONENT_ID}.filter.trigger`}
          icon={<FilterIcon />}
          aria-label={intl.formatMessage({
            defaultMessage: 'Filter issues',
            description: 'Experiment issues tab > Accessible label for the filter toolbar button',
          })}
        />
      </Popover.Trigger>
      <Popover.Content align={align}>
        <div css={{ display: 'flex', flexDirection: 'column', minWidth: 260 }}>
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              minHeight: theme.general.heightSm,
            }}
          >
            <Typography.Title level={4} withoutMargins>
              <FormattedMessage defaultMessage="Filters" description="Experiment issues tab > Filter popover title" />
            </Typography.Title>
            {hasActiveFilters && (
              <Button
                componentId={`${COMPONENT_ID}.filter.clear`}
                size="small"
                type="tertiary"
                aria-label={intl.formatMessage({
                  defaultMessage: 'Clear filters',
                  description: "Experiment issues tab > Accessible label for the filter popover's clear action",
                })}
                onClick={clearFilters}
              >
                <FormattedMessage
                  defaultMessage="Clear"
                  description="Experiment issues tab > Clear all filters button"
                />
              </Button>
            )}
          </div>
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              gap: theme.spacing.md,
              borderTop: `1px solid ${theme.colors.border}`,
              paddingTop: theme.spacing.md,
            }}
          >
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: theme.spacing.md,
              }}
            >
              <Typography.Text color="secondary">{severityRowLabel}</Typography.Text>
              <DialogCombobox
                id={`${COMPONENT_ID}.filter.severity`}
                componentId={`${COMPONENT_ID}.filter.severity`}
                label={severityRowLabel}
                value={severities}
                multiSelect
              >
                <DialogComboboxTrigger
                  width={180}
                  withInlineLabel={false}
                  aria-label={severityRowLabel}
                  placeholder={intl.formatMessage(ALL_LABEL)}
                  showTagAfterValueCount={1}
                  renderDisplayedValue={severityDisplayValue}
                  onClear={() => setSeverities([])}
                />
                <DialogComboboxContent width={180} style={{ zIndex: theme.options.zIndexBase + 40 }} matchTriggerWidth>
                  <DialogComboboxOptionList>
                    <DialogComboboxOptionListCheckboxItem
                      value="all"
                      checked={severities.length === 0}
                      onChange={() => setSeverities([])}
                    >
                      {intl.formatMessage(ALL_LABEL)}
                    </DialogComboboxOptionListCheckboxItem>
                    {ISSUE_DETECTION_SEVERITIES.map((value) => (
                      <DialogComboboxOptionListCheckboxItem
                        key={value}
                        value={value}
                        checked={severities.includes(value)}
                        onChange={() => toggleSeverity(value)}
                      >
                        {intl.formatMessage(SEVERITY_LABELS[value])}
                      </DialogComboboxOptionListCheckboxItem>
                    ))}
                  </DialogComboboxOptionList>
                </DialogComboboxContent>
              </DialogCombobox>
            </div>
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: theme.spacing.md,
              }}
            >
              <Typography.Text color="secondary">{categoryRowLabel}</Typography.Text>
              <DialogCombobox
                id={`${COMPONENT_ID}.filter.category`}
                componentId={`${COMPONENT_ID}.filter.category`}
                label={categoryRowLabel}
                value={categories}
                multiSelect
              >
                <DialogComboboxTrigger
                  width={180}
                  withInlineLabel={false}
                  aria-label={categoryRowLabel}
                  placeholder={intl.formatMessage(ALL_LABEL)}
                  showTagAfterValueCount={1}
                  renderDisplayedValue={categoryDisplayValue}
                  onClear={() => setCategories([])}
                />
                <DialogComboboxContent width={180} style={{ zIndex: theme.options.zIndexBase + 40 }} matchTriggerWidth>
                  <DialogComboboxOptionList>
                    <DialogComboboxOptionListSearch
                      aria-label={intl.formatMessage({
                        defaultMessage: 'Search categories',
                        description: 'Experiment Issues category filter search',
                      })}
                      controlledValue={categorySearch}
                      setControlledValue={setCategorySearch}
                    >
                      {!query && (
                        <DialogComboboxOptionListCheckboxItem
                          value="all"
                          checked={categories.length === 0}
                          onChange={() => setCategories([])}
                        >
                          {intl.formatMessage(ALL_LABEL)}
                        </DialogComboboxOptionListCheckboxItem>
                      )}
                      {/* Anchored right so it does not cover the adjacent option. `asChild` keeps
                        the item's `role="option"` div, which the list's keyboard nav queries. */}
                      {ISSUE_DETECTION_CATEGORIES.filter(matchesCategory).map((value) => (
                        <Tooltip
                          key={value}
                          componentId={`${COMPONENT_ID}.filter.category.definition_tooltip`}
                          content={categoryDescription(value)}
                          side="right"
                        >
                          <DialogComboboxOptionListCheckboxItem
                            value={value}
                            checked={categories.includes(value)}
                            onChange={() => toggleCategory(value)}
                          >
                            {intl.formatMessage(CATEGORY_LABELS[value])}
                          </DialogComboboxOptionListCheckboxItem>
                        </Tooltip>
                      ))}
                      {customCategories.map((value) => (
                        <DialogComboboxOptionListCheckboxItem
                          key={value}
                          value={value}
                          checked={categories.includes(value)}
                          onChange={() => toggleCategory(value)}
                        >
                          {value}
                        </DialogComboboxOptionListCheckboxItem>
                      ))}
                    </DialogComboboxOptionListSearch>
                  </DialogComboboxOptionList>
                </DialogComboboxContent>
              </DialogCombobox>
            </div>
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: theme.spacing.md,
              }}
            >
              <Typography.Text color="secondary">{statusRowLabel}</Typography.Text>
              <DialogCombobox
                id={`${COMPONENT_ID}.filter.status`}
                componentId={`${COMPONENT_ID}.filter.status`}
                label={statusRowLabel}
                value={statuses}
                multiSelect
              >
                <DialogComboboxTrigger
                  width={180}
                  withInlineLabel={false}
                  aria-label={statusRowLabel}
                  placeholder={intl.formatMessage(ALL_LABEL)}
                  showTagAfterValueCount={1}
                  renderDisplayedValue={statusDisplayValue}
                  onClear={() => setStatuses([])}
                />
                <DialogComboboxContent width={180} style={{ zIndex: theme.options.zIndexBase + 40 }} matchTriggerWidth>
                  <DialogComboboxOptionList>
                    <DialogComboboxOptionListCheckboxItem
                      value="all"
                      checked={statuses.length === 0}
                      onChange={() => setStatuses([])}
                    >
                      {intl.formatMessage(ALL_LABEL)}
                    </DialogComboboxOptionListCheckboxItem>
                    {ISSUE_DETECTION_TRIAGE_STATUSES.map((value) => (
                      <DialogComboboxOptionListCheckboxItem
                        key={value}
                        value={value}
                        checked={statuses.includes(value)}
                        onChange={() => toggleStatus(value)}
                      >
                        {intl.formatMessage(STATUS_LABELS[value])}
                      </DialogComboboxOptionListCheckboxItem>
                    ))}
                  </DialogComboboxOptionList>
                </DialogComboboxContent>
              </DialogCombobox>
            </div>
          </div>
        </div>
      </Popover.Content>
    </Popover.Root>
  );
};

/**
 * The sort button and its menu of the sorts `controls` offers. `labeled` names the current sort
 * ("Sort Severity") instead of showing only an icon.
 */
export const ExperimentIssuesSortControl = ({
  controls,
  labeled = false,
}: {
  controls: ExperimentIssuesToolbarControls;
  labeled?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { sortKey, setSortKey } = controls;
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        {labeled ? (
          <Button
            componentId={`${COMPONENT_ID}.sort.trigger`}
            type="tertiary"
            size="small"
            endIcon={<ChevronDownIcon />}
          >
            <span css={{ display: 'inline-flex', alignItems: 'baseline', gap: theme.spacing.sm }}>
              <Typography.Text color="secondary">
                <FormattedMessage
                  defaultMessage="Sort"
                  description="Experiment issues tab > Label before the current sort on the sort button"
                />
              </Typography.Text>
              {/* Not shown (the gap spaces them), but it makes the button's name "Sort Severity". */}{' '}
              <Typography.Text>{intl.formatMessage(SORT_LABELS[sortKey])}</Typography.Text>
            </span>
          </Button>
        ) : (
          <Button
            componentId={`${COMPONENT_ID}.sort.trigger`}
            icon={<SortUnsortedIcon />}
            aria-label={intl.formatMessage({
              defaultMessage: 'Sort issues',
              description: 'Experiment issues tab > Accessible label for the sort toolbar button',
            })}
          />
        )}
      </DropdownMenu.Trigger>
      <DropdownMenu.Content align="end">
        <DropdownMenu.RadioGroup
          componentId={`${COMPONENT_ID}.sort`}
          value={sortKey}
          onValueChange={(value) => {
            if (isIssueSortKey(value)) {
              setSortKey(value);
            }
          }}
        >
          {BACKEND_ISSUE_SORT_KEYS.map((key) => (
            <DropdownMenu.RadioItem key={key} value={key}>
              <DropdownMenu.ItemIndicator />
              {intl.formatMessage(SORT_LABELS[key])}
            </DropdownMenu.RadioItem>
          ))}
        </DropdownMenu.RadioGroup>
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};

/** A dismissible chip for each active filter value, or nothing when no filter is set. */
export const ExperimentIssuesFilterChips = ({ controls }: { controls: ExperimentIssuesToolbarControls }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { severities, setSeverities, categories, setCategories, statuses, setStatuses, hasActiveFilters } = controls;

  // `color` is only passed for category chips, so a chip echoes the colour its tag carries in the
  // list. Severity and status have no colour language of their own here -- severity is a glyph, not
  // a tag -- so they fall back to the neutral default rather than borrowing a category's hue.
  const chip = ({
    componentId,
    dimension,
    value,
    color,
    onClose,
  }: {
    componentId: string;
    dimension: MessageDescriptor;
    value: MessageDescriptor | string;
    color?: TagColors;
    onClose: () => void;
  }) => (
    <Tag
      key={componentId}
      componentId={componentId}
      color={color}
      css={TAG_MARGIN_RESET}
      closable
      onClose={onClose}
      closeButtonProps={{
        'aria-label': intl.formatMessage(
          {
            defaultMessage: 'Remove {value} from {dimension} filter',
            description: 'Experiment issues tab > Accessible label for removing one selected value from a filter',
          },
          {
            dimension: intl.formatMessage(dimension),
            value: typeof value === 'string' ? value : intl.formatMessage(value),
          },
        ),
      }}
    >
      {intl.formatMessage(
        {
          defaultMessage: '{dimension}: {value}',
          description: 'Experiment issues tab > Active filter chip showing a dimension and its selected value',
        },
        {
          dimension: intl.formatMessage(dimension),
          value: typeof value === 'string' ? value : intl.formatMessage(value),
        },
      )}
    </Tag>
  );

  if (!hasActiveFilters) {
    return null;
  }
  return (
    <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.sm }}>
      {severities.map((severity) =>
        chip({
          componentId: `${COMPONENT_ID}.chip.severity.${severity}`,
          dimension: SEVERITY_DIMENSION_LABEL,
          value: SEVERITY_LABELS[severity],
          onClose: () => setSeverities(severities.filter((value) => value !== severity)),
        }),
      )}
      {categories.map((category) =>
        chip({
          componentId: `${COMPONENT_ID}.chip.category.${category}`,
          dimension: CATEGORY_DIMENSION_LABEL,
          value: ISSUE_DETECTION_CATEGORIES.some((id) => id === category)
            ? CATEGORY_LABELS[category as IssueCategory]
            : category,
          color: categoryColor(category),
          onClose: () => setCategories(categories.filter((value) => value !== category)),
        }),
      )}
      {statuses.map((status) =>
        chip({
          componentId: `${COMPONENT_ID}.chip.status.${status}`,
          dimension: STATUS_DIMENSION_LABEL,
          value: STATUS_LABELS[status],
          onClose: () => setStatuses(statuses.filter((value) => value !== status)),
        }),
      )}
    </div>
  );
};
