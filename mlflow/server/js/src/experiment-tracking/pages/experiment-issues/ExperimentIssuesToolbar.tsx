import { useCallback, type ReactNode } from 'react';
import {
  Button,
  ChevronDownIcon,
  DropdownMenu,
  FilterIcon,
  Input,
  LegacySelect,
  Popover,
  SearchIcon,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { defineMessage, FormattedMessage, useIntl } from 'react-intl';
import { useSearchParams } from '../../../common/utils/RoutingUtils';
import { useDebouncedSearchInput } from '../experiment-evaluation-datasets-v2/hooks/useDebouncedSearchInput';
import { ISSUE_CATEGORY_DEFINITIONS } from '../../components/experiment-page/components/traces-v3/IssueDetectionCategories';
import { SEVERITY_TAG_CONFIG } from '../../components/run-page/IssueCard';
import type { IssueFilters } from './useExperimentIssues';
import type { IssueSeverity, IssueStatus } from '../../components/run-page/hooks/useSearchIssuesQuery';
import { STATUS_LABELS, categoryColor, categoryLabel } from './issueLabels';

const SORTS = {
  severity: ['status ASC', 'severity DESC', 'last_seen_timestamp DESC'],
  lastSeen: ['last_seen_timestamp DESC'],
  newest: ['created_timestamp DESC'],
  updated: ['last_updated_timestamp DESC'],
};

type SortKey = keyof typeof SORTS;

export const useIssueListControls = () => {
  const [params, setParams] = useSearchParams();
  const setValues = useCallback(
    (key: string, values: string[]) =>
      setParams(
        (current) => {
          current.delete(key);
          values.forEach((value) => current.append(key, value));
          return current;
        },
        { replace: true },
      ),
    [setParams],
  );
  const setName = useCallback((value: string) => setValues('issueSearch', value ? [value] : []), [setValues]);
  const search = useDebouncedSearchInput({
    committedValue: params.get('issueSearch') ?? '',
    onCommit: setName,
    debounceMs: 300,
  });
  const statuses = params.has('issueStatus')
    ? (Object.keys(STATUS_LABELS) as IssueStatus[]).filter((value) => params.getAll('issueStatus').includes(value))
    : ['pending' as const];
  const severities = (Object.keys(SEVERITY_TAG_CONFIG) as IssueSeverity[]).filter((value) =>
    params.getAll('issueSeverity').includes(value),
  );
  const categories = params.getAll('issueCategory');
  const sortParam = params.get('issueSort') ?? '';
  const sort: SortKey = Object.keys(SORTS).includes(sortParam) ? (sortParam as SortKey) : 'severity';
  const filters: IssueFilters = {
    name: params.get('issueSearch') ?? '',
    statuses,
    severities,
    categories: categories.flatMap((category) =>
      ISSUE_CATEGORY_DEFINITIONS.some((definition) => definition.id === category)
        ? [category, `[${category}]`]
        : [category],
    ),
    orderBy: SORTS[sort],
  };
  const clearFilters = () =>
    setParams(
      (current) => {
        current.delete('issueStatus');
        current.append('issueStatus', 'all');
        current.delete('issueSeverity');
        current.delete('issueCategory');
        return current;
      },
      { replace: true },
    );
  return { search, filters, categories, sort, setValues, clearFilters };
};

type IssueListControls = ReturnType<typeof useIssueListControls>;

const SORT_LABELS = {
  severity: defineMessage({ defaultMessage: 'Severity', description: 'Experiment Issues default sort option' }),
  lastSeen: defineMessage({
    defaultMessage: 'Last seen',
    description: 'Experiment Issues last observation sort option',
  }),
  newest: defineMessage({ defaultMessage: 'Newest', description: 'Experiment Issues creation time sort option' }),
  updated: defineMessage({ defaultMessage: 'Last updated', description: 'Experiment Issues update time sort option' }),
};

export const ExperimentIssuesToolbar = ({
  controls,
  children,
}: {
  controls: IssueListControls;
  children?: ReactNode;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { search, filters, categories, setValues, clearFilters } = controls;
  const hasFilters = Boolean(filters.statuses.length || filters.severities.length || categories.length);
  const statusLabel = intl.formatMessage({
    defaultMessage: 'Status',
    description: 'Experiment Issues status filter label',
  });
  const severityLabel = intl.formatMessage({
    defaultMessage: 'Severity',
    description: 'Experiment Issues severity filter label',
  });
  const categoryFilterLabel = intl.formatMessage({
    defaultMessage: 'Category',
    description: 'Experiment Issues category filter label',
  });
  const allLabel = intl.formatMessage({
    defaultMessage: 'All',
    description: 'Experiment Issues unfiltered dimension label',
  });
  const selectRow = (label: string, select: ReactNode) => (
    <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: theme.spacing.md }}>
      <Typography.Text color="secondary">{label}</Typography.Text>
      {select}
    </div>
  );
  return (
    <div
      role="toolbar"
      aria-label={intl.formatMessage({
        defaultMessage: 'Search and detect issues',
        description: 'Experiment Issues action toolbar label',
      })}
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
        paddingBottom: theme.spacing.md,
        borderBottom: `1px solid ${theme.colors.border}`,
        flexShrink: 0,
      }}
    >
      <div css={{ flex: '0 1 400px', minWidth: 0 }}>
        <Input
          componentId="mlflow.experiment-issues.search"
          prefix={<SearchIcon />}
          value={search.input}
          onChange={(event) => search.setInput(event.target.value)}
          onClear={search.clear}
          onPressEnter={search.flush}
          placeholder={intl.formatMessage({
            defaultMessage: 'Search issues by title',
            description: 'Experiment Issues title search placeholder',
          })}
          aria-label={intl.formatMessage({
            defaultMessage: 'Search issues by title',
            description: 'Experiment Issues title search accessible label',
          })}
          allowClear
        />
      </div>
      <Popover.Root componentId="mlflow.experiment-issues.filter">
        <Popover.Trigger asChild>
          <Button
            componentId="mlflow.experiment-issues.filter-trigger"
            type="tertiary"
            icon={<FilterIcon />}
            aria-label={intl.formatMessage({
              defaultMessage: 'Filter issues',
              description: 'Experiment Issues filter button label',
            })}
          />
        </Popover.Trigger>
        <Popover.Content align="start">
          <div css={{ display: 'flex', flexDirection: 'column', minWidth: 280, gap: theme.spacing.md }}>
            <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Typography.Title level={4} withoutMargins>
                <FormattedMessage defaultMessage="Filters" description="Experiment Issues filter popover title" />
              </Typography.Title>
              {hasFilters && (
                <Button
                  componentId="mlflow.experiment-issues.clear-filters"
                  type="tertiary"
                  size="small"
                  onClick={clearFilters}
                >
                  <FormattedMessage
                    defaultMessage="Clear filters"
                    description="Experiment Issues clear filters action"
                  />
                </Button>
              )}
            </div>
            {selectRow(
              severityLabel,
              <LegacySelect
                mode="multiple"
                aria-label={severityLabel}
                placeholder={allLabel}
                value={filters.severities}
                allowClear
                css={{ width: 180 }}
                onChange={(values: string[]) => setValues('issueSeverity', values)}
              >
                {Object.entries(SEVERITY_TAG_CONFIG).map(([value, config]) => (
                  <LegacySelect.Option key={value} value={value}>
                    {intl.formatMessage(config.label)}
                  </LegacySelect.Option>
                ))}
              </LegacySelect>,
            )}
            {selectRow(
              categoryFilterLabel,
              <LegacySelect
                mode="tags"
                aria-label={categoryFilterLabel}
                placeholder={allLabel}
                value={categories}
                allowClear
                css={{ width: 180 }}
                onChange={(values: string[]) => setValues('issueCategory', values)}
              >
                {ISSUE_CATEGORY_DEFINITIONS.map(({ id, title }) => (
                  <LegacySelect.Option key={id} value={id}>
                    {title}
                  </LegacySelect.Option>
                ))}
              </LegacySelect>,
            )}
            {selectRow(
              statusLabel,
              <LegacySelect
                mode="multiple"
                aria-label={statusLabel}
                placeholder={allLabel}
                value={filters.statuses}
                allowClear
                css={{ width: 180 }}
                onChange={(values: string[]) => setValues('issueStatus', values.length ? values : ['all'])}
              >
                {Object.entries(STATUS_LABELS).map(([value, label]) => (
                  <LegacySelect.Option key={value} value={value}>
                    {intl.formatMessage(label)}
                  </LegacySelect.Option>
                ))}
              </LegacySelect>,
            )}
          </div>
        </Popover.Content>
      </Popover.Root>
      <div css={{ flex: 1 }} />
      {children}
    </div>
  );
};

export const ExperimentIssuesListHeader = ({ controls, total }: { controls: IssueListControls; total?: number }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { filters, categories, sort, setValues } = controls;
  const chips = [
    ...filters.statuses.map((value) => ({
      key: 'issueStatus',
      value,
      label: intl.formatMessage(STATUS_LABELS[value]),
      dimension: intl.formatMessage({ defaultMessage: 'Status', description: 'Experiment Issues status filter label' }),
      values: filters.statuses,
    })),
    ...filters.severities.map((value) => ({
      key: 'issueSeverity',
      value,
      label: intl.formatMessage(SEVERITY_TAG_CONFIG[value].label),
      dimension: intl.formatMessage({
        defaultMessage: 'Severity',
        description: 'Experiment Issues severity filter label',
      }),
      values: filters.severities,
    })),
    ...categories.map((value) => ({
      key: 'issueCategory',
      value,
      label: categoryLabel(value),
      dimension: intl.formatMessage({
        defaultMessage: 'Category',
        description: 'Experiment Issues category filter label',
      }),
      values: categories,
    })),
  ];
  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
        gap: theme.spacing.sm,
        padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Typography.Text bold>
          <FormattedMessage defaultMessage="Issues" description="Experiment Issues page heading" />
        </Typography.Text>
        {total !== undefined && (
          <Typography.Text color="secondary" css={{ whiteSpace: 'nowrap' }}>
            <FormattedMessage
              defaultMessage="{count, plural, one {# result} other {# results}}"
              description="Experiment Issues matching result count"
              values={{ count: total }}
            />
          </Typography.Text>
        )}
        <div css={{ marginLeft: 'auto' }}>
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <Button
                componentId="mlflow.experiment-issues.sort-trigger"
                type="tertiary"
                size="small"
                endIcon={<ChevronDownIcon />}
              >
                <span css={{ display: 'inline-flex', alignItems: 'baseline', gap: theme.spacing.sm }}>
                  <Typography.Text color="secondary">
                    <FormattedMessage defaultMessage="Sort" description="Experiment Issues sort prefix" />
                  </Typography.Text>{' '}
                  <Typography.Text>{intl.formatMessage(SORT_LABELS[sort])}</Typography.Text>
                </span>
              </Button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Content align="end">
              <DropdownMenu.RadioGroup
                componentId="mlflow.experiment-issues.sort"
                value={sort}
                onValueChange={(value) => {
                  if (Object.keys(SORTS).includes(value)) setValues('issueSort', [value]);
                }}
              >
                {Object.entries(SORT_LABELS).map(([key, label]) => (
                  <DropdownMenu.RadioItem key={key} value={key}>
                    <DropdownMenu.ItemIndicator />
                    {intl.formatMessage(label)}
                  </DropdownMenu.RadioItem>
                ))}
              </DropdownMenu.RadioGroup>
            </DropdownMenu.Content>
          </DropdownMenu.Root>
        </div>
      </div>
      {chips.length > 0 && (
        <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.sm }}>
          {chips.map(({ key, value, label, dimension, values }) => (
            <Tag
              key={`${key}:${value}`}
              componentId="mlflow.experiment-issues.filter-chip"
              css={{ marginRight: 0 }}
              closable
              color={key === 'issueCategory' ? categoryColor(value) : undefined}
              onClose={() => {
                const remaining = values.filter((candidate) => candidate !== value);
                setValues(key, key === 'issueStatus' && !remaining.length ? ['all'] : remaining);
              }}
              closeButtonProps={{
                'aria-label': intl.formatMessage(
                  {
                    defaultMessage: 'Remove {value} from {dimension} filter',
                    description: 'Experiment Issues remove filter chip action',
                  },
                  { value: typeof label === 'string' ? label : value, dimension },
                ),
              }}
            >
              <FormattedMessage
                defaultMessage="{dimension}: {value}"
                description="Experiment Issues active filter chip"
                values={{ dimension, value: label }}
              />
            </Tag>
          ))}
        </div>
      )}
    </div>
  );
};
