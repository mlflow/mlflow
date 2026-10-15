import { useCallback } from 'react';
import { Input, LegacySelect, SimpleSelect, SimpleSelectOption, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useSearchParams } from '../../../common/utils/RoutingUtils';
import { useDebouncedSearchInput } from '../experiment-evaluation-datasets-v2/hooks/useDebouncedSearchInput';
import { ISSUE_CATEGORY_DEFINITIONS } from '../../components/experiment-page/components/traces-v3/IssueDetectionCategories';
import { STATUS_TAG_CONFIG, SEVERITY_TAG_CONFIG } from '../../components/run-page/IssueCard';
import type { IssueFilters } from './useExperimentIssues';
import type { IssueSeverity, IssueStatus } from '../../components/run-page/hooks/useSearchIssuesQuery';

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
    ? (Object.keys(STATUS_TAG_CONFIG) as IssueStatus[]).filter((value) => params.getAll('issueStatus').includes(value))
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
  return { search, filters, categories, sort, setValues };
};

export const ExperimentIssuesToolbar = ({ controls }: { controls: ReturnType<typeof useIssueListControls> }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { search, filters, categories, sort, setValues } = controls;
  const statusLabel = intl.formatMessage({
    defaultMessage: 'Status',
    description: 'Experiment Issues status filter label',
  });
  const severityLabel = intl.formatMessage({
    defaultMessage: 'Severity',
    description: 'Experiment Issues severity filter label',
  });
  const categoryLabel = intl.formatMessage({
    defaultMessage: 'Category',
    description: 'Experiment Issues category filter label',
  });
  return (
    <div css={{ display: 'flex', gap: theme.spacing.sm, flexWrap: 'wrap', paddingBlock: theme.spacing.md }}>
      <Input
        componentId="mlflow.experiment-issues.search"
        value={search.input}
        onChange={(event) => search.setInput(event.target.value)}
        placeholder={intl.formatMessage({
          defaultMessage: 'Search issues',
          description: 'Experiment Issues title search placeholder',
        })}
        aria-label={intl.formatMessage({
          defaultMessage: 'Search issues',
          description: 'Experiment Issues title search accessible label',
        })}
        css={{ width: 240 }}
        allowClear
      />
      <LegacySelect
        mode="multiple"
        aria-label={statusLabel}
        placeholder={statusLabel}
        value={filters.statuses}
        allowClear
        css={{ minWidth: 160 }}
        onChange={(values: string[]) => setValues('issueStatus', values.length ? values : ['all'])}
      >
        {Object.entries(STATUS_TAG_CONFIG).map(([value, config]) => (
          <LegacySelect.Option key={value} value={value}>
            {intl.formatMessage(config.label)}
          </LegacySelect.Option>
        ))}
      </LegacySelect>
      <LegacySelect
        mode="multiple"
        aria-label={severityLabel}
        placeholder={severityLabel}
        value={filters.severities}
        allowClear
        css={{ minWidth: 160 }}
        onChange={(values: string[]) => setValues('issueSeverity', values)}
      >
        {Object.entries(SEVERITY_TAG_CONFIG).map(([value, config]) => (
          <LegacySelect.Option key={value} value={value}>
            {intl.formatMessage(config.label)}
          </LegacySelect.Option>
        ))}
      </LegacySelect>
      <LegacySelect
        mode="tags"
        aria-label={categoryLabel}
        placeholder={categoryLabel}
        value={categories}
        allowClear
        css={{ minWidth: 160 }}
        onChange={(values: string[]) => setValues('issueCategory', values)}
      >
        {ISSUE_CATEGORY_DEFINITIONS.map(({ id, title }) => (
          <LegacySelect.Option key={id} value={id}>
            {title}
          </LegacySelect.Option>
        ))}
      </LegacySelect>
      <SimpleSelect
        id="experiment-issues-sort"
        componentId="mlflow.experiment-issues.sort"
        value={sort}
        aria-label={intl.formatMessage({
          defaultMessage: 'Sort issues',
          description: 'Experiment Issues sort selector label',
        })}
        onChange={({ target }) => setValues('issueSort', [target.value])}
      >
        <SimpleSelectOption value="severity">
          <FormattedMessage defaultMessage="Status and severity" description="Experiment Issues default sort option" />
        </SimpleSelectOption>
        <SimpleSelectOption value="lastSeen">
          <FormattedMessage defaultMessage="Last seen" description="Experiment Issues last observation sort option" />
        </SimpleSelectOption>
        <SimpleSelectOption value="newest">
          <FormattedMessage defaultMessage="Newest" description="Experiment Issues creation time sort option" />
        </SimpleSelectOption>
        <SimpleSelectOption value="updated">
          <FormattedMessage defaultMessage="Last updated" description="Experiment Issues update time sort option" />
        </SimpleSelectOption>
      </SimpleSelect>
    </div>
  );
};
