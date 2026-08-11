import {
  Button,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  Typography,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { EvalRunsDatePreset, EvalRunsTypeFilter } from './hooks/useEvalRunsFilters';
import type { EvalRunsFiltersState } from './hooks/useEvalRunsFilters';

const DATE_PRESET_ORDER: EvalRunsDatePreset[] = [
  EvalRunsDatePreset.LAST_HOUR,
  EvalRunsDatePreset.LAST_24_HOURS,
  EvalRunsDatePreset.LAST_7_DAYS,
  EvalRunsDatePreset.LAST_30_DAYS,
  EvalRunsDatePreset.LAST_YEAR,
  EvalRunsDatePreset.ALL,
];

const TYPE_FILTER_ORDER: EvalRunsTypeFilter[] = [
  EvalRunsTypeFilter.ALL,
  EvalRunsTypeFilter.EVAL_ONLY,
  EvalRunsTypeFilter.ISSUE_DETECTION,
  EvalRunsTypeFilter.TEST,
];

/**
 * The filter controls, rendered inline in the table toolbar rather than as their
 * own row. They sit alongside Columns and Group by because all three do the same
 * job — deciding what the table shows — and a separate filter row cost a band of
 * vertical space on a page whose problem is that the table is hard to see.
 *
 * There is deliberately no "Run by" filter: `getUser()` returns a placeholder in
 * OSS and every run carries the same `user_id`, so the control could only ever
 * offer one option and would never partition the list.
 */
export const EvalRunsFilterControls = ({
  filters,
  isAnyFilterActive,
  setDatePreset,
  setTypeFilter,
  clearAll,
}: {
  filters: EvalRunsFiltersState;
  isAnyFilterActive: boolean;
  setDatePreset: (preset: EvalRunsDatePreset) => void;
  setTypeFilter: (filter: EvalRunsTypeFilter) => void;
  clearAll: () => void;
}) => {
  const intl = useIntl();

  const datePresetLabels: Record<EvalRunsDatePreset, string> = {
    [EvalRunsDatePreset.LAST_HOUR]: intl.formatMessage({
      defaultMessage: 'Last hour',
      description: 'Date range preset in the evaluation runs filter row',
    }),
    [EvalRunsDatePreset.LAST_24_HOURS]: intl.formatMessage({
      defaultMessage: 'Last 24 hours',
      description: 'Date range preset in the evaluation runs filter row',
    }),
    [EvalRunsDatePreset.LAST_7_DAYS]: intl.formatMessage({
      defaultMessage: 'Last 7 days',
      description: 'Date range preset in the evaluation runs filter row',
    }),
    [EvalRunsDatePreset.LAST_30_DAYS]: intl.formatMessage({
      defaultMessage: 'Last 30 days',
      description: 'Date range preset in the evaluation runs filter row',
    }),
    [EvalRunsDatePreset.LAST_YEAR]: intl.formatMessage({
      defaultMessage: 'Last year',
      description: 'Date range preset in the evaluation runs filter row',
    }),
    [EvalRunsDatePreset.ALL]: intl.formatMessage({
      defaultMessage: 'All time',
      description: 'Date range preset in the evaluation runs filter row',
    }),
  };

  const typeFilterLabels: Record<EvalRunsTypeFilter, string> = {
    [EvalRunsTypeFilter.ALL]: intl.formatMessage({
      defaultMessage: 'All types',
      description: 'Run type filter option in the evaluation runs filter row',
    }),
    [EvalRunsTypeFilter.EVAL_ONLY]: intl.formatMessage({
      defaultMessage: 'Evaluations only',
      description: 'Run type filter option in the evaluation runs filter row',
    }),
    [EvalRunsTypeFilter.ISSUE_DETECTION]: intl.formatMessage({
      defaultMessage: 'Issue detection',
      description: 'Run type filter option in the evaluation runs filter row',
    }),
    [EvalRunsTypeFilter.TEST]: intl.formatMessage({
      defaultMessage: 'Test runs',
      description: 'Run type filter option in the evaluation runs filter row',
    }),
  };

  return (
    <>
      <DialogCombobox
        componentId="mlflow.eval-runs.filter-date-preset"
        label={<FormattedMessage defaultMessage="Date" description="Label for the evaluation runs date filter" />}
        value={[datePresetLabels[filters.datePreset]]}
      >
        <DialogComboboxTrigger allowClear={false} showTagAfterValueCount={1} css={{ flexShrink: 0 }} />
        <DialogComboboxContent>
          <DialogComboboxOptionList>
            {DATE_PRESET_ORDER.map((preset) => (
              <DialogComboboxOptionListSelectItem
                key={preset}
                value={datePresetLabels[preset]}
                checked={filters.datePreset === preset}
                onChange={() => setDatePreset(preset)}
              >
                {datePresetLabels[preset]}
              </DialogComboboxOptionListSelectItem>
            ))}
          </DialogComboboxOptionList>
        </DialogComboboxContent>
      </DialogCombobox>

      <DialogCombobox
        componentId="mlflow.eval-runs.filter-run-type"
        label={<FormattedMessage defaultMessage="Type" description="Label for the evaluation runs type filter" />}
        value={[typeFilterLabels[filters.typeFilter]]}
      >
        <DialogComboboxTrigger allowClear={false} showTagAfterValueCount={1} css={{ flexShrink: 0 }} />
        <DialogComboboxContent>
          <DialogComboboxOptionList>
            {TYPE_FILTER_ORDER.map((option) => (
              <DialogComboboxOptionListSelectItem
                key={option}
                value={typeFilterLabels[option]}
                checked={filters.typeFilter === option}
                onChange={() => setTypeFilter(option)}
              >
                {typeFilterLabels[option]}
              </DialogComboboxOptionListSelectItem>
            ))}
          </DialogComboboxOptionList>
        </DialogComboboxContent>
      </DialogCombobox>

      {isAnyFilterActive && (
        <Button componentId="mlflow.eval-runs.filter-clear-all" type="tertiary" onClick={clearAll}>
          <FormattedMessage
            defaultMessage="Clear all"
            description="Button that resets every filter on the evaluation runs page"
          />
        </Button>
      )}
    </>
  );
};

/**
 * The run count, split from the controls so it can be right-aligned at the end of
 * the toolbar row instead of sitting between the filters and the Actions menu.
 *
 * Both counts are of the runs fetched so far, not the experiment totals:
 * search_runs pages at 50 with infinite scroll and returns no total count.
 * Claiming an exact total would be wrong on any experiment past one page, so the
 * copy is hedged while more pages remain.
 */
export const EvalRunsFilterCount = ({
  isAnyFilterActive,
  visibleCount,
  totalCount,
  hasMoreRuns,
}: {
  isAnyFilterActive: boolean;
  visibleCount: number;
  totalCount: number;
  hasMoreRuns: boolean;
}) => {
  return (
    <>
      <Typography.Hint css={{ marginLeft: 'auto', whiteSpace: 'nowrap', alignSelf: 'center' }}>
        {isAnyFilterActive ? (
          hasMoreRuns ? (
            <FormattedMessage
              defaultMessage="Showing {visibleCount} of {totalCount}+ runs loaded"
              description="Count of matching evaluation runs when more runs are still unloaded"
              values={{ visibleCount, totalCount }}
            />
          ) : (
            <FormattedMessage
              defaultMessage="Showing {visibleCount} of {totalCount} runs"
              description="Count of evaluation runs currently shown versus the unfiltered total"
              values={{ visibleCount, totalCount }}
            />
          )
        ) : hasMoreRuns ? (
          <FormattedMessage
            defaultMessage="{totalCount}+ runs loaded"
            description="Count of loaded evaluation runs when more pages remain"
            values={{ totalCount }}
          />
        ) : (
          <FormattedMessage
            defaultMessage="{totalCount, plural, one {# run} other {# runs}}"
            description="Total count of evaluation runs when no filter is applied"
            values={{ totalCount }}
          />
        )}
      </Typography.Hint>
    </>
  );
};
