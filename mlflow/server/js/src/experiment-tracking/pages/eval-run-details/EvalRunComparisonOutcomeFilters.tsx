import {
  Button,
  CloseIcon,
  FormUI,
  Popover,
  SimpleSelect,
  SimpleSelectOption,
  Tooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { EvalRunComparisonColumnAggregate } from './evalRunComparisonColumnAggregates';
import type { EvalRunComparisonOutcomeFilter as OutcomeFilter } from './useEvalRunComparisonTracesV4Controller';

export interface EvalRunComparisonOutcomeFiltersProps {
  filters: OutcomeFilter[];
  aggregates: Record<string, EvalRunComparisonColumnAggregate>;
  columns: { id: string; label: string }[];
  runs: { runUuid: string; label: string }[];
  onReplace: (oldFilter: OutcomeFilter, nextFilter: OutcomeFilter) => void;
  onRemove: (filter: OutcomeFilter) => void;
  onClear: () => void;
}

export const EvalRunComparisonOutcomeFilters = ({
  filters,
  aggregates,
  columns,
  runs,
  onReplace,
  onRemove,
  onClear,
}: EvalRunComparisonOutcomeFiltersProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  if (!filters.length) return null;

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: theme.spacing.xs,
      }}
    >
      {filters.map((filter) => {
        const column = columns.find(({ id }) => id === filter.columnId)?.label ?? filter.columnId;
        const run = runs.find(({ runUuid }) => runUuid === filter.runUuid)?.label ?? filter.runUuid;
        const outcome =
          filter.outcome === 'regression'
            ? intl.formatMessage({
                defaultMessage: 'regression',
                description: 'Regression outcome filter label',
              })
            : intl.formatMessage({
                defaultMessage: 'improvement',
                description: 'Improvement outcome filter label',
              });
        const label = intl.formatMessage(
          {
            defaultMessage: '{column} {outcome} in {run}',
            description: 'Evaluation comparison outcome filter summary',
          },
          { column, outcome, run },
        );
        const color =
          filter.outcome === 'regression' ? theme.colors.textValidationDanger : theme.colors.textValidationSuccess;
        const key = `outcome-filter-${filter.id}`;
        const runIndex = runs.findIndex(({ runUuid }) => runUuid === filter.runUuid);
        const counts = aggregates[filter.columnId]?.values[runIndex]?.changeCounts;
        const count = (filter.outcome === 'regression' ? counts?.regressed : counts?.improved) ?? 0;
        const tooltip =
          filter.outcome === 'regression'
            ? intl.formatMessage(
                {
                  defaultMessage:
                    '{count, plural, one {# regression} other {# regressions}} for {column} in {run} compared with baseline',
                  description: 'Explanation of a regression comparison filter',
                },
                { count, column, run },
              )
            : intl.formatMessage(
                {
                  defaultMessage:
                    '{count, plural, one {# improvement} other {# improvements}} for {column} in {run} compared with baseline',
                  description: 'Explanation of an improvement comparison filter',
                },
                { count, column, run },
              );
        const update = (draft: Partial<OutcomeFilter>) => onReplace(filter, { ...filter, ...draft });
        return (
          <Tooltip key={filter.id} componentId="mlflow.eval-run-comparison.outcome-filter.tooltip" content={tooltip}>
            <div
              css={{
                display: 'inline-flex',
                alignItems: 'center',
                border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
                borderRadius: theme.borders.borderRadiusSm,
              }}
            >
              <Popover.Root componentId="mlflow.eval-run-comparison.outcome-filter.popover">
                <Popover.Trigger asChild>
                  <Button
                    componentId="mlflow.eval-run-comparison.outcome-filter.edit"
                    size="small"
                    type="tertiary"
                    aria-label={intl.formatMessage(
                      {
                        defaultMessage: 'Edit comparison filter: {label}',
                        description: 'Edit comparison outcome filter accessible label',
                      },
                      { label },
                    )}
                  >
                    <span
                      css={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: theme.spacing.xs,
                      }}
                    >
                      <span aria-hidden="true" css={{ color }}>
                        {filter.outcome === 'regression' ? '↘' : '↗'}
                      </span>
                      <FormattedMessage
                        defaultMessage="{column} {outcome} in {run}"
                        description="Evaluation comparison outcome filter summary"
                        values={{
                          column,
                          outcome: <span css={{ color }}>{outcome}</span>,
                          run,
                        }}
                      />
                    </span>
                  </Button>
                </Popover.Trigger>
                <Popover.Content
                  align="start"
                  css={{
                    padding: theme.spacing.md,
                    width: 520,
                    maxWidth: 'calc(100vw - 32px)',
                  }}
                >
                  <div
                    css={{
                      display: 'flex',
                      gap: theme.spacing.sm,
                      alignItems: 'end',
                    }}
                  >
                    <div css={{ flex: 1, minWidth: 0 }}>
                      <FormUI.Label htmlFor={`${key}-column`}>
                        <FormattedMessage defaultMessage="Column" description="Comparison outcome column field" />
                      </FormUI.Label>
                      <SimpleSelect
                        id={`${key}-column`}
                        componentId="mlflow.eval-run-comparison.outcome-filter.column"
                        value={filter.columnId}
                        onChange={({ target }) => update({ columnId: target.value })}
                        width="100%"
                        contentProps={{
                          style: { zIndex: theme.options.zIndexBase + 100 },
                        }}
                      >
                        {columns.map(({ id, label: name }) => (
                          <SimpleSelectOption key={id} value={id}>
                            {name}
                          </SimpleSelectOption>
                        ))}
                      </SimpleSelect>
                    </div>
                    <div css={{ flex: 1, minWidth: 0 }}>
                      <FormUI.Label htmlFor={`${key}-outcome`}>
                        <FormattedMessage defaultMessage="Outcome" description="Comparison outcome direction field" />
                      </FormUI.Label>
                      <SimpleSelect
                        id={`${key}-outcome`}
                        componentId="mlflow.eval-run-comparison.outcome-filter.outcome"
                        value={filter.outcome}
                        onChange={({ target }) => {
                          if (target.value !== 'regression' && target.value !== 'improvement') return;
                          update({ outcome: target.value });
                        }}
                        width="100%"
                        contentProps={{
                          style: { zIndex: theme.options.zIndexBase + 100 },
                        }}
                      >
                        <SimpleSelectOption value="regression">
                          <FormattedMessage defaultMessage="Regression" description="Regression outcome option" />
                        </SimpleSelectOption>
                        <SimpleSelectOption value="improvement">
                          <FormattedMessage defaultMessage="Improvement" description="Improvement outcome option" />
                        </SimpleSelectOption>
                      </SimpleSelect>
                    </div>
                    <div css={{ flex: 1, minWidth: 0 }}>
                      <FormUI.Label htmlFor={`${key}-run`}>
                        <FormattedMessage defaultMessage="Run" description="Comparison outcome run field" />
                      </FormUI.Label>
                      <SimpleSelect
                        id={`${key}-run`}
                        componentId="mlflow.eval-run-comparison.outcome-filter.run"
                        value={filter.runUuid}
                        onChange={({ target }) => update({ runUuid: target.value })}
                        width="100%"
                        contentProps={{
                          style: { zIndex: theme.options.zIndexBase + 100 },
                        }}
                      >
                        {runs.slice(1).map(({ runUuid, label: name }) => (
                          <SimpleSelectOption key={runUuid} value={runUuid}>
                            {name}
                          </SimpleSelectOption>
                        ))}
                      </SimpleSelect>
                    </div>
                  </div>
                </Popover.Content>
              </Popover.Root>
              <Button
                componentId="mlflow.eval-run-comparison.outcome-filter.remove"
                type="tertiary"
                size="small"
                icon={<CloseIcon />}
                aria-label={intl.formatMessage(
                  {
                    defaultMessage: 'Remove comparison filter: {label}',
                    description: 'Remove comparison outcome filter accessible label',
                  },
                  { label },
                )}
                onClick={() => onRemove(filter)}
              />
            </div>
          </Tooltip>
        );
      })}
      {filters.length > 1 && (
        <Button
          componentId="mlflow.eval-run-comparison.outcome-filter.clear-all"
          type="tertiary"
          size="small"
          aria-label={intl.formatMessage({
            defaultMessage: 'Clear all comparison filters',
            description: 'Clear all comparison outcome filters accessible label',
          })}
          onClick={onClear}
        >
          <FormattedMessage defaultMessage="Clear all" description="Clear all comparison outcome filters button" />
        </Button>
      )}
    </div>
  );
};
