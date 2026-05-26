import { useEffect, useMemo, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import {
  Button,
  ChevronDownIcon,
  CloseIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSearch,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  FilterIcon,
  FormUI,
  Input,
  PlusIcon,
  Popover,
  SimpleSelect,
  SimpleSelectOption,
  XCircleFillIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { MetricFilter } from './MetricsFilter.utils';
import { isCompleteFilter, type MetricFilterColumn, type MetricFilterColumnOption } from './MetricsFilter.utils';

// UI-only row state. The `id` is a stable per-row identifier used as the
// React key so that adding/removing rows does not cause adjacent rows to
// inherit one another's component state (focus, dropdown open state, etc.).
// It is stripped before filters are forwarded to the parent via setFilters.
interface FilterRowState extends MetricFilter {
  id: string;
}

const toRowState = (filter: MetricFilter): FilterRowState => ({ ...filter, id: uuidv4() });

interface MetricsFilterProps {
  filters: MetricFilter[];
  setFilters: (filters: MetricFilter[]) => void;
  columnOptions: MetricFilterColumnOption[];
}

export const MetricsFilter = ({ filters, setFilters, columnOptions }: MetricsFilterProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const hasActiveFilters = filters.length > 0;

  return (
    <Popover.Root componentId="mlflow.usage.metrics_filter">
      <Popover.Trigger asChild>
        <Button
          endIcon={<ChevronDownIcon />}
          componentId="mlflow.usage.metrics_filter.button"
          css={{
            border: hasActiveFilters ? `1px solid ${theme.colors.actionDefaultBorderFocus} !important` : '',
            backgroundColor: hasActiveFilters ? `${theme.colors.actionDefaultBackgroundHover} !important` : '',
          }}
        >
          <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
            <FilterIcon />
            {intl.formatMessage(
              {
                defaultMessage: 'Filters{numFilters}',
                description: 'Usage overview > metrics filter dropdown button',
              },
              { numFilters: hasActiveFilters ? ` (${filters.length})` : '' },
            )}
            {hasActiveFilters && (
              <span
                role="button"
                tabIndex={0}
                aria-label={intl.formatMessage({
                  defaultMessage: 'Clear filters',
                  description: 'Usage overview > clear metrics filters button',
                })}
                css={{
                  fontSize: theme.typography.fontSizeSm,
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  color: theme.colors.grey400,
                  '&:hover': { color: theme.colors.grey600 },
                  '&:focus-visible': {
                    outline: `2px solid ${theme.colors.actionDefaultBorderFocus}`,
                    outlineOffset: 2,
                    borderRadius: theme.general.borderRadiusBase,
                  },
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  setFilters([]);
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    e.stopPropagation();
                    setFilters([]);
                  }
                }}
              >
                <XCircleFillIcon css={{ fontSize: theme.typography.fontSizeSm }} />
              </span>
            )}
          </div>
        </Button>
      </Popover.Trigger>
      <Popover.Content align="start" css={{ padding: theme.spacing.md, minWidth: 480 }}>
        <FilterForm filters={filters} setFilters={setFilters} columnOptions={columnOptions} />
      </Popover.Content>
    </Popover.Root>
  );
};

const FilterForm = ({ filters, setFilters, columnOptions }: MetricsFilterProps) => {
  const { theme } = useDesignSystemTheme();
  const emptyFilter = useMemo<MetricFilter>(
    () => ({ column: columnOptions[0]?.value ?? ('' as MetricFilterColumn), value: '' }),
    [columnOptions],
  );
  const [localFilters, setLocalFilters] = useState<FilterRowState[]>(() =>
    (filters.length > 0 ? filters : [emptyFilter]).map(toRowState),
  );

  // Keep the draft in sync with the applied filters so that external changes
  // (e.g. clicking the clear icon in the trigger while the popover is open)
  // are reflected in the form instead of being clobbered on the next Apply.
  useEffect(() => {
    setLocalFilters((filters.length > 0 ? filters : [emptyFilter]).map(toRowState));
  }, [filters, emptyFilter]);

  const updateAt = (id: string, next: MetricFilter) => {
    setLocalFilters((prev) => prev.map((row) => (row.id === id ? { ...row, ...next } : row)));
  };

  const removeAt = (id: string) => {
    setLocalFilters((prev) => {
      const next = prev.filter((row) => row.id !== id);
      return next.length === 0 ? [toRowState(emptyFilter)] : next;
    });
  };

  const addRow = () => {
    setLocalFilters((prev) => [...prev, toRowState(emptyFilter)]);
  };

  const apply = () => {
    setFilters(
      localFilters.filter(isCompleteFilter).map((row): MetricFilter => ({ column: row.column, value: row.value })),
    );
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        {localFilters.map((row) => (
          <FilterRow
            key={row.id}
            rowId={row.id}
            filter={row}
            columnOptions={columnOptions}
            onChange={(next) => updateAt(row.id, next)}
            onDelete={() => removeAt(row.id)}
          />
        ))}
      </div>
      <div>
        <Button componentId="mlflow.usage.metrics_filter.add_row" icon={<PlusIcon />} onClick={addRow}>
          <FormattedMessage defaultMessage="Add filter" description="Usage overview > add filter row button" />
        </Button>
      </div>
      <div css={{ display: 'flex', justifyContent: 'flex-end' }}>
        <Button componentId="mlflow.usage.metrics_filter.apply" type="primary" onClick={apply}>
          <FormattedMessage defaultMessage="Apply filters" description="Usage overview > apply filters button" />
        </Button>
      </div>
    </div>
  );
};

interface FilterRowProps {
  rowId: string;
  filter: MetricFilter;
  columnOptions: MetricFilterColumnOption[];
  onChange: (next: MetricFilter) => void;
  onDelete: () => void;
}

const FilterRow = ({ rowId, filter, columnOptions, onChange, onDelete }: FilterRowProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const selectedOption = columnOptions.find((o) => o.value === filter.column);

  return (
    <div css={{ display: 'flex', flexDirection: 'row', gap: theme.spacing.sm, alignItems: 'flex-end' }}>
      {/* Field (Column) */}
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor={`usage-metrics-filter-column-${rowId}`}>
          <FormattedMessage defaultMessage="Field" description="Usage overview > filter row > field column label" />
        </FormUI.Label>
        <DialogCombobox
          componentId="mlflow.usage.metrics_filter.column"
          id={`usage-metrics-filter-column-${rowId}`}
          value={filter.column ? [filter.column] : []}
        >
          <DialogComboboxTrigger
            withInlineLabel={false}
            placeholder={intl.formatMessage({
              defaultMessage: 'Select field',
              description: 'Usage overview > filter row > field column placeholder',
            })}
            renderDisplayedValue={() => selectedOption?.label ?? ''}
            width={160}
            allowClear={false}
          />
          <DialogComboboxContent width={160} style={{ zIndex: theme.options.zIndexBase + 100 }}>
            <DialogComboboxOptionList>
              {columnOptions.map((option) => (
                <DialogComboboxOptionListSelectItem
                  key={option.value}
                  value={option.value}
                  checked={option.value === filter.column}
                  onChange={(value: string) =>
                    onChange({
                      ...filter,
                      column: value as MetricFilterColumn,
                      value: value === filter.column ? filter.value : '',
                    })
                  }
                >
                  {option.label}
                </DialogComboboxOptionListSelectItem>
              ))}
            </DialogComboboxOptionList>
          </DialogComboboxContent>
        </DialogCombobox>
      </div>

      {/* Operator (locked to "=") */}
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor={`usage-metrics-filter-operator-${rowId}`}>
          <FormattedMessage
            defaultMessage="Operator"
            description="Usage overview > filter row > operator column label"
          />
        </FormUI.Label>
        <SimpleSelect
          aria-label={intl.formatMessage({
            defaultMessage: 'Operator',
            description: 'Usage overview > filter row > operator dropdown aria label',
          })}
          componentId="mlflow.usage.metrics_filter.operator"
          id={`usage-metrics-filter-operator-${rowId}`}
          value="="
          disabled
          contentProps={{ style: { zIndex: theme.options.zIndexBase + 100 } }}
          css={{ width: 80 }}
        >
          <SimpleSelectOption value="=">=</SimpleSelectOption>
        </SimpleSelect>
      </div>

      {/* Value */}
      <div css={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
        <FormUI.Label htmlFor={`usage-metrics-filter-value-${rowId}`}>
          <FormattedMessage defaultMessage="Value" description="Usage overview > filter row > value column label" />
        </FormUI.Label>
        {selectedOption?.valueOptions ? (
          <DialogCombobox
            componentId="mlflow.usage.metrics_filter.value"
            id={`usage-metrics-filter-value-${rowId}`}
            value={filter.value ? [filter.value] : []}
          >
            <DialogComboboxTrigger
              withInlineLabel={false}
              placeholder={intl.formatMessage({
                defaultMessage: 'Select value',
                description: 'Usage overview > filter row > value dropdown placeholder',
              })}
              renderDisplayedValue={() =>
                selectedOption.valueOptions?.find((o) => o.value === filter.value)?.label ?? ''
              }
              width={200}
              allowClear={false}
            />
            <DialogComboboxContent width={200} style={{ zIndex: theme.options.zIndexBase + 100 }}>
              <DialogComboboxOptionList>
                <DialogComboboxOptionListSearch onSearch={() => {}}>
                  {selectedOption.valueOptions.map((option) => (
                    <DialogComboboxOptionListSelectItem
                      key={option.value}
                      value={option.value}
                      checked={option.value === filter.value}
                      onChange={(value: string) => onChange({ ...filter, value })}
                    >
                      {option.label}
                    </DialogComboboxOptionListSelectItem>
                  ))}
                </DialogComboboxOptionListSearch>
              </DialogComboboxOptionList>
            </DialogComboboxContent>
          </DialogCombobox>
        ) : (
          <Input
            componentId="mlflow.usage.metrics_filter.value"
            id={`usage-metrics-filter-value-${rowId}`}
            value={filter.value}
            onChange={(e) => onChange({ ...filter, value: e.target.value })}
            placeholder={intl.formatMessage({
              defaultMessage: 'Enter value',
              description: 'Usage overview > filter row > value input placeholder',
            })}
          />
        )}
      </div>

      {/* Delete row */}
      <Button
        componentId="mlflow.usage.metrics_filter.delete_row"
        icon={<CloseIcon />}
        onClick={onDelete}
        aria-label={intl.formatMessage({
          defaultMessage: 'Remove filter',
          description: 'Usage overview > filter row > remove row button aria label',
        })}
      />
    </div>
  );
};
