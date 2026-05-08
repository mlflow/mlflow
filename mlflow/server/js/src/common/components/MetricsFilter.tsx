import { useState } from 'react';
import {
  Button,
  ChevronDownIcon,
  CloseIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
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
import type {
  MetricFilter
} from './MetricsFilter.utils';
import {
  isCompleteFilter,
  type MetricFilterColumn,
  type MetricFilterColumnOption,
} from './MetricsFilter.utils';

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
              <XCircleFillIcon
                css={{
                  fontSize: 12,
                  cursor: 'pointer',
                  color: theme.colors.grey400,
                  '&:hover': { color: theme.colors.grey600 },
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  setFilters([]);
                }}
              />
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
  const emptyFilter: MetricFilter = { column: columnOptions[0]?.value ?? ('' as MetricFilterColumn), value: '' };
  const [localFilters, setLocalFilters] = useState<MetricFilter[]>(
    filters.length > 0 ? filters : [emptyFilter],
  );

  const updateAt = (index: number, next: MetricFilter) => {
    setLocalFilters((prev) => prev.map((f, i) => (i === index ? next : f)));
  };

  const removeAt = (index: number) => {
    setLocalFilters((prev) => {
      const next = prev.filter((_, i) => i !== index);
      return next.length === 0 ? [emptyFilter] : next;
    });
  };

  const addRow = () => {
    setLocalFilters((prev) => [...prev, emptyFilter]);
  };

  const apply = () => {
    setFilters(localFilters.filter(isCompleteFilter));
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        {localFilters.map((filter, index) => (
          <FilterRow
            key={index}
            index={index}
            filter={filter}
            columnOptions={columnOptions}
            onChange={(next) => updateAt(index, next)}
            onDelete={() => removeAt(index)}
          />
        ))}
      </div>
      <div>
        <Button
          componentId="mlflow.usage.metrics_filter.add_row"
          icon={<PlusIcon />}
          onClick={addRow}
        >
          <FormattedMessage
            defaultMessage="Add filter"
            description="Usage overview > add filter row button"
          />
        </Button>
      </div>
      <div css={{ display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          componentId="mlflow.usage.metrics_filter.apply"
          type="primary"
          onClick={apply}
        >
          <FormattedMessage
            defaultMessage="Apply filters"
            description="Usage overview > apply filters button"
          />
        </Button>
      </div>
    </div>
  );
};

interface FilterRowProps {
  index: number;
  filter: MetricFilter;
  columnOptions: MetricFilterColumnOption[];
  onChange: (next: MetricFilter) => void;
  onDelete: () => void;
}

const FilterRow = ({ index, filter, columnOptions, onChange, onDelete }: FilterRowProps) => {
  const { theme } = useDesignSystemTheme();
  const selectedOption = columnOptions.find((o) => o.value === filter.column);

  return (
    <div css={{ display: 'flex', flexDirection: 'row', gap: theme.spacing.sm, alignItems: 'flex-end' }}>
      {/* Field (Column) */}
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor={`usage-metrics-filter-column-${index}`}>
          <FormattedMessage
            defaultMessage="Field"
            description="Usage overview > filter row > field column label"
          />
        </FormUI.Label>
        <DialogCombobox
          componentId="mlflow.usage.metrics_filter.column"
          id={`usage-metrics-filter-column-${index}`}
          value={filter.column ? [filter.column] : []}
        >
          <DialogComboboxTrigger
            withInlineLabel={false}
            placeholder="Select field"
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
                  onChange={(value: string) => onChange({ ...filter, column: value as MetricFilterColumn })}
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
        <FormUI.Label htmlFor={`usage-metrics-filter-operator-${index}`}>
          <FormattedMessage
            defaultMessage="Operator"
            description="Usage overview > filter row > operator column label"
          />
        </FormUI.Label>
        <SimpleSelect
          aria-label="Operator"
          componentId="mlflow.usage.metrics_filter.operator"
          id={`usage-metrics-filter-operator-${index}`}
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
        <FormUI.Label htmlFor={`usage-metrics-filter-value-${index}`}>
          <FormattedMessage
            defaultMessage="Value"
            description="Usage overview > filter row > value column label"
          />
        </FormUI.Label>
        <Input
          componentId="mlflow.usage.metrics_filter.value"
          id={`usage-metrics-filter-value-${index}`}
          value={filter.value}
          onChange={(e) => onChange({ ...filter, value: e.target.value })}
          placeholder="Enter value"
        />
      </div>

      {/* Delete row */}
      <Button
        componentId="mlflow.usage.metrics_filter.delete_row"
        icon={<CloseIcon />}
        onClick={onDelete}
        aria-label="Remove filter"
      />
    </div>
  );
};
