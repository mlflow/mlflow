import { useCallback } from 'react';
import { FormProvider, useForm } from 'react-hook-form';

import {
  Button,
  ChevronDownIcon,
  useDesignSystemTheme,
  FilterIcon,
  Popover,
  PlusIcon,
  XCircleFillIcon,
  Spinner,
  DangerIcon,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import { TableFilterItem } from './components/filters/TableFilterItem';
import { FilterOperator } from './types';
import type { AssessmentInfo, TableFilterFormState, TableFilter, TableFilterOptions, TracesTableColumn } from './types';
import { FILTER_DROPDOWN_COMPONENT_ID } from './utils/EvaluationLogging';

export const GenAiTracesTableFilter = ({
  filters,
  setFilters,
  assessmentInfos,
  experimentId,
  tableFilterOptions,
  allColumns,
  isMetadataLoading,
  metadataError,
}: {
  filters: TableFilter[];
  setFilters: (filters: TableFilter[]) => void;
  assessmentInfos: AssessmentInfo[];
  experimentId: string;
  tableFilterOptions: TableFilterOptions;
  allColumns: TracesTableColumn[];
  isMetadataLoading?: boolean;
  metadataError?: Error | null;
}) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const hasActiveFilters = filters.length > 0;

  const clearFilters = useCallback(() => {
    setFilters([]);
  }, [setFilters]);

  return (
    <Popover.Root componentId={FILTER_DROPDOWN_COMPONENT_ID}>
      <Popover.Trigger asChild>
        <Button
          endIcon={<ChevronDownIcon />}
          componentId="mlflow.evaluations_review.table_ui.filter_button"
          css={{
            border: hasActiveFilters ? `1px solid ${theme.colors.actionDefaultBorderFocus} !important` : '',
            backgroundColor: hasActiveFilters ? `${theme.colors.actionDefaultBackgroundHover} !important` : '',
          }}
        >
          <div
            css={{
              display: 'flex',
              gap: theme.spacing.sm,
              alignItems: 'center',
            }}
          >
            <FilterIcon />
            {intl.formatMessage(
              {
                defaultMessage: 'Filters{numFilters}',
                description: 'Evaluation review > evaluations list > filter dropdown button',
              },
              {
                numFilters: hasActiveFilters ? ` (${filters.length})` : '',
              },
            )}
            {hasActiveFilters && (
              <XCircleFillIcon
                css={{
                  fontSize: 12,
                  cursor: 'pointer',
                  color: theme.colors.grey400,
                  '&:hover': {
                    color: theme.colors.grey600,
                  },
                }}
                onClick={() => {
                  clearFilters();
                }}
              />
            )}
          </div>
        </Button>
      </Popover.Trigger>
      <Popover.Content align="start" css={{ padding: theme.spacing.md }}>
        {metadataError ? (
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: theme.spacing.xs,
              padding: `${theme.spacing.md}px`,
              color: theme.colors.textValidationDanger,
            }}
            data-testid="filter-dropdown-error"
          >
            <DangerIcon />
            <FormattedMessage
              defaultMessage="Fetching traces failed"
              description="Error message for fetching traces failed"
            />
          </div>
        ) : isMetadataLoading ? (
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: theme.spacing.xs,
              padding: `${theme.spacing.md}px`,
              color: theme.colors.textSecondary,
            }}
            data-testid="filter-dropdown-loading"
          >
            <Spinner size="small" />
          </div>
        ) : (
          <FilterForm
            filters={filters}
            assessmentInfos={assessmentInfos}
            setFilters={setFilters}
            experimentId={experimentId}
            tableFilterOptions={tableFilterOptions}
            allColumns={allColumns}
          />
        )}
      </Popover.Content>
    </Popover.Root>
  );
};

const useFilterForm = (filters: TableFilter[]) => {
  const form = useForm<TableFilterFormState>({
    defaultValues: {
      filters: filters.length > 0 ? filters : [{ column: '', operator: FilterOperator.EQUALS, value: '' }],
    },
  });

  return form;
};

const FilterForm = ({
  filters,
  assessmentInfos,
  setFilters,
  experimentId,
  tableFilterOptions,
  allColumns,
}: {
  filters: TableFilter[];
  assessmentInfos: AssessmentInfo[];
  setFilters: (filters: TableFilter[]) => void;
  experimentId: string;
  tableFilterOptions: TableFilterOptions;
  allColumns: TracesTableColumn[];
}) => {
  const { theme } = useDesignSystemTheme();

  const filterForm = useFilterForm(filters);

  const { setValue, watch } = filterForm;

  const localFilters = watch('filters');

  return (
    <FormProvider {...filterForm}>
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.lg,
          overflow: 'auto',
        }}
      >
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
          }}
        >
          {localFilters.map((filter, index) => (
            <TableFilterItem
              key={filter.column + filter.operator + filter.value + index}
              tableFilter={filter}
              index={index}
              onChange={(newFilter) => {
                localFilters[index] = newFilter;
                setValue('filters', localFilters);
              }}
              onDelete={() => {
                const newFilters = [...localFilters];
                newFilters.splice(index, 1);
                // If there are no filters, add an initial filter
                if (newFilters.length === 0) {
                  newFilters.push({ column: '', operator: FilterOperator.EQUALS, value: '' });
                }
                setValue('filters', newFilters);
              }}
              assessmentInfos={assessmentInfos}
              experimentId={experimentId}
              tableFilterOptions={tableFilterOptions}
              allColumns={allColumns}
            />
          ))}
        </div>
        <div>
          <Button
            componentId="mlflow.evaluations_review.table_ui.add_filter_button"
            icon={<PlusIcon />}
            onClick={() => {
              setValue('filters', [...localFilters, { column: '', operator: FilterOperator.EQUALS, value: '' }]);
            }}
          >
            <FormattedMessage
              defaultMessage="Add filter"
              description="Button label for adding a filter in the GenAI Traces Table filter form"
            />
          </Button>
        </div>
        <div
          css={{
            display: 'flex',
            justifyContent: 'flex-end',
          }}
        >
          <Button
            componentId="mlflow.evaluations_review.table_ui.apply_filters_button"
            type="primary"
            onClick={() => setFilters(localFilters)}
            css={{
              display: 'flex',
              justifyContent: 'flex-end',
            }}
          >
            <FormattedMessage
              defaultMessage="Apply filters"
              description="Button label for applying the filters in the GenAI Traces Table"
            />
          </Button>
        </div>
      </div>
    </FormProvider>
  );
};
