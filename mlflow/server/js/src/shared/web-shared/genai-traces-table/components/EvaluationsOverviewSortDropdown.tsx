import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import {
  ArrowDownIcon,
  ArrowUpIcon,
  SortAscendingIcon,
  SortDescendingIcon,
  Input,
  SearchIcon,
  useDesignSystemTheme,
  DropdownMenu,
  Button,
  ChevronDownIcon,
  SortUnsortedIcon,
  ToggleButton,
  Spinner,
  DangerIcon,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import { sortGroupedColumns } from '../GenAiTracesTable.utils';
import { SESSION_COLUMN_ID, SORTABLE_INFO_COLUMNS } from '../hooks/useTableColumns';
import type { EvaluationsOverviewTableSort, AssessmentInfo, TracesTableColumn } from '../types';
import { TracesTableColumnType, TracesTableColumnGroup, TracesTableColumnGroupToLabelMap } from '../types';

export interface SortOption {
  label: string;
  key: string;
  type: TracesTableColumnType;
  group?: TracesTableColumnGroup;
}

export const EvaluationsOverviewSortDropdown = React.memo(
  ({
    tableSort,
    columns = [],
    onChange,
    enableGrouping,
    isMetadataLoading,
    metadataError,
  }: {
    tableSort: EvaluationsOverviewTableSort | undefined;
    columns: TracesTableColumn[];
    onChange: (sortOption: SortOption, orderByAsc: boolean) => void;
    enableGrouping?: boolean;
    isMetadataLoading?: boolean;
    metadataError?: Error | null;
  }) => {
    const intl = useIntl();
    const { theme } = useDesignSystemTheme();
    const [open, setOpen] = useState(false);

    const sortOptions = useMemo(() => {
      const sortOptions: SortOption[] = [];

      const assessmentLabelPrefix = intl.formatMessage({
        defaultMessage: 'Assessment',
        description: 'Evaluation review > evaluations list > assessment sort option label prefix',
      });

      if (enableGrouping) {
        const sortedColumns = sortGroupedColumns(columns);

        for (const column of sortedColumns) {
          if (column.type === TracesTableColumnType.ASSESSMENT) {
            const sortableAssessmentInfo = column.assessmentInfo as AssessmentInfo;
            const assessmentLabel = sortableAssessmentInfo.displayName;

            sortOptions.push({
              label: assessmentLabel,
              key: column.id,
              type: TracesTableColumnType.ASSESSMENT,
              group: TracesTableColumnGroup.ASSESSMENT,
            });
          } else if (column.type === TracesTableColumnType.INPUT) {
            sortOptions.push({
              label: column.label,
              key: column.id,
              type: TracesTableColumnType.INPUT,
              group: TracesTableColumnGroup.INFO,
            });
          } else if (column.type === TracesTableColumnType.TRACE_INFO) {
            const label =
              column.id === SESSION_COLUMN_ID
                ? intl.formatMessage({
                    defaultMessage: 'Session time',
                    description: 'Session time sort option label',
                  })
                : column.label;
            if (SORTABLE_INFO_COLUMNS.includes(column.id)) {
              sortOptions.push({
                label,
                key: column.id,
                type: TracesTableColumnType.TRACE_INFO,
                group: TracesTableColumnGroup.INFO,
              });
            }
          }
        }

        return sortOptions;
      }

      // First add assessments.
      for (const sortableAssessmentCol of columns.filter(
        (column) => column.type === TracesTableColumnType.ASSESSMENT,
      )) {
        const sortableAssessmentInfo = sortableAssessmentCol.assessmentInfo as AssessmentInfo;
        const assessmentLabel = sortableAssessmentInfo.displayName;
        sortOptions.push({
          label: `${assessmentLabelPrefix} ﹥ ${assessmentLabel}`,
          key: sortableAssessmentCol.id,
          type: TracesTableColumnType.ASSESSMENT,
        });
      }
      // Next add inputs.
      const inputLabelPrefix = intl.formatMessage({
        defaultMessage: 'Input',
        description: 'Evaluation review > evaluations list > input sort option label prefix',
      });
      for (const inputColumn of columns.filter((column) => column.type === TracesTableColumnType.INPUT)) {
        sortOptions.push({
          label: `${inputLabelPrefix} ﹥ ${inputColumn.label}`,
          key: inputColumn.id,
          type: TracesTableColumnType.INPUT,
        });
      }

      // Add info columns
      for (const infoColumn of columns.filter(
        (column) =>
          (column.type === TracesTableColumnType.TRACE_INFO ||
            column.type === TracesTableColumnType.INTERNAL_MONITOR_REQUEST_TIME) &&
          SORTABLE_INFO_COLUMNS.includes(column.id),
      )) {
        sortOptions.push({
          label: infoColumn.label,
          key: infoColumn.id,
          type: TracesTableColumnType.TRACE_INFO,
        });
      }

      return sortOptions;
    }, [columns, intl, enableGrouping]);

    // Generate the label for the sort select dropdown
    const currentSortSelectLabel = useMemo(() => {
      // Search through all sort options generated basing on the fetched runs
      const sortOption = sortOptions.find((option) => option.key === tableSort?.key);

      let sortOptionLabel = sortOption?.label;

      // If the actually chosen sort value is not found in the sort option list (e.g. because the list of fetched runs is empty),
      // use it to generate the label
      if (!sortOptionLabel) {
        // The following regex extracts plain sort key name from its canonical form, i.e.
        // metrics.`metric_key_name` => metric_key_name
        const extractedKeyName = tableSort?.key?.match(/^.+\.`(.+)`$/);
        if (extractedKeyName) {
          // eslint-disable-next-line prefer-destructuring
          sortOptionLabel = extractedKeyName[1];
        }
      }
      const sortMessage = intl.formatMessage({
        defaultMessage: 'Sort',
        description: 'Experiment page > sort selector > label for the dropdown button',
      });

      return !sortOptionLabel ? sortMessage : `${sortMessage}: ${sortOptionLabel}`;
    }, [sortOptions, intl, tableSort]);

    return (
      <DropdownMenu.Root open={open} onOpenChange={setOpen} modal={false}>
        <DropdownMenu.Trigger data-testid="sort-select-dropdown" asChild>
          <Button
            componentId="mlflow.experiment_page.sort_select_v2.toggle"
            icon={tableSort ? tableSort.asc ? <SortAscendingIcon /> : <SortDescendingIcon /> : <SortUnsortedIcon />}
            css={{ minWidth: 32 }}
            aria-label={currentSortSelectLabel}
            endIcon={<ChevronDownIcon />}
          >
            {currentSortSelectLabel}
          </Button>
        </DropdownMenu.Trigger>
        <DropdownMenu.Content minWidth={250}>
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
                padding: `${theme.spacing.lg}px`,
                color: theme.colors.textSecondary,
              }}
              data-testid="sort-dropdown-loading"
            >
              <Spinner size="small" />
            </div>
          ) : enableGrouping ? (
            <EvaluationsOverviewSortDropdownBodyGrouped
              sortOptions={sortOptions}
              tableSort={tableSort}
              onOptionSelected={(sortOption, orderByAsc) => {
                onChange(sortOption, orderByAsc);
                setOpen(false);
              }}
            />
          ) : (
            <EvaluationsOverviewSortDropdownBody
              sortOptions={sortOptions}
              tableSort={tableSort}
              onOptionSelected={(sortOption, orderByAsc) => {
                onChange(sortOption, orderByAsc);
                setOpen(false);
              }}
            />
          )}
        </DropdownMenu.Content>
      </DropdownMenu.Root>
    );
  },
);

const EvaluationsOverviewSortDropdownBodyGrouped = ({
  sortOptions,
  tableSort,
  onOptionSelected,
}: {
  sortOptions: SortOption[];
  tableSort?: EvaluationsOverviewTableSort;
  onOptionSelected: (opt: SortOption, asc: boolean) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const inputRef = useRef<React.ComponentRef<typeof Input>>(null);
  const firstOptionRef = useRef<HTMLDivElement>(null);

  const [filter, setFilter] = useState<string>('');

  const filtered = useMemo(
    () => sortOptions.filter((opt) => opt.label.toLowerCase().includes(filter.toLowerCase())),
    [sortOptions, filter],
  );

  const grouped = useMemo(() => {
    const m: Record<string, SortOption[]> = {};
    filtered.forEach((opt) => {
      const grp = opt.group ?? TracesTableColumnGroup.INFO;
      if (!m[grp]) m[grp] = [];
      m[grp].push(opt);
    });
    return m;
  }, [filtered]);

  const handleChange = useCallback(
    (key: string) => {
      const opt = sortOptions.find((o) => o.key === key);
      if (!opt) return;
      onOptionSelected(opt, tableSort?.asc ?? true);
    },
    [onOptionSelected, sortOptions, tableSort?.asc],
  );

  const setOrder = useCallback(
    (asc: boolean) => {
      const opt = sortOptions.find((o) => o.key === tableSort?.key) ?? sortOptions[0];
      onOptionSelected(opt, asc);
    },
    [onOptionSelected, sortOptions, tableSort?.key],
  );

  // Autofocus won't work everywhere so let's focus input everytime the dropdown is opened
  useEffect(() => {
    requestAnimationFrame(() => inputRef.current?.focus());
  }, []);

  return (
    <>
      <div
        css={{
          padding: `${theme.spacing.sm}px ${theme.spacing.lg / 2}px`,
          display: 'flex',
          gap: theme.spacing.xs,
        }}
      >
        <Input
          ref={inputRef}
          componentId="mlflow.genai_traces_table.sort_dropdown.search"
          prefix={<SearchIcon />}
          placeholder="Search"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          type="search"
          onKeyDown={(e) => {
            if (e.key === 'ArrowDown' || e.key === 'Tab') {
              firstOptionRef.current?.focus();
              e.preventDefault();
            }
          }}
        />
        <div css={{ display: 'flex', gap: theme.spacing.xs }}>
          <ToggleButton
            pressed={tableSort?.asc === false}
            icon={<ArrowDownIcon />}
            componentId="mlflow.genai_traces_table.sort_dropdown.sort_desc"
            onClick={() => setOrder(false)}
            aria-label="Sort descending"
          />
          <ToggleButton
            pressed={tableSort?.asc === true}
            icon={<ArrowUpIcon />}
            componentId="mlflow.experiment_page.sort_dropdown.sort_asc"
            onClick={() => setOrder(true)}
            aria-label="Sort ascending"
          />
        </div>
      </div>

      <div css={{ maxHeight: 400, overflowY: 'auto', padding: `0 ${theme.spacing.sm}px` }}>
        {Object.entries(grouped).map(([groupName, opts], gi) => (
          <React.Fragment key={groupName}>
            <DropdownMenu.Group>
              <DropdownMenu.Label>
                {groupName === TracesTableColumnGroup.INFO
                  ? 'Attributes'
                  : TracesTableColumnGroupToLabelMap[groupName as TracesTableColumnGroup]}
              </DropdownMenu.Label>
              {opts.map((opt, idx) => (
                <DropdownMenu.CheckboxItem
                  componentId="mlflow.genai_traces_table.sort_dropdown.sort_option"
                  key={opt.key}
                  checked={opt.key === tableSort?.key}
                  onClick={() => handleChange(opt.key)}
                  ref={gi === 0 && idx === 0 ? firstOptionRef : undefined}
                  data-testid={`sort-select-${opt.label}`}
                >
                  <DropdownMenu.ItemIndicator />
                  <span css={{ display: 'flex', alignItems: 'center', gap: 4 }}>{opt.label}</span>
                </DropdownMenu.CheckboxItem>
              ))}
            </DropdownMenu.Group>
          </React.Fragment>
        ))}

        {/* No results fallback */}
        {filtered.length === 0 && (
          <DropdownMenu.Item disabled componentId="mlflow.genai_traces_table.sort_dropdown.no_results">
            <FormattedMessage
              defaultMessage="No results"
              description="Experiment page > sort selector > no results after filtering"
            />
          </DropdownMenu.Item>
        )}
      </div>
    </>
  );
};

const EvaluationsOverviewSortDropdownBody = ({
  sortOptions,
  tableSort,
  onOptionSelected,
}: {
  sortOptions: SortOption[];
  tableSort?: EvaluationsOverviewTableSort;
  onOptionSelected: (sortOption: SortOption, orderByAsc: boolean) => void;
}) => {
  const { theme } = useDesignSystemTheme();

  const inputElementRef = useRef<React.ComponentRef<typeof Input>>(null);
  const [filter, setFilter] = useState('');
  const firstElementRef = useRef<HTMLDivElement>(null);

  // Merge all sort options and filter them by the search query
  const filteredSortOptions = useMemo(
    () =>
      sortOptions.filter((option) => {
        return option.label.toLowerCase().includes(filter.toLowerCase());
      }),
    [sortOptions, filter],
  );

  const handleChange = useCallback(
    (orderByKey: string) => {
      const orderByKeyOption = sortOptions.find((option) => option.key === orderByKey);
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion -- TODO(FEINF-3982)
      onOptionSelected(orderByKeyOption!, tableSort?.asc || true);
    },
    [onOptionSelected, tableSort, sortOptions],
  );

  const setOrder = useCallback(
    (ascending: boolean) => {
      const orderByKeyOption = sortOptions.find((option) => option.key === tableSort?.key);
      onOptionSelected(orderByKeyOption || sortOptions[0], ascending);
    },
    [onOptionSelected, sortOptions, tableSort],
  );

  // Autofocus won't work everywhere so let's focus input everytime the dropdown is opened
  useEffect(() => {
    requestAnimationFrame(() => {
      inputElementRef.current?.focus();
    });
  }, []);

  return (
    <>
      <div
        css={{
          padding: `${theme.spacing.sm}px ${theme.spacing.lg / 2}px ${theme.spacing.sm}px`,
          width: '100%',
          display: 'flex',
          gap: theme.spacing.xs,
        }}
      >
        <Input
          componentId="mlflow.experiment_page.sort_dropdown.search"
          prefix={<SearchIcon />}
          value={filter}
          type="search"
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Search"
          autoFocus
          ref={inputElementRef}
          onKeyDown={(e) => {
            if (e.key === 'ArrowDown' || e.key === 'Tab') {
              firstElementRef.current?.focus();
              return;
            }
            e.stopPropagation();
          }}
        />
        <div
          css={{
            display: 'flex',
            gap: theme.spacing.xs,
          }}
        >
          <ToggleButton
            pressed={tableSort?.asc === false}
            icon={<ArrowDownIcon />}
            componentId="mlflow.experiment_page.sort_dropdown.sort_desc"
            onClick={() => setOrder(false)}
            aria-label="Sort descending"
            data-testid="sort-select-desc"
          />
          <ToggleButton
            pressed={tableSort?.asc === true}
            icon={<ArrowUpIcon />}
            componentId="mlflow.experiment_page.sort_dropdown.sort_asc"
            onClick={() => setOrder(true)}
            aria-label="Sort ascending"
            data-testid="sort-select-asc"
          />
        </div>
      </div>
      <DropdownMenu.Group css={{ maxHeight: 400, overflowY: 'auto' }}>
        {filteredSortOptions.map((sortOption, index) => (
          <DropdownMenu.CheckboxItem
            componentId="mlflow.experiment_page.sort_dropdown.sort_option"
            key={sortOption.key}
            onClick={() => handleChange(sortOption.key)}
            checked={sortOption.key === tableSort?.key}
            data-testid={`sort-select-${sortOption.label}`}
            ref={index === 0 ? firstElementRef : undefined}
          >
            <DropdownMenu.ItemIndicator />
            <span css={{ display: 'flex', alignItems: 'center', gap: 4 }}>{sortOption.label}</span>
          </DropdownMenu.CheckboxItem>
        ))}
        {!filteredSortOptions.length && (
          <DropdownMenu.Item
            componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunssortselectorv2.tsx_151"
            disabled
          >
            <FormattedMessage
              defaultMessage="No results"
              description="Experiment page > sort selector > no results after filtering by search query"
            />
          </DropdownMenu.Item>
        )}
      </DropdownMenu.Group>
    </>
  );
};
