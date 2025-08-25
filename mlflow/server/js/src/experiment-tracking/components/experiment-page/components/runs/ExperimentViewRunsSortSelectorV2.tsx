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
} from '@databricks/design-system';
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { middleTruncateStr } from '../../../../../common/utils/StringUtils';
import { ATTRIBUTE_COLUMN_SORT_KEY, ATTRIBUTE_COLUMN_SORT_LABEL, COLUMN_TYPES } from '../../../../constants';
import { useUpdateExperimentPageSearchFacets } from '../../hooks/useExperimentPageSearchFacets';
import { useUpdateExperimentViewUIState } from '../../contexts/ExperimentPageUIStateContext';
import { ToggleIconButton } from '../../../../../common/components/ToggleIconButton';
import { makeCanonicalSortKey } from '../../utils/experimentPage.common-utils';
import { customMetricBehaviorDefs } from '../../utils/customMetricBehaviorUtils';

type SORT_KEY_TYPE = keyof (typeof ATTRIBUTE_COLUMN_SORT_KEY & typeof ATTRIBUTE_COLUMN_SORT_LABEL);

const ExperimentViewRunsSortSelectorV2Body = ({
  sortOptions,
  orderByKey,
  orderByAsc,
  onOptionSelected,
}: {
  sortOptions: {
    label: string;
    value: string;
  }[];
  orderByKey: string;
  orderByAsc: boolean;
  onOptionSelected: () => void;
}) => {
  const { theme } = useDesignSystemTheme();

  const setUrlSearchFacets = useUpdateExperimentPageSearchFacets();
  const updateUIState = useUpdateExperimentViewUIState();
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

  const handleChange = (orderByKey: string) => {
    setUrlSearchFacets({
      orderByKey,
    });

    updateUIState((currentUIState) => {
      if (!currentUIState.selectedColumns.includes(orderByKey)) {
        return {
          ...currentUIState,
          selectedColumns: [...currentUIState.selectedColumns, orderByKey],
        };
      }
      return currentUIState;
    });

    onOptionSelected();
  };
  const setOrder = (ascending: boolean) => {
    setUrlSearchFacets({
      orderByAsc: ascending,
    });
    onOptionSelected();
  };

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
          componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunssortselectorv2.tsx_97"
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
          <ToggleIconButton
            pressed={!orderByAsc}
            icon={<ArrowDownIcon />}
            componentId="mlflow.experiment_page.sort_select_v2.sort_desc"
            onClick={() => setOrder(false)}
            aria-label="Sort descending"
            data-testid="sort-select-desc"
          />
          <ToggleIconButton
            pressed={orderByAsc}
            icon={<ArrowUpIcon />}
            componentId="mlflow.experiment_page.sort_select_v2.sort_asc"
            onClick={() => setOrder(true)}
            aria-label="Sort ascending"
            data-testid="sort-select-asc"
          />
        </div>
      </div>
      <DropdownMenu.Group css={{ maxHeight: 400, overflowY: 'auto' }}>
        {filteredSortOptions.map((sortOption, index) => (
          <DropdownMenu.CheckboxItem
            componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunssortselectorv2.tsx_137"
            key={sortOption.value}
            onClick={() => handleChange(sortOption.value)}
            checked={sortOption.value === orderByKey}
            data-testid={`sort-select-${sortOption.label}`}
            ref={index === 0 ? firstElementRef : undefined}
          >
            <DropdownMenu.ItemIndicator />
            <span css={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              {middleTruncateStr(sortOption.label, 50)}
            </span>
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

export const ExperimentViewRunsSortSelectorV2 = React.memo(
  ({
    metricKeys,
    paramKeys,
    orderByAsc,
    orderByKey,
  }: {
    orderByKey: string;
    orderByAsc: boolean;
    metricKeys: string[];
    paramKeys: string[];
  }) => {
    const intl = useIntl();
    const [open, setOpen] = useState(false);
    const { theme } = useDesignSystemTheme();

    // Get sort options for attributes (e.g. start time, run name, etc.)
    const attributeSortOptions = useMemo(
      () =>
        Object.keys(ATTRIBUTE_COLUMN_SORT_LABEL).map((sortLabelKey) => ({
          label: ATTRIBUTE_COLUMN_SORT_LABEL[sortLabelKey as SORT_KEY_TYPE],
          value: ATTRIBUTE_COLUMN_SORT_KEY[sortLabelKey as SORT_KEY_TYPE],
        })),
      [],
    );

    // Get sort options for metrics
    const metricsSortOptions = useMemo(
      () =>
        metricKeys.map((sortLabelKey) => {
          const canonicalSortKey = makeCanonicalSortKey(COLUMN_TYPES.METRICS, sortLabelKey);
          const displayName = customMetricBehaviorDefs[sortLabelKey]?.displayName ?? sortLabelKey;
          return {
            label: displayName,
            value: canonicalSortKey,
          };
        }),
      [
        // A list of metric key names that need to be turned into canonical sort keys
        metricKeys,
      ],
    );

    // Get sort options for params
    const paramsSortOptions = useMemo(
      () =>
        paramKeys.map((sortLabelKey) => ({
          label: sortLabelKey,
          value: `${makeCanonicalSortKey(COLUMN_TYPES.PARAMS, sortLabelKey)}`,
        })),
      [paramKeys],
    );

    const sortOptions = useMemo(
      () => [...attributeSortOptions, ...metricsSortOptions, ...paramsSortOptions],
      [attributeSortOptions, metricsSortOptions, paramsSortOptions],
    );

    // Generate the label for the sort select dropdown
    const currentSortSelectLabel = useMemo(() => {
      // Search through all sort options generated basing on the fetched runs
      const sortOption = sortOptions.find((option) => option.value === orderByKey);

      let sortOptionLabel = sortOption?.label;

      // If the actually chosen sort value is not found in the sort option list (e.g. because the list of fetched runs is empty),
      // use it to generate the label
      if (!sortOptionLabel) {
        // The following regex extracts plain sort key name from its canonical form, i.e.
        // metrics.`metric_key_name` => metric_key_name
        const extractedKeyName = orderByKey.match(/^.+\.`(.+)`$/);
        if (extractedKeyName) {
          // eslint-disable-next-line prefer-destructuring
          sortOptionLabel = extractedKeyName[1];
        }
      }
      return `${intl.formatMessage({
        defaultMessage: 'Sort',
        description: 'Experiment page > sort selector > label for the dropdown button',
      })}: ${sortOptionLabel}`;
    }, [sortOptions, intl, orderByKey]);

    return (
      <DropdownMenu.Root open={open} onOpenChange={setOpen} modal={false}>
        <DropdownMenu.Trigger data-testid="sort-select-dropdown" asChild>
          <Button
            componentId="mlflow.experiment_page.sort_select_v2.toggle"
            icon={orderByAsc ? <SortAscendingIcon /> : <SortDescendingIcon />}
            css={{ minWidth: 32 }}
            aria-label={currentSortSelectLabel}
            endIcon={<ChevronDownIcon />}
          >
            {currentSortSelectLabel}
          </Button>
        </DropdownMenu.Trigger>
        <DropdownMenu.Content minWidth={250}>
          <ExperimentViewRunsSortSelectorV2Body
            sortOptions={sortOptions}
            orderByKey={orderByKey}
            orderByAsc={orderByAsc}
            onOptionSelected={() => setOpen(false)}
          />
        </DropdownMenu.Content>
      </DropdownMenu.Root>
    );
  },
);
