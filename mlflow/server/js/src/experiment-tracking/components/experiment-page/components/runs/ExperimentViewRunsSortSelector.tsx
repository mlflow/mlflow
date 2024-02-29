import {
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSearch,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  ArrowDownIcon,
  ArrowUpIcon,
  SortAscendingIcon,
  SortDescendingIcon,
} from '@databricks/design-system';
import React, { useMemo, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { middleTruncateStr } from '../../../../../common/utils/StringUtils';
import { COLUMN_SORT_BY_ASC, COLUMN_SORT_BY_DESC, SORT_DELIMITER_SYMBOL } from '../../../../constants';
import { ExperimentRunSortOption } from '../../hooks/useRunSortOptions';
import { useUpdateExperimentPageSearchFacets } from '../../hooks/useExperimentPageSearchFacets';
import { shouldEnableShareExperimentViewByTags } from '../../../../../common/utils/FeatureUtils';
import { useUpdateExperimentViewUIState } from '../../contexts/ExperimentPageUIStateContext';

export const ExperimentViewRunsSortSelector = React.memo(
  (props: {
    orderByKey: string;
    orderByAsc: boolean;
    sortOptions: ExperimentRunSortOption[];
    onSortKeyChanged: (valueContainer: any) => void;
  }) => {
    const usingNewViewStateModel = shouldEnableShareExperimentViewByTags();
    const setUrlSearchFacets = useUpdateExperimentPageSearchFacets();
    const updateUIState = useUpdateExperimentViewUIState();
    const intl = useIntl();

    const { sortOptions } = props;
    const { orderByKey, orderByAsc } = props;

    // In the new view state model, manipulate URL search facets directly instead of using the callback
    const onSortKeyChanged = usingNewViewStateModel
      ? ({ value }: { value: string }) => {
          const [newOrderBy, newOrderAscending] = value.split(SORT_DELIMITER_SYMBOL);

          setUrlSearchFacets({
            orderByAsc: newOrderAscending === COLUMN_SORT_BY_ASC,
            orderByKey: newOrderBy,
          });

          updateUIState((currentUIState) => {
            if (!currentUIState.selectedColumns.includes(newOrderBy)) {
              return {
                ...currentUIState,
                selectedColumns: [...currentUIState.selectedColumns, newOrderBy],
              };
            }
            return currentUIState;
          });
        }
      : props.onSortKeyChanged;

    // Currently used canonical "sort by" value in form of "COLUMN_NAME***DIRECTION", e.g. "metrics.`metric`***DESCENDING"
    const currentSortSelectValue = useMemo(
      () => `${orderByKey}${SORT_DELIMITER_SYMBOL}${orderByAsc ? COLUMN_SORT_BY_ASC : COLUMN_SORT_BY_DESC}`,
      [orderByAsc, orderByKey],
    );

    /**
     * Calculate and memoize a label displayed in the "sort by" select.
     *
     * If full metrics and params list is populated by runs from the API, use the
     * value corresponding to the calculated sort option list.
     *
     * If the sort option list is incomplete (e.g. because fetched run set is empty) while the
     * order key is given (e.g. because URL state says so), use it to extract the key name.
     */
    const currentSortSelectLabel = useMemo(() => {
      // Search through all sort options generated basing on the fetched runs
      const sortOption = sortOptions.find((option) => option.value === currentSortSelectValue);

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
        description: 'Sort by default option for sort by select dropdown for experiment runs',
      })}: ${sortOptionLabel}`;
    }, [currentSortSelectValue, orderByKey, sortOptions, intl]);

    const sortLabelElement = useMemo(() => {
      return (
        <span css={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          {orderByAsc ? <SortAscendingIcon /> : <SortDescendingIcon />} {currentSortSelectLabel}
        </span>
      );
    }, [currentSortSelectLabel, orderByAsc]);

    const handleChange = (updatedValue: string) => {
      onSortKeyChanged({ value: updatedValue });
      setOpen(false);
    };

    const handleClear = () => {
      onSortKeyChanged({ value: '' });
    };

    const [open, setOpen] = useState(false);

    return (
      <DialogCombobox label={sortLabelElement} onOpenChange={setOpen} open={open}>
        <DialogComboboxTrigger
          onClear={handleClear}
          data-test-id="sort-select-dropdown"
          aria-label={currentSortSelectLabel}
        />
        <DialogComboboxContent minWidth={250}>
          <DialogComboboxOptionList>
            <DialogComboboxOptionListSearch>
              {sortOptions.map((sortOption) => (
                <DialogComboboxOptionListSelectItem
                  key={sortOption.value}
                  value={sortOption.value}
                  onChange={handleChange}
                  checked={sortOption.value === currentSortSelectValue}
                  data-test-id={`sort-select-${sortOption.label}-${sortOption.order}`}
                >
                  <span css={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    {sortOption.order === COLUMN_SORT_BY_ASC ? <ArrowUpIcon /> : <ArrowDownIcon />}
                    {middleTruncateStr(sortOption.label, 50)}
                  </span>
                </DialogComboboxOptionListSelectItem>
              ))}
            </DialogComboboxOptionListSearch>
          </DialogComboboxOptionList>
        </DialogComboboxContent>
      </DialogCombobox>
    );
  },
);
