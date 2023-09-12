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
  useDesignSystemTheme,
} from '@databricks/design-system';
import React, { useMemo } from 'react';
import { FormattedMessage } from 'react-intl';
import { middleTruncateStr } from '../../../../../common/utils/StringUtils';
import {
  COLUMN_SORT_BY_ASC,
  COLUMN_SORT_BY_DESC,
  SORT_DELIMITER_SYMBOL,
} from '../../../../constants';
import { ExperimentRunSortOption } from '../../hooks/useRunSortOptions';
import { SearchExperimentRunsFacetsState } from '../../models/SearchExperimentRunsFacetsState';

export const ExperimentViewRunsSortSelector = React.memo(
  ({
    searchFacetsState,
    sortOptions,
    onSortKeyChanged,
  }: {
    searchFacetsState: SearchExperimentRunsFacetsState;
    sortOptions: ExperimentRunSortOption[];
    onSortKeyChanged: (valueContainer: any) => void;
  }) => {
    const { orderByKey, orderByAsc } = searchFacetsState;
    const { theme } = useDesignSystemTheme();

    // Currently used canonical "sort by" value in form of "COLUMN_NAME***DIRECTION", e.g. "metrics.`metric`***DESCENDING"
    const currentSortSelectValue = useMemo(
      () =>
        `${orderByKey}${SORT_DELIMITER_SYMBOL}${
          orderByAsc ? COLUMN_SORT_BY_ASC : COLUMN_SORT_BY_DESC
        }`,
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
      return (
        <span css={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          {orderByAsc ? <SortAscendingIcon /> : <SortDescendingIcon />}{' '}
          <FormattedMessage
            defaultMessage='Sort'
            description='Sort by default option for sort by select dropdown for experiment runs'
          />
          : {sortOptionLabel}
        </span>
      );
    }, [currentSortSelectValue, orderByAsc, orderByKey, sortOptions]);

    const handleChange = (updatedValue: string) => {
      onSortKeyChanged({ value: updatedValue });
    };

    const handleClear = () => {
      onSortKeyChanged({ value: '' });
    };

    return (
      <DialogCombobox label={currentSortSelectLabel}>
        <DialogComboboxTrigger onClear={handleClear} data-test-id='sort-select-dropdown' />
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
