import {
  ArrowDownIcon,
  ArrowUpIcon,
  Option,
  Select,
  SortAscendingIcon,
  SortDescendingIcon,
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

    return (
      <Select
        className='sort-select'
        css={styles.sortSelectControl}
        value={{
          value: currentSortSelectValue,
          label: currentSortSelectLabel,
        }}
        labelInValue
        // Temporarily we're disabling virtualized list to maintain
        // backwards compatiblity. Functional unit tests rely heavily
        // on non-virtualized values.
        dangerouslySetAntdProps={
          { virtual: false, dropdownStyle: styles.sortSelectDropdown } as any
        }
        onChange={onSortKeyChanged}
        data-test-id='sort-select-dropdown'
      >
        {sortOptions.map((sortOption) => (
          <Option
            key={sortOption.value}
            title={sortOption.label}
            data-test-id={`sort-select-${sortOption.label}-${sortOption.order}`}
            value={sortOption.value}
          >
            <span css={styles.sortMenuArrowWrapper}>
              {sortOption.order === COLUMN_SORT_BY_ASC ? <ArrowUpIcon /> : <ArrowDownIcon />}
            </span>{' '}
            {middleTruncateStr(sortOption.label, 50)}
          </Option>
        ))}
      </Select>
    );
  },
);

const styles = {
  sortSelectControl: { minWidth: 140, maxWidth: 360 },
  sortSelectDropdown: { minWidth: 360 },
  sortMenuArrowWrapper: { svg: { width: 12, height: 12 } },
};
