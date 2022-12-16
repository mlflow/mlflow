import {
  ArrowDownIcon,
  ArrowUpIcon,
  Button,
  CloseIcon,
  DownloadIcon,
  Dropdown,
  InfoBorderIcon,
  Input,
  Menu,
  Option,
  OverflowIcon,
  Search1Icon,
  Select,
  SortAscendingIcon,
  SortDescendingIcon,
  Tooltip,
} from '@databricks/design-system';
import { Theme } from '@emotion/react';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { ExperimentSearchSyntaxDocUrl } from '../../../../../common/constants';
import { middleTruncateStr } from '../../../../../common/utils/StringUtils';
import {
  COLUMN_SORT_BY_ASC,
  COLUMN_SORT_BY_DESC,
  SORT_DELIMITER_SYMBOL,
} from '../../../../constants';
import { UpdateExperimentSearchFacetsFn, UpdateExperimentViewStateFn } from '../../../../types';
import { ExperimentRunSortOption } from '../../hooks/useRunSortOptions';
import { SearchExperimentRunsFacetsState } from '../../models/SearchExperimentRunsFacetsState';
import { SearchExperimentRunsViewState } from '../../models/SearchExperimentRunsViewState';
import { TAGS_TO_COLUMNS_MAP } from '../../utils/experimentPage.column-utils';
import { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { ExperimentViewRefreshButton } from './ExperimentViewRefreshButton';
import { ExperimentViewRunsColumnSelector } from './ExperimentViewRunsColumnSelector';
import { shouldUseNextRunsComparisonUI } from '../../../../../common/utils/FeatureUtils';
import { ExperimentViewRunsModeSwitch } from './ExperimentViewRunsModeSwitch';

// A default placeholder for the search box
const SEARCH_BOX_PLACEHOLDER = 'metrics.rmse < 1 and params.model = "tree"';

export type ExperimentViewRunsControlsFiltersProps = {
  searchFacetsState: SearchExperimentRunsFacetsState;
  updateSearchFacets: UpdateExperimentSearchFacetsFn;
  viewState: SearchExperimentRunsViewState;
  updateViewState: UpdateExperimentViewStateFn;
  sortOptions: ExperimentRunSortOption[];
  runsData: ExperimentRunsSelectorResult;
  onDownloadCsv: () => void;
};

export const ExperimentViewRunsControlsFilters = React.memo(
  ({
    searchFacetsState,
    updateSearchFacets,
    runsData,
    viewState,
    updateViewState,
    onDownloadCsv,
    sortOptions,
  }: ExperimentViewRunsControlsFiltersProps) => {
    const { orderByKey, orderByAsc, isComparingRuns } = searchFacetsState;

    const [searchFilterValue, setSearchFilterValue] = useState<string>();

    // Callback fired when the sort column value changes
    const sortKeyChanged = useCallback(
      ({ value: compiledOrderByKey }) => {
        const [newOrderBy, newOrderAscending] = compiledOrderByKey.split(SORT_DELIMITER_SYMBOL);

        const columnToAdd = TAGS_TO_COLUMNS_MAP[newOrderBy] || newOrderBy;
        const isOrderAscending = newOrderAscending === COLUMN_SORT_BY_ASC;

        updateSearchFacets((currentFacets) => {
          const { selectedColumns } = currentFacets;
          if (!selectedColumns.includes(columnToAdd)) {
            selectedColumns.push(columnToAdd);
          }
          return {
            ...currentFacets,
            selectedColumns,
            orderByKey: newOrderBy,
            orderByAsc: isOrderAscending,
          };
        });
      },
      [updateSearchFacets],
    );

    // Callback fired when search filter is being used
    const triggerSearch: React.KeyboardEventHandler<HTMLInputElement> = (e) => {
      if (e.key === 'Enter') {
        updateSearchFacets({ searchFilter: searchFilterValue });
      }
    };

    // Each time we're setting search filter externally, update it here as well
    useEffect(() => {
      setSearchFilterValue(searchFacetsState.searchFilter);
    }, [searchFacetsState]);

    // Callback fired when "clear" button is clicked
    const clearFacetsState = useCallback(() => {
      const cleanSearchFacetsState = new SearchExperimentRunsFacetsState();
      const { selectedColumns, runsExpanded, runsPinned } = searchFacetsState;
      updateSearchFacets(
        Object.assign(cleanSearchFacetsState, {
          runsExpanded,
          runsPinned,
          selectedColumns,
        }),
      );
    }, [searchFacetsState, updateSearchFacets]);

    // Shows or hides the column selector
    const changeColumnSelectorVisible = useCallback(
      (value: boolean) => updateViewState({ columnSelectorVisible: value }),
      [updateViewState],
    );

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
      <>
        <div css={styles.groupBar}>
          <div css={styles.controlBar}>
            {shouldUseNextRunsComparisonUI() && (
              <ExperimentViewRunsModeSwitch
                isComparingRuns={isComparingRuns}
                setIsComparingRuns={(newIsComparingRuns) =>
                  updateSearchFacets({ isComparingRuns: newIsComparingRuns })
                }
              />
            )}
            <div css={styles.searchBox}>
              <Input
                value={searchFilterValue}
                prefix={<Search1Icon css={styles.searchBarIcon} />}
                onKeyDown={triggerSearch}
                onChange={(e) => setSearchFilterValue(e.target.value)}
                placeholder={SEARCH_BOX_PLACEHOLDER}
                data-test-id='search-box'
                suffix={
                  <div css={styles.searchInputSuffix}>
                    {searchFilterValue && (
                      <Button onClick={clearFacetsState} type='link' data-test-id='clear-button'>
                        <CloseIcon />
                      </Button>
                    )}
                    <Tooltip
                      title={
                        <div className='search-input-tooltip-content'>
                          <FormattedMessage
                            defaultMessage='Search runs using a simplified version of the SQL {whereBold} clause.'
                            description='Tooltip string to explain how to search runs from the experiments table'
                            values={{ whereBold: <b>WHERE</b> }}
                          />{' '}
                          <FormattedMessage
                            defaultMessage='<link>Learn more</link>'
                            description='Learn more tooltip link to learn more on how to search in an experiments run table'
                            values={{
                              link: (chunks: any) => (
                                <a
                                  href={ExperimentSearchSyntaxDocUrl}
                                  target='_blank'
                                  rel='noopener noreferrer'
                                >
                                  {chunks}
                                </a>
                              ),
                            }}
                          />
                          <br />
                          <FormattedMessage
                            defaultMessage='Examples:'
                            description='Text header for examples of mlflow search syntax'
                          />
                          <br />
                          {'• metrics.rmse >= 0.8'}
                          <br />
                          {'• metrics.`f1 score` < 1'}
                          <br />
                          {"• params.model = 'tree'"}
                          <br />
                          {"• attributes.run_name = 'my run'"}
                          <br />
                          {"• tags.`mlflow.user` = 'myUser'"}
                          <br />
                          {"• metric.f1_score > 0.9 AND params.model = 'tree'"}
                          <br />
                          {"• attributes.run_id IN ('a1b2c3d4', 'e5f6g7h8')"}
                          <br />
                          {"• tags.model_class LIKE 'sklearn.linear_model%'"}
                        </div>
                      }
                      placement='right'
                      dangerouslySetAntdProps={{ overlayInnerStyle: { width: '150%' } }}
                    >
                      <InfoBorderIcon css={styles.searchBarIcon} />
                    </Tooltip>
                  </div>
                }
              />
            </div>
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
              onChange={sortKeyChanged}
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

            {!isComparingRuns && (
              <ExperimentViewRunsColumnSelector
                columnSelectorVisible={viewState.columnSelectorVisible}
                onChangeColumnSelectorVisible={changeColumnSelectorVisible}
                runsData={runsData}
              />
            )}
          </div>
          <div css={styles.groupSeparator} />
          <div css={styles.controlBar}>
            <Dropdown
              trigger={['click']}
              placement='bottomRight'
              overlay={
                <Menu>
                  <Menu.Item className='csv-button' onClick={onDownloadCsv}>
                    <DownloadIcon />{' '}
                    <FormattedMessage
                      defaultMessage='Download CSV'
                      description='String for the download csv button to download experiments offline in a CSV format'
                    />
                  </Menu.Item>
                </Menu>
              }
            >
              <Button type='tertiary' icon={<OverflowIcon />} />
            </Dropdown>
            <ExperimentViewRefreshButton />
          </div>
        </div>
      </>
    );
  },
);

const styles = {
  groupBar: { display: 'grid', gridTemplateColumns: 'auto 1fr auto' },
  controlBar: (theme: Theme) => ({
    display: 'flex',
    gap: theme.spacing.sm,
    alignItems: 'center',
    flexWrap: 'wrap' as const,
  }),
  searchBarIcon: (theme: Theme) => ({
    svg: { width: 16, height: 16, color: theme.colors.textSecondary },
  }),
  groupSeparator: (theme: Theme) => ({ minWidth: theme.spacing.sm }),
  searchInputSuffix: { display: 'flex', gap: 4, alignItems: 'center' },
  columnSwitch: { margin: '5px' },
  searchBox: (theme: Theme) => ({ display: 'flex', gap: theme.spacing.sm, width: 560 }),
  sortSelectControl: { minWidth: 140, maxWidth: 360 },

  lifecycleFilters: (theme: Theme) => ({
    display: 'flex',
    gap: 8,
    alignItems: 'center',
    marginTop: theme.spacing.sm,
    marginBottom: theme.spacing.sm,
    marginLeft: theme.spacing.lg * 2,
  }),
  sortSelectDropdown: { minWidth: 360 },
  sortMenuArrowWrapper: { svg: { width: 12, height: 12 } },
};
