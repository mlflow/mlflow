import { Button, DownloadIcon, Dropdown, Menu, OverflowIcon } from '@databricks/design-system';
import { Theme } from '@emotion/react';
import React, { useCallback } from 'react';
import { FormattedMessage } from 'react-intl';
import { ErrorWrapper } from '../../../../../common/utils/ErrorWrapper';
import { shouldUseNextRunsComparisonUI } from '../../../../../common/utils/FeatureUtils';
import { COLUMN_SORT_BY_ASC, SORT_DELIMITER_SYMBOL } from '../../../../constants';
import { UpdateExperimentSearchFacetsFn, UpdateExperimentViewStateFn } from '../../../../types';
import { ExperimentRunSortOption } from '../../hooks/useRunSortOptions';
import { SearchExperimentRunsFacetsState } from '../../models/SearchExperimentRunsFacetsState';
import { SearchExperimentRunsViewState } from '../../models/SearchExperimentRunsViewState';
import { TAGS_TO_COLUMNS_MAP } from '../../utils/experimentPage.column-utils';
import { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { ExperimentViewRefreshButton } from './ExperimentViewRefreshButton';
import { ExperimentViewRunsColumnSelector } from './ExperimentViewRunsColumnSelector';
import { ExperimentViewRunsModeSwitch } from './ExperimentViewRunsModeSwitch';
import { ExperimentViewRunsSortSelector } from './ExperimentViewRunsSortSelector';
import { RunsSearchAutoComplete } from './RunsSearchAutoComplete';

export type ExperimentViewRunsControlsFiltersProps = {
  searchFacetsState: SearchExperimentRunsFacetsState;
  updateSearchFacets: UpdateExperimentSearchFacetsFn;
  viewState: SearchExperimentRunsViewState;
  updateViewState: UpdateExperimentViewStateFn;
  sortOptions: ExperimentRunSortOption[];
  runsData: ExperimentRunsSelectorResult;
  onDownloadCsv: () => void;
  requestError: ErrorWrapper | null;
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
    requestError,
  }: ExperimentViewRunsControlsFiltersProps) => {
    const { isComparingRuns } = searchFacetsState;

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

    // Shows or hides the column selector
    const changeColumnSelectorVisible = useCallback(
      (value: boolean) => updateViewState({ columnSelectorVisible: value }),
      [updateViewState],
    );

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
            <RunsSearchAutoComplete
              runsData={runsData}
              searchFacetsState={searchFacetsState}
              updateSearchFacets={updateSearchFacets}
              requestError={requestError}
            />
            <ExperimentViewRunsSortSelector
              onSortKeyChanged={sortKeyChanged}
              searchFacetsState={searchFacetsState}
              sortOptions={sortOptions}
            />
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
  groupSeparator: (theme: Theme) => ({ minWidth: theme.spacing.sm }),
  columnSwitch: { margin: '5px' },
  searchBox: (theme: Theme) => ({ display: 'flex', gap: theme.spacing.sm, width: 560 }),
  lifecycleFilters: (theme: Theme) => ({
    display: 'flex',
    gap: 8,
    alignItems: 'center',
    marginTop: theme.spacing.sm,
    marginBottom: theme.spacing.sm,
    marginLeft: theme.spacing.lg * 2,
  }),
};
