import { Theme } from '@emotion/react';
import React, { useCallback } from 'react';
import Utils from '../../../../../common/utils/Utils';
import { UpdateExperimentSearchFacetsFn, UpdateExperimentViewStateFn } from '../../../../types';
import { useRunSortOptions } from '../../hooks/useRunSortOptions';
import { SearchExperimentRunsFacetsState } from '../../models/SearchExperimentRunsFacetsState';
import { SearchExperimentRunsViewState } from '../../models/SearchExperimentRunsViewState';
import { downloadRunsCsv } from '../../utils/experimentPage.common-utils';
import { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { ExperimentViewRunsControlsActions } from './ExperimentViewRunsControlsActions';
import { ExperimentViewRunsControlsFilters } from './ExperimentViewRunsControlsFilters';
import { ErrorWrapper } from '../../../../../common/utils/ErrorWrapper';

type ExperimentViewRunsControlsProps = {
  viewState: SearchExperimentRunsViewState;
  updateViewState: UpdateExperimentViewStateFn;

  searchFacetsState: SearchExperimentRunsFacetsState;
  updateSearchFacets: UpdateExperimentSearchFacetsFn;

  runsData: ExperimentRunsSelectorResult;

  requestError: ErrorWrapper | null;
};

/**
 * This component houses all controls related to searching runs: sort controls,
 * filters and run related actions (delete, restore, download CSV).
 */
export const ExperimentViewRunsControls = React.memo(
  ({
    runsData,
    viewState,
    updateViewState,
    updateSearchFacets,
    searchFacetsState,
    requestError,
  }: ExperimentViewRunsControlsProps) => {
    const { paramKeyList, metricKeyList, tagsList } = runsData;

    const filteredParamKeys = paramKeyList;
    const filteredMetricKeys = metricKeyList;
    const filteredTagKeys = Utils.getVisibleTagKeyList(tagsList);

    const onDownloadCsv = useCallback(
      () => downloadRunsCsv(runsData, filteredTagKeys, filteredParamKeys, filteredMetricKeys),
      [filteredMetricKeys, filteredParamKeys, filteredTagKeys, runsData],
    );

    const sortOptions = useRunSortOptions(filteredMetricKeys, filteredParamKeys);

    return (
      <div css={styles.wrapper}>
        <ExperimentViewRunsControlsFilters
          onDownloadCsv={onDownloadCsv}
          updateSearchFacets={updateSearchFacets}
          searchFacetsState={searchFacetsState}
          viewState={viewState}
          updateViewState={updateViewState}
          sortOptions={sortOptions}
          runsData={runsData}
          requestError={requestError}
        />
        <ExperimentViewRunsControlsActions
          runsData={runsData}
          updateSearchFacets={updateSearchFacets}
          searchFacetsState={searchFacetsState}
          viewState={viewState}
        />
      </div>
    );
  },
);

const styles = {
  wrapper: (theme: Theme) => ({
    display: 'flex',
    gap: theme.spacing.sm,
    flexDirection: 'column' as const,
    marginTop: theme.spacing.md,
    marginBottom: theme.spacing.sm,
  }),
};
