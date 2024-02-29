import { Button, SyncIcon, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import React, { useCallback, useEffect, useState } from 'react';
// TODO: de-antd-ify Badge as soon as it appears in the design system
import { Theme } from '@emotion/react';
import { Badge } from 'antd';
import { FormattedMessage } from 'react-intl';
import { connect } from 'react-redux';
import { MAX_DETECT_NEW_RUNS_RESULTS, POLL_INTERVAL } from '../../../../constants';
import { ExperimentStoreEntities } from '../../../../types';
import { useExperimentIds } from '../../hooks/useExperimentIds';
import { useFetchExperimentRuns } from '../../hooks/useFetchExperimentRuns';
import { shouldEnableShareExperimentViewByTags } from '../../../../../common/utils/FeatureUtils';
import { searchRunsPayload as searchRunsPayloadDirect } from '../../../../actions';

export interface ExperimentViewRefreshButtonProps {
  runInfos: ExperimentStoreEntities['runInfosByUuid'];
  refreshRuns?: () => void;
}

// Local hook that returns proper variant of refresh runs function based on the feature flag.
const useGetRefreshFn = (props: ExperimentViewRefreshButtonProps) => {
  const usingNewViewStateModel = shouldEnableShareExperimentViewByTags();
  const { refreshRuns: refreshRunsFromProps } = props;
  if (usingNewViewStateModel && refreshRunsFromProps) {
    return refreshRunsFromProps;
  }
  // We can disable this eslint rule because condition uses a stable feature flag evaluation
  /* eslint-disable react-hooks/rules-of-hooks */
  const { updateSearchFacets } = useFetchExperimentRuns();
  return useCallback(
    () =>
      updateSearchFacets(
        {},
        {
          forceRefresh: true,
          preservePristine: true,
        },
      ),
    [updateSearchFacets],
  );
};

const useGetSearchRunsAction = () => {
  const usingNewViewStateModel = shouldEnableShareExperimentViewByTags();

  const {
    actions: { searchRunsPayload: searchRunsPayloadFromContext },
  } = useFetchExperimentRuns();

  if (usingNewViewStateModel) {
    return searchRunsPayloadDirect;
  }

  return searchRunsPayloadFromContext;
};

/**
 * A component that displays "refresh runs" button with the relevant number
 * of the new runs and handles the refresh action.
 */
export const ExperimentViewRefreshButtonImpl = React.memo(
  (props: React.PropsWithChildren<ExperimentViewRefreshButtonProps>) => {
    const { runInfos } = props;
    const { theme } = useDesignSystemTheme();

    const searchRunsPayload = useGetSearchRunsAction();
    const refreshRuns = useGetRefreshFn(props);

    const experimentIds = useExperimentIds();

    // Keeps the time of the last runs fetch
    const [lastFetchTime, setLastFetchTime] = useState(0);

    // Keeps the number of available new runs
    const [newRunsCount, setNewRunsCount] = useState(0);

    // We're resetting number of new runs and the fetch date
    // every time when the runs payload has changed
    useEffect(() => {
      setNewRunsCount(0);
      setLastFetchTime(() => Date.now());
    }, [runInfos]);

    useEffect(
      () => {
        if (!lastFetchTime) {
          return undefined;
        }
        const interval = setInterval(() => {
          // Let's query for new runs that have started after a certain time
          const searchPayloadData: any = {
            experimentIds,
            filter: `attributes.start_time > ${lastFetchTime}`,
            // We're not interested in more than 26 new runs
            maxResults: MAX_DETECT_NEW_RUNS_RESULTS,
          };
          searchRunsPayload(searchPayloadData).then((result) => {
            const newRuns = result.runs?.length || 0;
            setNewRunsCount(newRuns);
          });
        }, POLL_INTERVAL);
        return () => clearInterval(interval);
      },
      // We're resetting the interval each time the reference time or experiment IDs have changed
      [lastFetchTime, searchRunsPayload, experimentIds],
    );

    return (
      <Badge
        count={newRunsCount}
        offset={[-5, 5]}
        css={styles.pill(theme)}
        overflowCount={MAX_DETECT_NEW_RUNS_RESULTS - 1}
      >
        <Tooltip
          title={
            <FormattedMessage
              defaultMessage="Refresh"
              description="refresh button text to refresh the experiment runs"
            />
          }
          useAsLabel
        >
          <Button
            componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrefreshbutton.tsx_123"
            onClick={refreshRuns}
            data-testid="runs-refresh-button"
            icon={<SyncIcon />}
          />
        </Tooltip>
      </Badge>
    );
  },
);

const styles = {
  pill: (theme: Theme) => ({ sup: { backgroundColor: theme.colors.lime, zIndex: 1 } }),
};

/**
 * The only thing that we're interested in the store is the current set of runInfos.
 * We're going to monitor it so we will know when new runs are fetched.
 */
const mapStateToProps = (state: { entities: ExperimentStoreEntities }) => {
  return { runInfos: state.entities.runInfosByUuid };
};

export const ExperimentViewRefreshButton = connect(mapStateToProps, undefined, undefined, {
  // We're interested only in "entities" sub-tree so we won't
  // re-render on other state changes (e.g. API request IDs)
  areStatesEqual: (nextState, prevState) => nextState.entities.runInfosByUuid === prevState.entities.runInfosByUuid,
})(ExperimentViewRefreshButtonImpl);
