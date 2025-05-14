import { Button, SyncIcon, LegacyTooltip, useDesignSystemTheme } from '@databricks/design-system';
import React, { useEffect, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { connect } from 'react-redux';
import { MAX_DETECT_NEW_RUNS_RESULTS, POLL_INTERVAL } from '../../../../constants';
import { ExperimentStoreEntities } from '../../../../types';
import { useExperimentIds } from '../../hooks/useExperimentIds';
import { searchRunsPayload } from '../../../../actions';

export interface ExperimentViewRefreshButtonProps {
  runInfos: ExperimentStoreEntities['runInfosByUuid'];
  refreshRuns?: () => void;
}

/**
 * A component that displays "refresh runs" button with the relevant number
 * of the new runs and handles the refresh action.
 */
export const ExperimentViewRefreshButtonImpl = React.memo(
  (props: React.PropsWithChildren<ExperimentViewRefreshButtonProps>) => {
    const { runInfos } = props;
    const { theme } = useDesignSystemTheme();

    const { refreshRuns } = props;

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
      [lastFetchTime, experimentIds],
    );

    return (
      <div css={{ position: 'relative' }}>
        {/* Replace this bespoke Badge item with Dubois component if it ever becomes available. */}
        {newRunsCount > 0 && (
          <div
            title={
              MAX_DETECT_NEW_RUNS_RESULTS > newRunsCount ? `${newRunsCount}` : `${MAX_DETECT_NEW_RUNS_RESULTS - 1}+`
            }
            css={{
              position: 'absolute',
              top: 0,
              right: 0,
              transform: 'translate(50%, -50%)',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              width: newRunsCount > 9 ? 28 : 20, // Makes the badge wider when count is more than 2 digits
              height: 20,
              borderRadius: 10,
              border: `1px solid ${theme.colors.white}`,
              backgroundColor: theme.colors.lime, // Why lime?
              color: theme.colors.white,
              fontSize: 10,
              fontWeight: 'bold',
              userSelect: 'none',
              zIndex: 1,
            }}
          >
            {MAX_DETECT_NEW_RUNS_RESULTS > newRunsCount ? newRunsCount : `${MAX_DETECT_NEW_RUNS_RESULTS - 1}+`}
          </div>
        )}
        <LegacyTooltip
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
        </LegacyTooltip>
      </div>
    );
  },
);

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
