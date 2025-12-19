import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { listImagesApi } from '@mlflow/mlflow/src/experiment-tracking/actions';
import { EXPERIMENT_RUNS_IMAGE_AUTO_REFRESH_INTERVAL } from '@mlflow/mlflow/src/experiment-tracking/constants';
import { useEffect, useRef } from 'react';
import { useDispatch } from 'react-redux';
import type { ThunkDispatch } from '@mlflow/mlflow/src/redux-types';
import { NUM_RUNS_TO_SUPPORT_FOR_LOG_IMAGE } from '@mlflow/mlflow/src/experiment-tracking/constants';

export const usePopulateImagesByRunUuid = ({
  runUuids,
  runUuidsIsActive,
  autoRefreshEnabled,
  enabled,
}: {
  runUuids: string[];
  runUuidsIsActive: boolean[];
  autoRefreshEnabled?: boolean;
  enabled?: boolean;
}) => {
  // Retrieve image keys for each run. This should only re-render when the runUuids change.
  // This populates the imagesByRunUuid with imageKeys, which will be used elsewhere to fetch metadata.
  const dispatch = useDispatch<ThunkDispatch>();

  /**
   * The criteria to populate images for a run is
   * 1. The run is not hidden
   * 2. The run includes the mlflow.loggedImage tag
   * 3. The run's image is not already populated
   */
  // We need to use a serialized version of runUuids to avoid re-triggering the effect when using an array.
  const runUuidsSerialized = runUuids.slice(0, NUM_RUNS_TO_SUPPORT_FOR_LOG_IMAGE).join(',');
  useEffect(() => {
    // Regular single fetch event with no auto-refresh capabilities. Used if auto-refresh is disabled.
    if (!enabled || autoRefreshEnabled) {
      return;
    }
    runUuidsSerialized.split(',').forEach((runUuid) => {
      if (runUuid) {
        dispatch(listImagesApi(runUuid));
      }
    });
  }, [runUuidsSerialized, dispatch, autoRefreshEnabled, enabled]);

  const refreshTimeoutRef = useRef<number | undefined>(undefined);
  const autoRefreshEnabledRef = useRef(autoRefreshEnabled && enabled);
  autoRefreshEnabledRef.current = autoRefreshEnabled;

  const runUuidsIsActiveSerialized = runUuidsIsActive.slice(0, NUM_RUNS_TO_SUPPORT_FOR_LOG_IMAGE).join(',');
  // A fetch effect with auto-refresh capabilities. Used only if auto-refresh is enabled.
  useEffect(() => {
    let hookUnmounted = false;

    if (!enabled || !autoRefreshEnabled) {
      return;
    }

    const fetchImagesFn = async (autoRefresh: boolean) => {
      const runUuids = runUuidsSerialized.split(',');

      const activeRunUuids = runUuidsIsActiveSerialized.split(',');
      // If auto-refresh is enabled, only fetch images for runs that are currently active
      const filteredRunUuids = autoRefresh ? runUuids.filter((_, index) => activeRunUuids[index] === 'true') : runUuids;

      filteredRunUuids.forEach((runUuid) => {
        if (runUuid) {
          dispatch(listImagesApi(runUuid, autoRefresh));
        }
      });
    };

    const scheduleRefresh = async () => {
      // Initial check to confirm that auto-refresh is still enabled and the hook is still mounted
      if (!autoRefreshEnabledRef.current || hookUnmounted) {
        return;
      }
      try {
        await fetchImagesFn(true);
      } catch (e) {
        // In case of error during auto-refresh, log the error but do break the auto-refresh loop
        Utils.logErrorAndNotifyUser(e);
      }
      clearTimeout(refreshTimeoutRef.current);

      // After loading the data, schedule the next refresh if the hook is still enabled and mounted
      if (!autoRefreshEnabledRef.current || hookUnmounted) {
        return;
      }

      refreshTimeoutRef.current = window.setTimeout(scheduleRefresh, EXPERIMENT_RUNS_IMAGE_AUTO_REFRESH_INTERVAL);
    };

    fetchImagesFn(false).then(scheduleRefresh);

    return () => {
      fetchImagesFn(true);
      // Mark the hook as unmounted to prevent scheduling new auto-refreshes with current data
      hookUnmounted = true;
      // Clear the timeout
      clearTimeout(refreshTimeoutRef.current);
    };
  }, [dispatch, runUuidsSerialized, runUuidsIsActiveSerialized, autoRefreshEnabled, enabled]);
};
