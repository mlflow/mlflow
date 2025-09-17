import { useCallback, useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import type { ReduxState, ThunkDispatch } from '../../../redux-types';
import { getExperimentApi, getRunApi } from '../../actions';
import { searchModelVersionsApi } from '../../../model-registry/actions';
import { pickBy } from 'lodash';
import Utils from '../../../common/utils/Utils';

/**
 * Hook fetching data for the run page: both run and experiment entities.
 * The initial fetch action is omitted if entities are already in the store.
 */
export const useRunDetailsPageDataLegacy = (
  runUuid: string,
  experimentId: string,
  enableWorkspaceModelsRegistryCall = true,
) => {
  const [runRequestId, setRunRequestId] = useState('');
  const [experimentRequestId, setExperimentRequestId] = useState('');
  const dispatch = useDispatch<ThunkDispatch>();

  // Get the necessary data from the store

  const { runInfo, tags, latestMetrics, experiment, params, datasets } = useSelector((state: ReduxState) => ({
    runInfo: state.entities.runInfosByUuid[runUuid],
    // Filter out tags, metrics, and params that are entirely whitespace
    tags: pickBy(state.entities.tagsByRunUuid[runUuid], (tag) => tag.key.trim().length > 0),
    latestMetrics: pickBy(state.entities.latestMetricsByRunUuid[runUuid], (metric) => metric.key.trim().length > 0),
    params: pickBy(state.entities.paramsByRunUuid[runUuid], (param) => param.key.trim().length > 0),
    experiment: state.entities.experimentsById[experimentId],
    datasets: state.entities.runDatasetsByUuid[runUuid],
  }));

  const fetchRun = useCallback(() => {
    const action = getRunApi(runUuid);
    setRunRequestId(action.meta.id);
    return dispatch(action);
  }, [dispatch, runUuid]);

  const fetchExperiment = useCallback(() => {
    const action = getExperimentApi(experimentId);
    setExperimentRequestId(action.meta.id);
    return dispatch(action);
  }, [dispatch, experimentId]);

  const fetchModelVersions = useCallback(() => {
    if (enableWorkspaceModelsRegistryCall) {
      dispatch(searchModelVersionsApi({ run_id: runUuid }));
    }
  }, [dispatch, runUuid, enableWorkspaceModelsRegistryCall]);

  // Do the initial run & experiment fetch only if it's not in the store already
  useEffect(() => {
    if (!runInfo) {
      fetchRun().catch((e) => Utils.logErrorAndNotifyUser(e));
    }
    fetchModelVersions();
  }, [runInfo, fetchRun, fetchModelVersions]);

  useEffect(() => {
    if (!experiment) {
      fetchExperiment().catch((e) => Utils.logErrorAndNotifyUser(e));
    }
  }, [experiment, fetchExperiment]);

  // Check the "apis" store for the requests status
  const { loading: runLoading, error: runFetchError } = useSelector((state: ReduxState) => ({
    loading: !runRequestId || Boolean(state.apis?.[runRequestId]?.active),
    error: state.apis?.[runRequestId]?.error,
  }));

  const { loading: experimentLoading, error: experimentFetchError } = useSelector((state: ReduxState) => ({
    loading: !runRequestId || Boolean(state.apis?.[experimentRequestId]?.active),
    error: state.apis?.[experimentRequestId]?.error,
  }));

  const loading = runLoading || experimentLoading;

  return {
    loading,
    data: {
      runInfo,
      tags,
      params,
      latestMetrics,
      experiment,
      datasets,
    },
    refetchRun: fetchRun,
    errors: { runFetchError, experimentFetchError },
  };
};
