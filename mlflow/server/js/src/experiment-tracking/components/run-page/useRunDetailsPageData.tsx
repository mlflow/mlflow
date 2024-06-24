import { useCallback, useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { ReduxState, ThunkDispatch } from '../../../redux-types';
import { getExperimentApi, getRunApi } from '../../actions';
import { searchModelVersionsApi } from '../../../model-registry/actions';

/**
 * Hook fetching data for the run page: both run and experiment entities.
 * The initial fetch action is omitted if entities are already in the store.
 */
export const useRunDetailsPageData = (runUuid: string, experimentId: string) => {
  const [runRequestId, setRunRequestId] = useState('');
  const [experimentRequestId, setExperimentRequestId] = useState('');
  const dispatch = useDispatch<ThunkDispatch>();

  // Get the necessary data from the store
  const { runInfo, tags, latestMetrics, experiment, params } = useSelector((state: ReduxState) => ({
    runInfo: state.entities.runInfosByUuid[runUuid],
    tags: state.entities.tagsByRunUuid[runUuid],
    latestMetrics: state.entities.latestMetricsByRunUuid[runUuid],
    params: state.entities.paramsByRunUuid[runUuid],
    experiment: state.entities.experimentsById[experimentId],
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
    dispatch(searchModelVersionsApi({ run_id: runUuid }));
  }, [dispatch, runUuid]);

  // Do the initial run & experiment fetch only if it's not in the store already
  useEffect(() => {
    if (!runInfo) {
      fetchRun();
    }
    fetchModelVersions();
  }, [runInfo, fetchRun, fetchModelVersions]);

  useEffect(() => {
    if (!experiment) {
      fetchExperiment();
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
    },
    refetchRun: fetchRun,
    errors: { runFetchError, experimentFetchError },
  };
};
