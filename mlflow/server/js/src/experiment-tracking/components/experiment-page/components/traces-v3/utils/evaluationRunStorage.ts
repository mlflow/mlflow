import LocalStorageUtils from '../../../../../../common/utils/LocalStorageUtils';

export type PersistedEvaluationRun = {
  jobId: string;
  /**
   * The created evaluation run id, once available.
   * This may not be present immediately after job creation.
   */
  runId?: string;
  total: number;
  startedAt: number;
  runName?: string;
};

const STORAGE_COMPONENT = 'TracesV3EvaluationRun';
const STORAGE_KEY = 'activeEvaluationRun';
const RUN_ID_PREFIX = 'evaluationRunByRunId:';
export const EVALUATION_RUN_STORAGE_EVENT = 'mlflow:traces-evaluation-run-storage-updated';

const getStorage = (experimentId: string) => LocalStorageUtils.getStoreForComponent(STORAGE_COMPONENT, experimentId);

export const loadPersistedEvaluationRun = (experimentId: string): PersistedEvaluationRun | null => {
  const storage = getStorage(experimentId);
  const storedValue = storage.getItem(STORAGE_KEY);
  if (!storedValue) {
    return null;
  }
  try {
    const parsed = JSON.parse(storedValue) as PersistedEvaluationRun;
    if (parsed && typeof parsed.jobId === 'string') {
      return parsed;
    }
  } catch (error) {
    return null;
  }
  return null;
};

export const persistEvaluationRun = (experimentId: string, payload: PersistedEvaluationRun) => {
  const storage = getStorage(experimentId);
  storage.setItem(STORAGE_KEY, JSON.stringify(payload));
  // Also persist a stable mapping for deep-links: selectedRunUuid (runId) -> job info.
  // This enables the evaluation run page to show progress even when the "active" run differs.
  if (payload.runId) {
    storage.setItem(`${RUN_ID_PREFIX}${payload.runId}`, JSON.stringify(payload));
  }
  window.dispatchEvent(new CustomEvent(EVALUATION_RUN_STORAGE_EVENT, { detail: { experimentId } }));
};

export const loadPersistedEvaluationRunByRunId = (
  experimentId: string,
  runId: string,
): PersistedEvaluationRun | null => {
  const storage = getStorage(experimentId);
  const storedValue = storage.getItem(`${RUN_ID_PREFIX}${runId}`);
  if (!storedValue) {
    return null;
  }
  try {
    const parsed = JSON.parse(storedValue) as PersistedEvaluationRun;
    if (parsed && typeof parsed.jobId === 'string') {
      return parsed;
    }
  } catch (error) {
    return null;
  }
  return null;
};

export const clearPersistedEvaluationRun = (experimentId: string) => {
  const storage = getStorage(experimentId);
  window.localStorage.removeItem(storage.withScopePrefix(STORAGE_KEY));
  window.dispatchEvent(new CustomEvent(EVALUATION_RUN_STORAGE_EVENT, { detail: { experimentId } }));
};

export const clearPersistedEvaluationRunByRunId = (experimentId: string, runId: string) => {
  const storage = getStorage(experimentId);
  window.localStorage.removeItem(storage.withScopePrefix(`${RUN_ID_PREFIX}${runId}`));
  window.dispatchEvent(new CustomEvent(EVALUATION_RUN_STORAGE_EVENT, { detail: { experimentId } }));
};

export const clearAllPersistedEvaluationRuns = (experimentId: string) => {
  const storage = getStorage(experimentId);
  const scopedRunIdPrefix = storage.withScopePrefix(RUN_ID_PREFIX);
  const keys = Object.keys(window.localStorage);
  for (const key of keys) {
    if (key === storage.withScopePrefix(STORAGE_KEY) || key.startsWith(scopedRunIdPrefix)) {
      window.localStorage.removeItem(key);
    }
  }
  window.dispatchEvent(new CustomEvent(EVALUATION_RUN_STORAGE_EVENT, { detail: { experimentId } }));
};
