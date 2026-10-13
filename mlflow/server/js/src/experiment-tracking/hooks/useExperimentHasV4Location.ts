import { useMemo } from 'react';
import { shouldUseTracesV4API } from '@databricks/web-shared/genai-traces-table';
import { MLFLOW_EXPERIMENT_TRACE_STORAGE_UC_SCHEMA_TAG } from '../constants';

/**
 * Derives whether the experiment uses a V4 trace location (UC schema / table prefix)
 * from its tags. Returns true when the destination path tag is present and the V4
 * traces API is enabled.
 *
 * Use this in components that don't have access to SqlWarehouseContext (e.g. the
 * global sidebar). Components inside the context should read `hasV4Location` from
 * there instead.
 */
export const useExperimentHasV4Location = (tags?: { key?: string | null; value?: string | null }[] | null) => {
  return useMemo(() => {
    const destinationPath = tags?.find((tag) => tag.key === MLFLOW_EXPERIMENT_TRACE_STORAGE_UC_SCHEMA_TAG)?.value;
    return Boolean(destinationPath && shouldUseTracesV4API());
  }, [tags]);
};
