import { useCallback, useMemo, useState } from 'react';

import { getLocalStorageItem, setLocalStorageItem } from '@databricks/web-shared/hooks';

import type { NavigateOptions } from '@mlflow/mlflow/src/common/utils/RoutingUtils';

export enum EvalRunsDatePreset {
  ALL = 'ALL',
  LAST_HOUR = 'LAST_HOUR',
  LAST_24_HOURS = 'LAST_24_HOURS',
  LAST_7_DAYS = 'LAST_7_DAYS',
  LAST_30_DAYS = 'LAST_30_DAYS',
  LAST_YEAR = 'LAST_YEAR',
}

export const START_TIME_LABEL_PARAM = 'startTimeLabel';
const START_TIME_PARAM = 'startTime';
const END_TIME_PARAM = 'endTime';
const DATE_PRESET_STORAGE_VERSION = 1;
const DATE_PRESET_STORAGE_KEY_PREFIX = 'mlflow.eval-runs.date-preset';

const getDatePresetStorageKey = (experimentId: string) => `${DATE_PRESET_STORAGE_KEY_PREFIX}.${experimentId}`;

const readSavedDatePreset = (experimentId: string) =>
  getLocalStorageItem<EvalRunsDatePreset | undefined>(
    getDatePresetStorageKey(experimentId),
    DATE_PRESET_STORAGE_VERSION,
    true,
    undefined,
  );

const isEvalRunsDatePreset = (value: string): value is EvalRunsDatePreset =>
  Object.values(EvalRunsDatePreset).some((preset) => preset === value);

const toValidDatePreset = (value: string | null | undefined): EvalRunsDatePreset | undefined =>
  value && isEvalRunsDatePreset(value) ? value : undefined;

export const usePersistedDatePreset = ({
  enabled,
  experimentId,
  fallbackDatePreset,
  searchParams,
  setSearchParams,
}: {
  enabled: boolean;
  experimentId: string;
  fallbackDatePreset: EvalRunsDatePreset;
  searchParams: URLSearchParams;
  /** Shared with the other filter params so updates made in one handler compose. */
  setSearchParams: (update: (params: URLSearchParams) => URLSearchParams, navigateOptions?: NavigateOptions) => void;
}) => {
  const urlLabel = enabled ? searchParams.get(START_TIME_LABEL_PARAM) : null;
  const savedDatePreset = useMemo(
    () => (enabled ? readSavedDatePreset(experimentId) : undefined),
    [enabled, experimentId],
  );
  const [currentSessionSelection, setCurrentSessionSelection] = useState<
    { experimentId: string; datePreset: EvalRunsDatePreset } | undefined
  >();

  const currentExperimentSelection =
    currentSessionSelection?.experimentId === experimentId ? currentSessionSelection.datePreset : savedDatePreset;
  const datePreset = enabled
    ? (toValidDatePreset(urlLabel) ?? toValidDatePreset(currentExperimentSelection) ?? fallbackDatePreset)
    : fallbackDatePreset;

  const setDatePreset = useCallback(
    (nextDatePreset: EvalRunsDatePreset) => {
      if (!enabled) {
        return;
      }
      setLocalStorageItem(getDatePresetStorageKey(experimentId), DATE_PRESET_STORAGE_VERSION, true, nextDatePreset);
      setCurrentSessionSelection({ experimentId, datePreset: nextDatePreset });
      setSearchParams((params) => {
        params.set(START_TIME_LABEL_PARAM, nextDatePreset);
        params.delete(START_TIME_PARAM);
        params.delete(END_TIME_PARAM);
        return params;
      });
    },
    [enabled, experimentId, setSearchParams],
  );

  return [datePreset, setDatePreset] as const;
};
