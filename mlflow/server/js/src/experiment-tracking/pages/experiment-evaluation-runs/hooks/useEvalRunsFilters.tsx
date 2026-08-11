import { useCallback, useMemo, useState } from 'react';

import {
  MLFLOW_RUN_TYPE_TAG,
  MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION,
  MLFLOW_RUN_TYPE_VALUE_TEST,
} from '../../../constants';

/**
 * Date presets mirror the Traces monitoring filters so the two surfaces agree on
 * vocabulary. Defaults to ALL: a default that hides runs the user never chose to
 * hide makes a month-old experiment look empty on first visit.
 */
export enum EvalRunsDatePreset {
  ALL = 'ALL',
  LAST_HOUR = 'LAST_HOUR',
  LAST_24_HOURS = 'LAST_24_HOURS',
  LAST_7_DAYS = 'LAST_7_DAYS',
  LAST_30_DAYS = 'LAST_30_DAYS',
  LAST_YEAR = 'LAST_YEAR',
}

const PRESET_DURATIONS_MS: Record<Exclude<EvalRunsDatePreset, EvalRunsDatePreset.ALL>, number> = {
  [EvalRunsDatePreset.LAST_HOUR]: 60 * 60 * 1000,
  [EvalRunsDatePreset.LAST_24_HOURS]: 24 * 60 * 60 * 1000,
  [EvalRunsDatePreset.LAST_7_DAYS]: 7 * 24 * 60 * 60 * 1000,
  [EvalRunsDatePreset.LAST_30_DAYS]: 30 * 24 * 60 * 60 * 1000,
  [EvalRunsDatePreset.LAST_YEAR]: 365 * 24 * 60 * 60 * 1000,
};

export enum EvalRunsTypeFilter {
  ALL = 'ALL',
  /** Hides `test` and `issue_detection` runs, leaving deliberate evaluations. */
  EVAL_ONLY = 'EVAL_ONLY',
  ISSUE_DETECTION = 'ISSUE_DETECTION',
  TEST = 'TEST',
}

export interface EvalRunsFiltersState {
  datePreset: EvalRunsDatePreset;
  typeFilter: EvalRunsTypeFilter;
}

const INITIAL_STATE: EvalRunsFiltersState = {
  datePreset: EvalRunsDatePreset.ALL,
  typeFilter: EvalRunsTypeFilter.ALL,
};

/** Escapes single quotes so a value can't break out of the quoted literal. */
const quote = (value: string) => `'${value.replace(/'/g, "''")}'`;

export const buildEvalRunsFilterString = (state: EvalRunsFiltersState, searchFilter: string, now = Date.now()) => {
  const clauses: string[] = [];

  const trimmedSearch = searchFilter.trim();
  if (trimmedSearch) {
    clauses.push(trimmedSearch);
  }

  if (state.datePreset !== EvalRunsDatePreset.ALL) {
    // NB: the timestamp must be unquoted - a quoted number is rejected by the parser.
    clauses.push(`attributes.start_time > ${now - PRESET_DURATIONS_MS[state.datePreset]}`);
  }

  switch (state.typeFilter) {
    case EvalRunsTypeFilter.EVAL_ONLY:
      clauses.push(`tags.\`${MLFLOW_RUN_TYPE_TAG}\` != ${quote(MLFLOW_RUN_TYPE_VALUE_TEST)}`);
      clauses.push(`tags.\`${MLFLOW_RUN_TYPE_TAG}\` != ${quote(MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION)}`);
      break;
    case EvalRunsTypeFilter.ISSUE_DETECTION:
      clauses.push(`tags.\`${MLFLOW_RUN_TYPE_TAG}\` = ${quote(MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION)}`);
      break;
    case EvalRunsTypeFilter.TEST:
      clauses.push(`tags.\`${MLFLOW_RUN_TYPE_TAG}\` = ${quote(MLFLOW_RUN_TYPE_VALUE_TEST)}`);
      break;
    default:
      break;
  }

  return clauses.join(' AND ');
};

export const useEvalRunsFilters = ({ searchFilter }: { searchFilter: string }) => {
  const [filters, setFilters] = useState<EvalRunsFiltersState>(INITIAL_STATE);

  const setDatePreset = useCallback(
    (datePreset: EvalRunsDatePreset) => setFilters((prev) => ({ ...prev, datePreset })),
    [],
  );
  const setTypeFilter = useCallback(
    (typeFilter: EvalRunsTypeFilter) => setFilters((prev) => ({ ...prev, typeFilter })),
    [],
  );
  const clearAll = useCallback(() => setFilters(INITIAL_STATE), []);

  const isAnyFilterActive =
    filters.datePreset !== EvalRunsDatePreset.ALL || filters.typeFilter !== EvalRunsTypeFilter.ALL;

  // Recomputing on every render would make the relative date boundary drift and
  // change the query key, refetching constantly. Pin it to the filter state.
  const filterString = useMemo(
    () => buildEvalRunsFilterString(filters, searchFilter),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [filters, searchFilter],
  );

  return {
    filters,
    filterString,
    isAnyFilterActive,
    setDatePreset,
    setTypeFilter,
    clearAll,
  };
};
