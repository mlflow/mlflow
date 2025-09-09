import type { ExperimentEntity, RunInfoEntity } from '../../../types';
import { RunsChartsLineChartXAxisType } from '../../runs-charts/components/RunsCharts.common';
import type { ExperimentPageUIState } from '../models/ExperimentPageUIState';
import { RUNS_VISIBILITY_MODE } from '../models/ExperimentPageUIState';
import type { ExperimentRunsSelectorResult } from './experimentRuns.selector';

/**
 * Create a base UI state that matches the structure required by the initializer.
 */
export const createBaseUIState = (): ExperimentPageUIState => ({
  selectedColumns: [],
  runsExpanded: {},
  runsPinned: [],
  runsHidden: [],
  runsHiddenMode: RUNS_VISIBILITY_MODE.FIRST_10_RUNS,
  runsVisibilityMap: {},
  viewMaximized: false,
  runListHidden: false,
  groupBy: null,
  groupsExpanded: {},
  useGroupedValuesInCharts: true,
  hideEmptyCharts: true,
  globalLineChartConfig: {
    xAxisKey: RunsChartsLineChartXAxisType.STEP,
    lineSmoothness: 0,
    selectedXAxisMetricKey: '',
  },
  isAccordionReordered: false,
  autoRefreshEnabled: false,
});

/**
 * Create a base runs data object.
 */
export const createBaseRunsData = (): ExperimentRunsSelectorResult => ({
  paramKeyList: [],
  metricKeyList: [],
  datasetsList: [],
  experimentTags: {},
  metricsList: [],
  modelVersionsByRunUuid: {},
  paramsList: [],
  runInfos: [],
  runUuidsMatchingFilter: [],
  tagsList: [],
});

/**
 * Helper to create a tag.
 */
export function makeTag(key: string, value: string) {
  return { key, value };
}

/**
 * Create a base ExperimentEntity object.
 */
export const createBaseExperimentEntity = (): ExperimentEntity => ({
  allowedActions: [],
  artifactLocation: '',
  creationTime: 0,
  experimentId: 'experiment_1',
  lastUpdateTime: 0,
  lifecycleStage: 'active',
  name: 'AutoML Experiment',
  tags: [],
});

/**
 * Create a base RunInfoEntity object.
 */
export const createBaseRunsInfoEntity = (): RunInfoEntity => ({
  artifactUri: '',
  endTime: 0,
  experimentId: 'experiment_1',
  lifecycleStage: 'active',
  runName: 'run_1',
  runUuid: 'run_1',
  startTime: 0,
  status: 'FINISHED',
});
