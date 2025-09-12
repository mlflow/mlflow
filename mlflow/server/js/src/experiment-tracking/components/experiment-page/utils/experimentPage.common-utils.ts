import type { IntlShape } from 'react-intl';
import { saveAs } from 'file-saver';
import type { ExperimentEntity } from '../../../types';
import type { ExperimentRunsSelectorResult } from './experimentRuns.selector';
import { chartDataToCsv, chartMetricHistoryToCsv, runInfosToCsv } from '../../../utils/CsvUtils';
import type { RunsChartsRunData } from '../../runs-charts/components/RunsCharts.common';

export const EXPERIMENT_FIELD_PREFIX_PARAM = '$$$param$$$';
export const EXPERIMENT_FIELD_PREFIX_METRIC = '$$$metric$$$';
export const EXPERIMENT_FIELD_PREFIX_TAG = '$$$tag$$$';
export const EXPERIMENT_PARENT_ID_TAG = 'mlflow.parentRunId';
export const EXPERIMENT_LOG_MODEL_HISTORY_TAG = 'mlflow.log-model.history';
export const EXPERIMENT_RUNS_TABLE_ROW_HEIGHT = 32;

const MLFLOW_NOTEBOOK_TYPE = 'NOTEBOOK';
const MLFLOW_EXPERIMENT_TYPE = 'MLFLOW_EXPERIMENT';

const EXPERIMENT_TYPE_TAG = 'mlflow.experimentType';
const EXPERIMENT_SOURCE_TYPE_TAG = 'mlflow.experiment.sourceType';
const EXPERIMENT_SOURCE_ID_TAG = 'mlflow.experiment.sourceId';

enum ExperimentSourceType {
  REPO_NOTEBOOK = 'REPO_NOTEBOOK',
}

/**
 * Function that gets the experiment type for a given experiment object
 */
export const getExperimentType = (experiment: ExperimentEntity) => {
  const experimentType = experiment.tags.find((tag) => tag.key === EXPERIMENT_TYPE_TAG);
  if (experimentType) {
    return experimentType.value;
  }
  return null;
};

const hasExperimentType = (experiment: ExperimentEntity, type: string) => getExperimentType(experiment) === type;

/**
 * Function returns true if the experiment is of default ("MLFLOW_EXPERIMENT") type
 */
export const isExperimentTypeDefault = (experiment: ExperimentEntity) =>
  hasExperimentType(experiment, MLFLOW_EXPERIMENT_TYPE);

/**
 * Function returns true if the experiment is of notebook type
 */
export const isExperimentTypeNotebook = (experiment: ExperimentEntity) =>
  hasExperimentType(experiment, MLFLOW_NOTEBOOK_TYPE);

/**
 * Function that checks if experiment's allowed actions include
 * modification. TODO: fix typo in the const name.
 */
export const canModifyExperiment = (experiment: ExperimentEntity) =>
  (experiment.allowedActions || []).includes('MODIFIY_PERMISSION');

/**
 * Function that gets the experiment source ID for a given experiment object
 */
export const getExperimentSourceId = (experiment: ExperimentEntity) =>
  experiment.tags?.find((tag) => tag.key === EXPERIMENT_SOURCE_ID_TAG)?.value;

/**
 * Function that checks if experiment comes from repository notebook
 */
export const isRepoNotebookExperiment = (experiment: ExperimentEntity) =>
  experiment.tags?.find((tag) => tag.key === EXPERIMENT_SOURCE_TYPE_TAG)?.value === ExperimentSourceType.REPO_NOTEBOOK;

/**
 * Function used for downloading run data in CSV form.
 */
export const downloadRunsCsv = (
  runsData: ExperimentRunsSelectorResult,
  filteredTagKeys: string[],
  filteredParamKeys: string[],
  filteredMetricKeys: string[],
) => {
  const { runInfos, paramsList, metricsList, tagsList } = runsData;

  const csv = runInfosToCsv({
    runInfos,
    paramKeyList: filteredParamKeys,
    metricKeyList: filteredMetricKeys,
    tagKeyList: filteredTagKeys,
    paramsList,
    metricsList,
    tagsList,
  });
  const blob = new Blob([csv], { type: 'application/csv;charset=utf-8' });
  saveAs(blob, 'runs.csv');
};

/**
 * Function used for downloading metric history chart data in CSV form.
 */
export const downloadChartMetricHistoryCsv = (traces: RunsChartsRunData[], metricKeys: string[], title: string) => {
  const csv = chartMetricHistoryToCsv(traces, metricKeys);
  const blob = new Blob([csv], { type: 'application/csv;charset=utf-8' });
  saveAs(blob, `${title}.csv`);
};

/**
 * Function used for downloading latest chart data in CSV form.
 */
export const downloadChartDataCsv = (
  traces: RunsChartsRunData[],
  metricKeys: string[],
  paramKeys: string[],
  title: string,
) => {
  const csv = chartDataToCsv(traces, metricKeys, paramKeys);
  const blob = new Blob([csv], { type: 'application/csv;charset=utf-8' });
  saveAs(blob, `${title}.csv`);
};

/**
 * Function used for preparing values for "created" (start time) runs filter.
 */
export const getStartTimeColumnDisplayName = (intl: IntlShape) => ({
  LAST_HOUR: intl.formatMessage({
    defaultMessage: 'Last hour',
    description: 'Option for the start select dropdown to filter runs from the last hour',
  }),
  LAST_24_HOURS: intl.formatMessage({
    defaultMessage: 'Last 24 hours',
    description: 'Option for the start select dropdown to filter runs from the last 24 hours',
  }),
  LAST_7_DAYS: intl.formatMessage({
    defaultMessage: 'Last 7 days',
    description: 'Option for the start select dropdown to filter runs from the last 7 days',
  }),
  LAST_30_DAYS: intl.formatMessage({
    defaultMessage: 'Last 30 days',
    description: 'Option for the start select dropdown to filter runs from the last 30 days',
  }),
  LAST_YEAR: intl.formatMessage({
    defaultMessage: 'Last year',
    description: 'Option for the start select dropdown to filter runs since the last 1 year',
  }),
});

/**
 * Creates qualified entity name given a key type and name, wrapping in backticks
 * or quotes as needed and where appropriate
 */
export const getQualifiedEntityName = (keyType: string, keyName: string) => {
  let replace = '';
  if (
    keyName.includes('"') ||
    keyName.includes(' ') ||
    keyName.includes('.') ||
    keyName.includes('(') ||
    keyName.includes(')')
  ) {
    replace = '`';
  }
  if (keyName.includes('`')) {
    replace = '"';
  }
  return `${keyType}.${replace}${keyName}${replace}`;
};

/**
 * Creates canonical sort key name for metrics and params
 */
export const makeCanonicalSortKey = (keyType: string, keyName: string) => keyType + '.`' + keyName + '`';

export const isCanonicalSortKeyOfType = (canonicalKey: string, keyType: string) => canonicalKey.startsWith(keyType);

/**
 * Extracts param/metric/tag name from the canonical key
 */
export const extractCanonicalSortKey = (canonicalKey: string, keyType: string) =>
  canonicalKey.substring(keyType.length + 2).slice(0, -1);
