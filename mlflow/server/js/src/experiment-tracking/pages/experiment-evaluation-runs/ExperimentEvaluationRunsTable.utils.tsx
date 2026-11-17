import type { RunEntity } from '../../types';
import type { ExperimentRunsSelectorResult } from '../../components/experiment-page/utils/experimentRuns.selector';
import type { KeyValueEntity } from '../../../common/types';
import { EvalRunsTableKeyedColumnPrefix } from './ExperimentEvaluationRunsTable.constants';

export const createEvalRunsTableKeyedColumnKey = (columnType: EvalRunsTableKeyedColumnPrefix, key: string): string =>
  [columnType, key].join('.');
export const parseEvalRunsTableKeyedColumnKey = (
  key: string,
): undefined | { columnType: EvalRunsTableKeyedColumnPrefix; key: string } => {
  const [columnType, ...rest] = key.split('.');
  if (
    !rest.length ||
    !Object.values(EvalRunsTableKeyedColumnPrefix).includes(columnType as EvalRunsTableKeyedColumnPrefix)
  ) {
    return undefined;
  }
  return {
    columnType: columnType as EvalRunsTableKeyedColumnPrefix,
    key: rest.join('.'),
  };
};

export const getEvalRunCellValueBasedOnColumn = (columnId: string, rowData: RunEntity): string | number | undefined => {
  const { columnType, key: rowDataKey } = parseEvalRunsTableKeyedColumnKey(columnId) ?? {};
  if (!rowDataKey) {
    return undefined;
  }
  switch (columnType) {
    case EvalRunsTableKeyedColumnPrefix.METRIC:
      return rowData.data?.metrics?.find((metric) => metric.key === rowDataKey)?.value ?? undefined;
    case EvalRunsTableKeyedColumnPrefix.PARAM:
      return rowData.data?.params?.find((param) => param.key === rowDataKey)?.value ?? undefined;
    case EvalRunsTableKeyedColumnPrefix.TAG:
      return rowData.data?.tags?.find((tag) => tag.key === rowDataKey)?.value ?? undefined;
    default:
      return undefined;
  }
};

// Mirror the legacy `ExperimentRunsSelectorResult` shape so the shared RunsSearchAutoComplete /
// filter tooling can be reused as-is. Used by the V2 eval-runs toolbar.
export const getRunTableMetadata = (runsData: RunEntity[]): ExperimentRunsSelectorResult => {
  const metricKeys = new Set<string>();
  const paramKeys = new Set<string>();
  const tags: Record<string, KeyValueEntity>[] = [];
  const runInfos = runsData.map((run) => run.info);
  const metricsList = runsData.map((run) => run.data.metrics ?? []);
  const paramsList = runsData.map((run) => run.data.params ?? []);
  const datasetsList = runsData.map((run) => run.inputs?.datasetInputs ?? []);

  runsData.forEach((run) => {
    run.data.metrics?.forEach((metric) => metricKeys.add(metric.key));
    run.data.params?.forEach((param) => paramKeys.add(param.key));
    const runTags: Record<string, KeyValueEntity> = {};
    run.data.tags?.forEach((tag) => {
      runTags[tag.key] = { key: tag.key, value: tag.value };
    });
    tags.push(runTags);
  });

  return {
    runInfos,
    metricsList,
    paramsList,
    metricKeyList: Array.from(metricKeys),
    paramKeyList: Array.from(paramKeys),
    tagsList: tags,
    datasetsList,
  } as ExperimentRunsSelectorResult;
};
