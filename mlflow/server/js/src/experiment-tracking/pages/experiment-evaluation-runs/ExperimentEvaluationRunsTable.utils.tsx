import type { RunEntity } from '../../types';
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
