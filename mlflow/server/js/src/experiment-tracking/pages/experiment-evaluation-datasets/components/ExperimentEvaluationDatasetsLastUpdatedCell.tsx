import { Tooltip } from '@databricks/design-system';
import type { Row } from '@tanstack/react-table';
import type { EvaluationDataset } from '../types';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';

export const LastUpdatedCell = ({ row }: { row: Row<EvaluationDataset> }) => {
  return row.original.last_update_time ? (
    <Tooltip
      content={new Date(row.original.last_update_time).toLocaleString()}
      componentId="mlflow.eval-datasets.last-updated-cell-tooltip"
    >
      <span>{Utils.timeSinceStr(row.original.last_update_time)}</span>
    </Tooltip>
  ) : (
    <span>-</span>
  );
};
