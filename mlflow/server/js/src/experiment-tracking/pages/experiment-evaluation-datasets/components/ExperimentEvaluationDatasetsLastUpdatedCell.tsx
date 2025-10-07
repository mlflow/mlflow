import { Tooltip } from '@databricks/design-system';
import { timeSinceStr } from '@mlflow/mlflow/src/shared/web-shared/genai-traces-table/utils/DisplayUtils';
import { Row } from '@tanstack/react-table';
import { EvaluationDataset } from '../types';

export const LastUpdatedCell = ({ row }: { row: Row<EvaluationDataset> }) => {
  return row.original.last_update_time ? (
    <Tooltip
      content={new Date(row.original.last_update_time).toLocaleString()}
      componentId="mlflow.eval-datasets.last-updated-cell-tooltip"
    >
      <span>{timeSinceStr(row.original.last_update_time)}</span>
    </Tooltip>
  ) : (
    <span>-</span>
  );
};
