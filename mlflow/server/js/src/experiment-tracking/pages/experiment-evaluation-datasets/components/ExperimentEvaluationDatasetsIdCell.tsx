import React from 'react';
import { Tag, Tooltip } from '@databricks/design-system';
import type { Row } from '@tanstack/react-table';
import { useCopyController } from '@databricks/web-shared/copy';
import type { EvaluationDataset } from '../types';

export const DatasetIdCell = ({ row }: { row: Row<EvaluationDataset> }) => {
  const datasetId = row.original.dataset_id;
  const { copy, tooltipOpen, tooltipMessage, handleTooltipOpenChange } = useCopyController(datasetId, 'Click to copy');

  return (
    <Tooltip
      content={tooltipMessage}
      open={tooltipOpen}
      onOpenChange={handleTooltipOpenChange}
      componentId="mlflow.eval-datasets.dataset-id-tooltip"
    >
      <Tag
        css={{ width: 'fit-content', maxWidth: '100%', cursor: 'pointer' }}
        componentId="mlflow.eval-datasets.dataset-id"
        color="indigo"
        onClick={copy}
      >
        <span
          css={{
            display: 'block',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {datasetId}
        </span>
      </Tag>
    </Tooltip>
  );
};
