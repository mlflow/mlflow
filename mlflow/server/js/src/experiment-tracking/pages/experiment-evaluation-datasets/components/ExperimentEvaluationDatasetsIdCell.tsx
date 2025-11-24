import React, { useState } from 'react';
import { Tag, Tooltip } from '@databricks/design-system';
import type { Row } from '@tanstack/react-table';
import type { EvaluationDataset } from '../types';

export const DatasetIdCell = ({ row }: { row: Row<EvaluationDataset> }) => {
  const datasetId = row.original.dataset_id;
  const [showTooltip, setShowTooltip] = useState(false);

  const handleClick = () => {
    navigator.clipboard.writeText(datasetId);
    setShowTooltip(true);
    setTimeout(() => {
      setShowTooltip(false);
    }, 3000);
  };

  return (
    <Tooltip
      content={showTooltip ? 'Copied!' : 'Click to copy'}
      open={showTooltip ? true : undefined}
      componentId="mlflow.eval-datasets.dataset-id-tooltip"
    >
      <Tag
        css={{ width: 'fit-content', maxWidth: '100%', cursor: 'pointer' }}
        componentId="mlflow.eval-datasets.dataset-id"
        color="indigo"
        onClick={handleClick}
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
