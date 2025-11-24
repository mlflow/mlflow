import React, { useRef, useState } from 'react';
import { Tag, Tooltip } from '@databricks/design-system';
import type { Row } from '@tanstack/react-table';
import { useClipboard } from 'use-clipboard-copy';
import type { EvaluationDataset } from '../types';

export const DatasetIdCell = ({ row }: { row: Row<EvaluationDataset> }) => {
  const datasetId = row.original.dataset_id;
  const clipboard = useClipboard();
  const copiedTimerIdRef = useRef<number>();
  const [copied, setCopied] = useState(false);
  const [open, setOpen] = useState(false);

  const handleClick = () => {
    clipboard.copy(datasetId);
    window.clearTimeout(copiedTimerIdRef.current);
    setCopied(true);
    copiedTimerIdRef.current = window.setTimeout(() => {
      setCopied(false);
    }, 3000);
  };

  const tooltipOpen = open || copied;
  const tooltipMessage = copied ? 'Copied!' : 'Click to copy';

  return (
    <Tooltip
      content={tooltipMessage}
      open={tooltipOpen}
      onOpenChange={setOpen}
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
