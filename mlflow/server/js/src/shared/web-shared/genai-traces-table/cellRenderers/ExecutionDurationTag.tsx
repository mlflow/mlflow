import React from 'react';

import { ClockIcon, Tag } from '@databricks/design-system';

interface ExecutionDurationTagProps {
  value: string;
}

export const ExecutionDurationTag: React.FC<ExecutionDurationTagProps> = ({ value }) => (
  <Tag
    icon={<ClockIcon />}
    css={{ width: 'fit-content', maxWidth: '100%' }}
    componentId="mlflow.genai-traces-table.execution-time"
  >
    <span
      css={{
        display: 'block',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
      }}
      title={value}
    >
      {value}
    </span>
  </Tag>
);
