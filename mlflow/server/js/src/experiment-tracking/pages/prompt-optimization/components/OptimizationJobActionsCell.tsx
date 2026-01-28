import type React from 'react';
import type { ColumnDef } from '@tanstack/react-table';
import {
  Button,
  DropdownMenu,
  StopIcon,
  TrashIcon,
  OverflowIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import type { PromptOptimizationJob } from '../types';
import { isJobRunning } from '../types';
import type { OptimizationJobsTableMetadata } from './OptimizationJobsListTable';

export const OptimizationJobActionsCell: ColumnDef<PromptOptimizationJob>['cell'] = ({
  row: { original },
  table: {
    options: { meta },
  },
}) => {
  const { theme } = useDesignSystemTheme();
  const { onCancelJob, onDeleteJob } = (meta || {}) as OptimizationJobsTableMetadata;
  const jobId = original.job_id;
  const canCancel = isJobRunning(original.state?.status);

  if (!jobId) {
    return null;
  }

  // Stop propagation to prevent row click when interacting with the dropdown
  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
  };

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <Button
          componentId="mlflow.prompt-optimization.list.actions"
          icon={<OverflowIcon />}
          size="small"
          type="tertiary"
          onClick={handleClick}
          css={{
            opacity: 0,
            '[role=row]:hover &': { opacity: 1 },
          }}
        />
      </DropdownMenu.Trigger>
      <DropdownMenu.Content align="end">
        {canCancel && (
          <DropdownMenu.Item
            componentId="mlflow.prompt-optimization.list.actions.cancel"
            onClick={() => onCancelJob(jobId)}
          >
            <DropdownMenu.IconWrapper>
              <StopIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage
              defaultMessage="Cancel"
              description="Label for the cancel action in the optimization jobs table"
            />
          </DropdownMenu.Item>
        )}
        <DropdownMenu.Item
          componentId="mlflow.prompt-optimization.list.actions.delete"
          onClick={() => onDeleteJob(jobId)}
          css={{ color: theme.colors.textValidationDanger }}
        >
          <DropdownMenu.IconWrapper>
            <TrashIcon css={{ color: theme.colors.textValidationDanger }} />
          </DropdownMenu.IconWrapper>
          <FormattedMessage
            defaultMessage="Delete"
            description="Label for the delete action in the optimization jobs table"
          />
        </DropdownMenu.Item>
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};
