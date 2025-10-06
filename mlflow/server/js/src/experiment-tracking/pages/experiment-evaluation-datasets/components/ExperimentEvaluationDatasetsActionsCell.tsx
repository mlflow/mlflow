import { DropdownMenu, OverflowIcon, Spinner, TableRowAction, TrashIcon } from '@databricks/design-system';
import { Button } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Row } from '@tanstack/react-table';
import { EvaluationDataset } from '../types';
import { SEARCH_EVALUATION_DATASETS_QUERY_KEY } from '../constants';
import { useDeleteEvaluationDatasetMutation } from '../hooks/useDeleteEvaluationDatasetMutation';
import { useQueryClient } from '@tanstack/react-query';
import { useCallback } from 'react';

// Component for rendering dataset actions
export const ActionsCell = ({ row }: { row: Row<EvaluationDataset> }) => {
  const queryClient = useQueryClient();

  const { deleteEvaluationDatasetMutation, isLoading: isDeletingDataset } = useDeleteEvaluationDatasetMutation({
    onSuccess: () => {
      // invalidate the datasets query
      queryClient.invalidateQueries({ queryKey: [SEARCH_EVALUATION_DATASETS_QUERY_KEY] });
    },
  });

  const handleDelete = useCallback(() => {
    deleteEvaluationDatasetMutation({ datasetId: row.original.dataset_id });
  }, [deleteEvaluationDatasetMutation, row]);

  return (
    <TableRowAction css={{ padding: 0 }}>
      <DropdownMenu.Root>
        <DropdownMenu.Trigger asChild>
          <Button
            componentId="mlflow.eval-datasets.dataset-actions-menu"
            size="small"
            icon={<OverflowIcon />}
            aria-label="Dataset actions"
            css={{ padding: '4px' }}
          />
        </DropdownMenu.Trigger>
        <DropdownMenu.Content align="end">
          <DropdownMenu.Item
            componentId="mlflow.eval-datasets.delete-dataset-menu-option"
            disabled={isDeletingDataset}
            onClick={(e) => {
              e.stopPropagation();
              e.preventDefault();
              handleDelete();
            }}
          >
            <DropdownMenu.IconWrapper>
              <TrashIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage defaultMessage="Delete dataset" description="Delete evaluation dataset menu item" />
            {isDeletingDataset && (
              <DropdownMenu.HintColumn>
                <Spinner />
              </DropdownMenu.HintColumn>
            )}
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Root>
    </TableRowAction>
  );
};
