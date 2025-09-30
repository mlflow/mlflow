import { DropdownMenu, OverflowIcon, PencilIcon, TableRowAction, TrashIcon } from '@databricks/design-system';
import { Button } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Row } from '@tanstack/react-table';
import { EvaluationDataset } from '../types';

// Component for rendering dataset actions
export const ActionsCell = ({ row }: { row: Row<EvaluationDataset> }) => {
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
            onClick={(e) => {
              e.stopPropagation();
            }}
          >
            <DropdownMenu.IconWrapper>
              <TrashIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage defaultMessage="Delete dataset" description="Delete evaluation dataset menu item" />
          </DropdownMenu.Item>
          <DropdownMenu.Item
            componentId="mlflow.eval-datasets.edit-tags-menu-option"
            onClick={(e) => {
              e.stopPropagation();
            }}
          >
            <DropdownMenu.IconWrapper>
              <PencilIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage defaultMessage="Edit tags" description="Edit evaluation dataset tags menu item" />
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Root>
    </TableRowAction>
  );
};
