import { Button, ChevronDownIcon, ColumnsIcon, DropdownMenu } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { type RecordColumnId } from '../utils/constants';

interface DatasetRecordsColumnSelectorProps {
  visibleColumns: RecordColumnId[];
  onToggleColumn: (column: RecordColumnId) => void;
  onResetToDefaults: () => void;
}

interface ColumnOption {
  id: RecordColumnId;
  componentId: string;
  label: React.ReactNode;
}

// Adding a column to `RecordColumnId` triggers a TypeScript exhaustiveness check on this array.
const COLUMN_OPTIONS: ReadonlyArray<ColumnOption> = [
  {
    id: 'dataset_record_id',
    componentId: 'mlflow.eval-datasets-v2.records.column-selector.item.dataset_record_id',
    label: (
      <FormattedMessage
        defaultMessage="Record ID"
        description="Column selector label for the dataset record id column"
      />
    ),
  },
  {
    id: 'inputs',
    componentId: 'mlflow.eval-datasets-v2.records.column-selector.item.inputs',
    label: <FormattedMessage defaultMessage="Inputs" description="Column selector label for the inputs column" />,
  },
  {
    id: 'expectations',
    componentId: 'mlflow.eval-datasets-v2.records.column-selector.item.expectations',
    label: (
      <FormattedMessage defaultMessage="Expectations" description="Column selector label for the expectations column" />
    ),
  },
  {
    id: 'create_time',
    componentId: 'mlflow.eval-datasets-v2.records.column-selector.item.create_time',
    label: <FormattedMessage defaultMessage="Created" description="Column selector label for the create-time column" />,
  },
  {
    id: 'created_by',
    componentId: 'mlflow.eval-datasets-v2.records.column-selector.item.created_by',
    label: (
      <FormattedMessage defaultMessage="Created by" description="Column selector label for the created-by column" />
    ),
  },
  {
    id: 'source',
    componentId: 'mlflow.eval-datasets-v2.records.column-selector.item.source',
    label: <FormattedMessage defaultMessage="Source" description="Column selector label for the source column" />,
  },
  {
    id: 'last_updated',
    componentId: 'mlflow.eval-datasets-v2.records.column-selector.item.last_updated',
    label: (
      <FormattedMessage defaultMessage="Last updated" description="Column selector label for the last-updated column" />
    ),
  },
  {
    id: 'last_updated_by',
    componentId: 'mlflow.eval-datasets-v2.records.column-selector.item.last_updated_by',
    label: (
      <FormattedMessage
        defaultMessage="Last updated by"
        description="Column selector label for the last-updated-by column"
      />
    ),
  },
  {
    id: 'tags',
    componentId: 'mlflow.eval-datasets-v2.records.column-selector.item.tags',
    label: <FormattedMessage defaultMessage="Tags" description="Column selector label for the tags column" />,
  },
];

export const DatasetRecordsColumnSelector = ({
  visibleColumns,
  onToggleColumn,
  onResetToDefaults,
}: DatasetRecordsColumnSelectorProps) => {
  const intl = useIntl();
  const isVisible = (column: RecordColumnId) => visibleColumns.includes(column);

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <Button
          componentId="mlflow.eval-datasets-v2.records.column-selector.trigger"
          icon={<ColumnsIcon />}
          endIcon={<ChevronDownIcon />}
          aria-label={intl.formatMessage({
            defaultMessage: 'Select visible columns',
            description: 'Aria label for the column-selector dropdown trigger on the V2 dataset records page',
          })}
        >
          <FormattedMessage
            defaultMessage="Columns ({visible}/{total})"
            description="Column-selector trigger label on the V2 dataset records page, showing the count of currently visible columns out of the total available"
            values={{ visible: visibleColumns.length, total: COLUMN_OPTIONS.length }}
          />
        </Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Content align="end">
        {COLUMN_OPTIONS.map(({ id, componentId, label }) => (
          <DropdownMenu.CheckboxItem
            key={id}
            componentId={componentId}
            checked={isVisible(id)}
            onCheckedChange={() => onToggleColumn(id)}
          >
            <DropdownMenu.ItemIndicator />
            {label}
          </DropdownMenu.CheckboxItem>
        ))}
        <DropdownMenu.Separator />
        <DropdownMenu.Item
          componentId="mlflow.eval-datasets-v2.records.column-selector.reset"
          onClick={onResetToDefaults}
        >
          <FormattedMessage
            defaultMessage="Reset to defaults"
            description="Menu item that resets the dataset records column visibility to defaults"
          />
        </DropdownMenu.Item>
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};
