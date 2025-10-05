import {
  Button,
  ColumnsIcon,
  DropdownMenu,
  RowsIcon,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { ColumnDef } from '@tanstack/react-table';
import { FormattedMessage } from 'react-intl';
import { EvaluationDatasetRecord } from '../types';

export const ExperimentEvaluationDatasetRecordsToolbar = ({
  datasetName,
  columns,
  columnVisibility,
  setColumnVisibility,
  loadedRecordsCount,
  totalRecordsCount,
  rowSize,
  setRowSize,
}: {
  datasetName: string;
  columns: ColumnDef<EvaluationDatasetRecord, any>[];
  columnVisibility: Record<string, boolean>;
  setColumnVisibility: (columnVisibility: Record<string, boolean>) => void;
  loadedRecordsCount?: number;
  totalRecordsCount?: number;
  rowSize: 'sm' | 'md' | 'lg';
  setRowSize: (rowSize: 'sm' | 'md' | 'lg') => void;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'space-between',
        marginBottom: theme.spacing.sm,
      }}
    >
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          paddingLeft: theme.spacing.sm,
          paddingRight: theme.spacing.sm,
        }}
      >
        <Typography.Title level={3} withoutMargins>
          {datasetName}
        </Typography.Title>
        <Typography.Text color="secondary" size="sm">
          <FormattedMessage
            defaultMessage="Displaying {loadedRecordsCount} of {totalRecordsCount, plural, =1 {1 record} other {# records}}"
            description="Label for the number of records displayed"
            values={{ loadedRecordsCount: loadedRecordsCount ?? 0, totalRecordsCount: totalRecordsCount ?? 0 }}
          />
        </Typography.Text>
      </div>
      <div css={{ display: 'flex', alignItems: 'flex-start' }}>
        <DropdownMenu.Root>
          <DropdownMenu.Trigger asChild>
            <Button componentId="mlflow.eval-datasets.records-toolbar.row-size-toggle" icon={<RowsIcon />} />
          </DropdownMenu.Trigger>
          <DropdownMenu.Content align="end">
            <DropdownMenu.RadioGroup
              componentId="mlflow.eval-datasets.records-toolbar.row-size-radio"
              value={rowSize}
              onValueChange={(value) => setRowSize(value as 'sm' | 'md' | 'lg')}
            >
              <DropdownMenu.Label>
                <Typography.Text color="secondary">
                  <FormattedMessage defaultMessage="Row height" description="Label for the row height radio group" />
                </Typography.Text>
              </DropdownMenu.Label>
              <DropdownMenu.RadioItem key="sm" value="sm">
                <DropdownMenu.ItemIndicator />
                <Typography.Text>
                  <FormattedMessage defaultMessage="Small" description="Small row size" />
                </Typography.Text>
              </DropdownMenu.RadioItem>
              <DropdownMenu.RadioItem key="md" value="md">
                <DropdownMenu.ItemIndicator />
                <Typography.Text>
                  <FormattedMessage defaultMessage="Medium" description="Medium row size" />
                </Typography.Text>
              </DropdownMenu.RadioItem>
              <DropdownMenu.RadioItem key="lg" value="lg">
                <DropdownMenu.ItemIndicator />
                <Typography.Text>
                  <FormattedMessage defaultMessage="Large" description="Large row size" />
                </Typography.Text>
              </DropdownMenu.RadioItem>
            </DropdownMenu.RadioGroup>
          </DropdownMenu.Content>
        </DropdownMenu.Root>
        <DropdownMenu.Root>
          <DropdownMenu.Trigger asChild>
            <Button componentId="mlflow.eval-datasets.records-toolbar.columns-toggle" icon={<ColumnsIcon />} />
          </DropdownMenu.Trigger>
          <DropdownMenu.Content>
            {columns.map((column) => (
              <DropdownMenu.CheckboxItem
                componentId="YOUR_TRACKING_ID"
                key={column.id}
                checked={columnVisibility[column.id ?? ''] ?? false}
                onCheckedChange={(checked) =>
                  setColumnVisibility({
                    ...columnVisibility,
                    [column.id ?? '']: checked,
                  })
                }
              >
                <DropdownMenu.ItemIndicator />
                <Typography.Text>{column.header}</Typography.Text>
              </DropdownMenu.CheckboxItem>
            ))}
          </DropdownMenu.Content>
        </DropdownMenu.Root>
      </div>
    </div>
  );
};
