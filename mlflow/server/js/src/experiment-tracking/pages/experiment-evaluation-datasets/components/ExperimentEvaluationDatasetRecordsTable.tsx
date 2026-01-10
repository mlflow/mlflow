import {
  Button,
  ColumnsIcon,
  DeleteIcon,
  DialogCombobox,
  DropdownMenu,
  Input,
  RowsIcon,
  SearchIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useState } from 'react';
import type { ColumnDef } from '@tanstack/react-table';
import type { Table } from '@tanstack/react-table';
import { FormattedMessage, useIntl } from 'react-intl';
import type { EvaluationDataset, EvaluationDatasetRecord } from '../types';
import { parseJSONSafe } from '@mlflow/mlflow/src/common/utils/TagUtils';

const getTotalRecordsCount = (profile: string | undefined): number | undefined => {
  if (!profile) {
    return undefined;
  }

  const profileJson = parseJSONSafe(profile);
  return profileJson?.num_records ?? undefined;
};

export const ExperimentEvaluationDatasetRecordsToolbar = ({
  dataset,
  datasetRecords,
  columns,
  columnVisibility,
  setColumnVisibility,
  rowSize,
  setRowSize,
  searchFilter,
  setSearchFilter,
  rowSelection,
  setRowSelection,
  table,
  onDeleteRecords,
}: {
  dataset: EvaluationDataset;
  datasetRecords: EvaluationDatasetRecord[];
  columns: ColumnDef<EvaluationDatasetRecord, any>[];
  columnVisibility: Record<string, boolean>;
  setColumnVisibility: (columnVisibility: Record<string, boolean>) => void;
  rowSize: 'sm' | 'md' | 'lg';
  setRowSize: (rowSize: 'sm' | 'md' | 'lg') => void;
  searchFilter: string;
  setSearchFilter: (searchFilter: string) => void;
  rowSelection: Record<string, boolean>;
  setRowSelection: (rowSelection: Record<string, boolean>) => void;
  table: Table<EvaluationDatasetRecord>;
  onDeleteRecords?: (recordIds: string[]) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const datasetName = dataset?.name;
  const profile = dataset?.profile;
  const totalRecordsCount = getTotalRecordsCount(profile);
  const loadedRecordsCount = datasetRecords.length;

  const selectedRowCount = Object.keys(rowSelection).length;
  const hasSelectedRows = selectedRowCount > 0;
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);

  const handleDeleteSelected = () => {
    if (!onDeleteRecords) return;
    
    const selectedRows = table.getSelectedRowModel().rows;
    const recordIds = selectedRows.map((row) => row.original.dataset_record_id);
    
    onDeleteRecords(recordIds);
    setIsDeleteDialogOpen(false);
  };

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        marginBottom: theme.spacing.sm,
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
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
            {hasSelectedRows ? (
              <FormattedMessage
                defaultMessage="{selectedCount, plural, =1 {1 record selected} other {# records selected}}"
                description="Label for the number of selected records"
                values={{ selectedCount: selectedRowCount }}
              />
            ) : (
              <FormattedMessage
                defaultMessage="Displaying {loadedRecordsCount} of {totalRecordsCount, plural, =1 {1 record} other {# records}}"
                description="Label for the number of records displayed"
                values={{ loadedRecordsCount: loadedRecordsCount ?? 0, totalRecordsCount: totalRecordsCount ?? 0 }}
              />
            )}
          </Typography.Text>
        </div>
        <div css={{ display: 'flex', alignItems: 'flex-start', gap: theme.spacing.xs }}>
          {hasSelectedRows && (
            <>
              <Button
                componentId="mlflow.eval-datasets.records-toolbar.delete-button"
                icon={<DeleteIcon />}
                danger
                onClick={() => setIsDeleteDialogOpen(true)}
              >
                <FormattedMessage
                  defaultMessage="Delete {count, plural, =1 {1 record} other {# records}}"
                  description="Delete button label"
                  values={{ count: selectedRowCount }}
                />
              </Button>
              
              <DialogCombobox.Dialog
                componentId="mlflow.eval-datasets.records-toolbar.delete-dialog"
                title={intl.formatMessage({
                  defaultMessage: 'Delete records',
                  description: 'Delete confirmation dialog title',
                })}
                isOpen={isDeleteDialogOpen}
                onClose={() => setIsDeleteDialogOpen(false)}
              >
                <DialogCombobox.Body>
                  <Typography.Text>
                    <FormattedMessage
                      defaultMessage="Are you sure you want to delete {count, plural, =1 {this record} other {these # records}}? This action cannot be undone."
                      description="Delete confirmation message"
                      values={{ count: selectedRowCount }}
                    />
                  </Typography.Text>
                </DialogCombobox.Body>
                <DialogCombobox.Footer>
                  <Button
                    componentId="mlflow.eval-datasets.records-toolbar.delete-cancel"
                    onClick={() => setIsDeleteDialogOpen(false)}
                  >
                    <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
                  </Button>
                  <Button
                    componentId="mlflow.eval-datasets.records-toolbar.delete-confirm"
                    danger
                    type="primary"
                    onClick={handleDeleteSelected}
                  >
                    <FormattedMessage defaultMessage="Delete" description="Confirm delete button" />
                  </Button>
                </DialogCombobox.Footer>
              </DialogCombobox.Dialog>
            </>
          )}
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
                  componentId="mlflow.eval-datasets.records-toolbar.column-checkbox"
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
      <div
        css={{
          paddingLeft: theme.spacing.sm,
          paddingRight: theme.spacing.sm,
        }}
      >
        <Input
          componentId="mlflow.eval-datasets.records-toolbar.search-input"
          prefix={<SearchIcon />}
          placeholder="Search inputs and expectations"
          value={searchFilter}
          onChange={(e) => setSearchFilter(e.target.value)}
          css={{ maxWidth: '540px', flex: 1 }}
        />
      </div>
    </div>
  );
};