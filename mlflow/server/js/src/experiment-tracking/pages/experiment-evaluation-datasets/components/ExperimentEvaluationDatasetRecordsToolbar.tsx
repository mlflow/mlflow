import {
  Button,
  ChevronDownIcon,
  ColumnsIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxCustomButtonTriggerWrapper,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DropdownMenu,
  RowsIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { ColumnDef } from '@tanstack/react-table';
import { FormattedMessage, useIntl } from 'react-intl';
import { useMemo, useState } from 'react';
import type { EvaluationDataset, EvaluationDatasetRecord } from '../types';
import { parseJSONSafe } from '@mlflow/mlflow/src/common/utils/TagUtils';

const getTotalRecordsCount = (profile: string | undefined): number | undefined => {
  if (!profile) {
    return undefined;
  }

  const profileJson = parseJSONSafe(profile);
  return profileJson?.num_records ?? undefined;
};

const ColumnSelector = ({
  columns,
  columnVisibility,
  setColumnVisibility,
}: {
  columns: ColumnDef<EvaluationDatasetRecord, any>[];
  columnVisibility: Record<string, boolean>;
  setColumnVisibility: (columnVisibility: Record<string, boolean>) => void;
}) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const handleChange = (columnId: string) => {
    setColumnVisibility({
      ...columnVisibility,
      [columnId]: !columnVisibility[columnId],
    });
  };

  return (
    <DialogCombobox
      componentId="mlflow.eval-datasets.records-toolbar.column-selector"
      label="Columns"
      multiSelect
    >
      <DialogComboboxCustomButtonTriggerWrapper>
        <Button
          endIcon={<ChevronDownIcon />}
          componentId="mlflow.eval-datasets.records-toolbar.columns-toggle"
        >
          <div
            css={{
              display: 'flex',
              gap: theme.spacing.sm,
              alignItems: 'center',
            }}
          >
            <ColumnsIcon />
            {intl.formatMessage({
              defaultMessage: 'Columns',
              description: 'Column selector button for dataset records table',
            })}
          </div>
        </Button>
      </DialogComboboxCustomButtonTriggerWrapper>
      <DialogComboboxContent>
        <DialogComboboxOptionList>
          {columns
            .filter((column) => column.id !== 'outputs')
            .map((column) => (
              <DialogComboboxOptionListCheckboxItem
                key={column.id}
                value={column.header as string}
                checked={columnVisibility[column.id ?? ''] ?? false}
                onChange={() => handleChange(column.id ?? '')}
              />
            ))}
        </DialogComboboxOptionList>
      </DialogComboboxContent>
    </DialogCombobox>
  );
};

export const ExperimentEvaluationDatasetRecordsToolbar = ({
  dataset,
  datasetRecords,
  columns,
  columnVisibility,
  setColumnVisibility,
  rowSize,
  setRowSize,
}: {
  dataset: EvaluationDataset;
  datasetRecords: EvaluationDatasetRecord[];
  columns: ColumnDef<EvaluationDatasetRecord, any>[];
  columnVisibility: Record<string, boolean>;
  setColumnVisibility: (columnVisibility: Record<string, boolean>) => void;
  rowSize: 'sm' | 'md' | 'lg';
  setRowSize: (rowSize: 'sm' | 'md' | 'lg') => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const datasetName = dataset?.name;
  const profile = dataset?.profile;
  const totalRecordsCount = getTotalRecordsCount(profile);
  const loadedRecordsCount = datasetRecords.length;

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
        <ColumnSelector
          columns={columns}
          columnVisibility={columnVisibility}
          setColumnVisibility={setColumnVisibility}
        />
      </div>
    </div>
  );
};
