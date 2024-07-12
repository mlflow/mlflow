import React from 'react';
import { ModelTraceInfoWithRunName } from './hooks/useExperimentTraces';
import { Table } from '@tanstack/react-table';
import { Checkbox } from '@databricks/design-system';

type TracesViewTableCheckboxProps = {
  table: Table<ModelTraceInfoWithRunName>;
};

export const TracesViewTableHeaderCheckbox = ({ table }: TracesViewTableCheckboxProps) => {
  const isChecked = table.getIsAllRowsSelected() || (table.getIsSomeRowsSelected() ? null : false);

  return (
    <Checkbox
      data-testid="trace-table-header-checkbox"
      isChecked={isChecked}
      wrapperStyle={{ padding: 0, margin: 0 }}
      onChange={table.toggleAllRowsSelected}
    />
  );
};
