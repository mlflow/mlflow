import React from 'react';
import { ModelTraceInfoWithRunName } from './hooks/useExperimentTraces';
import { Row } from '@tanstack/react-table';
import { Checkbox } from '@databricks/design-system';

type TracesViewCellCheckboxProps = {
  row: Row<ModelTraceInfoWithRunName>;
};

export const TracesViewTableCellCheckbox = ({ row }: TracesViewCellCheckboxProps) => {
  return (
    <Checkbox
      data-testid={`trace-table-cell-checkbox-${row.id}`}
      disabled={!row.getCanSelect()}
      isChecked={row.getIsSelected()}
      wrapperStyle={{ padding: 0, margin: 0 }}
      onChange={() => row.toggleSelected()}
    />
  );
};
