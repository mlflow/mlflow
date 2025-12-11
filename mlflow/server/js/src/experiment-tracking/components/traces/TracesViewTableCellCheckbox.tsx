import React from 'react';
import type { ModelTraceInfoWithRunName } from './hooks/useExperimentTraces';
import type { Row } from '@tanstack/react-table';
import { Checkbox } from '@databricks/design-system';

type TracesViewCellCheckboxProps = {
  row: Row<ModelTraceInfoWithRunName>;
};

export const TracesViewTableCellCheckbox = ({ row }: TracesViewCellCheckboxProps) => {
  return (
    <Checkbox
      componentId="codegen_mlflow_app_src_experiment-tracking_components_traces_tracesviewtablecellcheckbox.tsx_12"
      data-testid={`trace-table-cell-checkbox-${row.id}`}
      disabled={!row.getCanSelect()}
      isChecked={row.getIsSelected()}
      wrapperStyle={{ padding: 0, margin: 0 }}
      onChange={() => row.toggleSelected()}
    />
  );
};
