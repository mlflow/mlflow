import React from 'react';
import type { ModelTraceInfoWithRunName } from './hooks/useExperimentTraces';
import type { Table } from '@tanstack/react-table';
import { Checkbox } from '@databricks/design-system';

type TracesViewTableCheckboxProps = {
  table: Table<ModelTraceInfoWithRunName>;
};

export const TracesViewTableHeaderCheckbox = ({ table }: TracesViewTableCheckboxProps) => {
  const isChecked = table.getIsAllRowsSelected() || (table.getIsSomeRowsSelected() ? null : false);

  return (
    <Checkbox
      componentId="codegen_mlflow_app_src_experiment-tracking_components_traces_tracesviewtableheadercheckbox.tsx_14"
      data-testid="trace-table-header-checkbox"
      isChecked={isChecked}
      wrapperStyle={{ padding: 0, margin: 0 }}
      onChange={table.toggleAllRowsSelected}
    />
  );
};
