import { Checkbox } from '@databricks/design-system';
import type { ColumnDef } from '@tanstack/react-table';
import { useIntl } from 'react-intl';

import type { RunEntityOrGroupData } from './ExperimentEvaluationRunsPage.utils';

/** Select-all checkbox header cell: toggles every selectable row, indeterminate when only some are checked. */
export const SelectAllHeaderCell: ColumnDef<RunEntityOrGroupData>['header'] = ({ table }) => {
  const intl = useIntl();
  return (
    <Checkbox
      componentId="mlflow.eval-runs.select-all-checkbox"
      data-testid="eval-runs-table-select-all-checkbox"
      aria-label={intl.formatMessage({
        defaultMessage: 'Select all runs',
        description: 'Accessible label for the select-all checkbox in the evaluation runs table header',
      })}
      isChecked={table.getIsAllRowsSelected() ? true : table.getIsSomeRowsSelected() ? null : false}
      wrapperStyle={{ padding: 0, margin: 0 }}
      onChange={(isChecked) => table.toggleAllRowsSelected(isChecked)}
    />
  );
};
