import {
  ChevronDownIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxCustomButtonTriggerWrapper,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  Button,
  ColumnsIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

import type { TracesTableColumn } from '../types';
import { COLUMN_SELECTOR_DROPDOWN_COMPONENT_ID } from '../utils/EvaluationLogging';

/**
 * Component for column selector in MLflow monitoring traces view. Allows user to control which assessments show up in table to prevent too much clutter.
 */
export const EvaluationsOverviewColumnSelector = ({
  columns,
  selectedColumns,
  setSelectedColumns,
  setSelectedColumnsWithHiddenColumns,
}: {
  columns: TracesTableColumn[];
  selectedColumns: TracesTableColumn[];
  // @deprecated use setSelectedColumnsWithHiddenColumns instead
  setSelectedColumns?: React.Dispatch<React.SetStateAction<TracesTableColumn[]>>;
  setSelectedColumnsWithHiddenColumns?: (newColumns: TracesTableColumn[]) => void;
}) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const handleChange = (newColumn: TracesTableColumn) => {
    if (setSelectedColumnsWithHiddenColumns) {
      return setSelectedColumnsWithHiddenColumns([newColumn]);
    } else if (setSelectedColumns) {
      setSelectedColumns((current: TracesTableColumn[]) => {
        const newSelectedColumns = current.some((col) => col.id === newColumn.id)
          ? current.filter((col) => col.id !== newColumn.id)
          : [...current, newColumn];
        return newSelectedColumns;
      });
    }
  };

  return (
    <DialogCombobox componentId={COLUMN_SELECTOR_DROPDOWN_COMPONENT_ID} label="Columns" multiSelect>
      <DialogComboboxCustomButtonTriggerWrapper>
        <Button endIcon={<ChevronDownIcon />} componentId="mlflow.evaluations_review.table_ui.filter_button">
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
              description: 'Evaluation review > evaluations list > filter dropdown button',
            })}
          </div>
        </Button>
      </DialogComboboxCustomButtonTriggerWrapper>
      <DialogComboboxContent>
        <DialogComboboxOptionList>
          {columns.map((column) => (
            <DialogComboboxOptionListCheckboxItem
              key={column.id}
              value={column.label}
              checked={selectedColumns.some((col) => col.id === column.id)}
              onChange={() => handleChange(column)}
            />
          ))}
        </DialogComboboxOptionList>
      </DialogComboboxContent>
    </DialogCombobox>
  );
};
