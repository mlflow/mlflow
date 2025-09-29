import { useMemo } from 'react';

import {
  Button,
  useDesignSystemTheme,
  FormUI,
  SimpleSelect,
  SimpleSelectOption,
  CloseSmallIcon,
  Input,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { TableFilterItemTypeahead } from './TableFilterItemTypeahead';
import { TableFilterItemValueInput } from './TableFilterItemValueInput';
import {
  EXECUTION_DURATION_COLUMN_ID,
  STATE_COLUMN_ID,
  TRACE_NAME_COLUMN_ID,
  USER_COLUMN_ID,
  RUN_NAME_COLUMN_ID,
  LOGGED_MODEL_COLUMN_ID,
  SOURCE_COLUMN_ID,
  CUSTOM_METADATA_COLUMN_ID,
} from '../../hooks/useTableColumns';
import { FilterOperator, TracesTableColumnGroup, TracesTableColumnGroupToLabelMap } from '../../types';
import type {
  AssessmentInfo,
  TableFilter,
  TableFilterOption,
  TableFilterOptions,
  TracesTableColumn,
} from '../../types';

const FILTERABLE_INFO_COLUMNS = [
  EXECUTION_DURATION_COLUMN_ID,
  STATE_COLUMN_ID,
  TRACE_NAME_COLUMN_ID,
  USER_COLUMN_ID,
  RUN_NAME_COLUMN_ID,
  LOGGED_MODEL_COLUMN_ID,
  SOURCE_COLUMN_ID,
];

export const TableFilterItem = ({
  tableFilter,
  index,
  onChange,
  onDelete,
  assessmentInfos,
  experimentId,
  tableFilterOptions,
  allColumns,
}: {
  tableFilter: TableFilter;
  index: number;
  onChange: (filter: TableFilter, index: number) => void;
  onDelete: () => void;
  assessmentInfos: AssessmentInfo[];
  experimentId: string;
  tableFilterOptions: TableFilterOptions;
  allColumns: TracesTableColumn[];
}) => {
  const { column, operator, key } = tableFilter;
  const { theme } = useDesignSystemTheme();

  // For now, we don't support filtering on numeric values.
  const assessmentKeyOptions: TableFilterOption[] = useMemo(
    () =>
      assessmentInfos
        .filter((assessment) => assessment.dtype !== 'numeric')
        .map((assessment) => ({ value: assessment.name, renderValue: () => assessment.displayName })),
    [assessmentInfos],
  );

  const columnOptions: TableFilterOption[] = useMemo(() => {
    const result = allColumns
      .filter(
        (column) => FILTERABLE_INFO_COLUMNS.includes(column.id) || column.id.startsWith(CUSTOM_METADATA_COLUMN_ID),
      )
      .map((column) => ({ value: column.id, renderValue: () => column.label }));

    // Add the tag and assessment column groups
    result.push(
      {
        value: TracesTableColumnGroup.TAG,
        renderValue: () => TracesTableColumnGroupToLabelMap[TracesTableColumnGroup.TAG],
      },
      {
        value: TracesTableColumnGroup.ASSESSMENT,
        renderValue: () => TracesTableColumnGroupToLabelMap[TracesTableColumnGroup.ASSESSMENT],
      },
    );
    return result;
  }, [allColumns]);

  return (
    <>
      <div
        css={{
          display: 'flex',
          flexDirection: 'row',
          gap: theme.spacing.sm,
        }}
      >
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <FormUI.Label htmlFor={`filter-column-${index}`}>
            <FormattedMessage
              defaultMessage="Column"
              description="Label for the column field in the GenAI Traces Table Filter form"
            />
          </FormUI.Label>
          <TableFilterItemTypeahead
            id={`filter-column-${index}`}
            item={columnOptions.find((item) => item.value === column)}
            options={columnOptions}
            onChange={(value: string) => {
              if (value !== column) {
                // Clear other fields as well on column change
                onChange({ column: value, operator: FilterOperator.EQUALS, value: '' }, index);
              }
            }}
            placeholder="Select column"
            width={160}
            canSearchCustomValue={false}
          />
        </div>
        {column === TracesTableColumnGroup.ASSESSMENT && (
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <FormUI.Label htmlFor={`filter-key-${index}`}>
              <FormattedMessage
                defaultMessage="Name"
                description="Label for the name field for assessments in the GenAI Traces Table Filter form"
              />
            </FormUI.Label>
            <TableFilterItemTypeahead
              id={`filter-key-${index}`}
              item={assessmentKeyOptions.find((item) => item.value === key)}
              options={assessmentKeyOptions}
              onChange={(value: string) => {
                onChange({ ...tableFilter, key: value }, index);
              }}
              placeholder="Select name"
              width={200}
              canSearchCustomValue={false}
            />
          </div>
        )}
        {column === TracesTableColumnGroup.TAG && (
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <FormUI.Label htmlFor={`filter-key-${index}`}>
              <FormattedMessage
                defaultMessage="Key"
                description="Label for the key field for tags in the GenAI Traces Table Filter form"
              />
            </FormUI.Label>
            <Input
              aria-label="Key"
              componentId="mlflow.evaluations_review.table_ui.filter_key"
              id={'filter-key-' + index}
              type="text"
              css={{ width: 200 }}
              placeholder={column === TracesTableColumnGroup.TAG ? 'Key' : 'Name'}
              value={key}
              onChange={(e) => {
                onChange({ ...tableFilter, key: e.target.value }, index);
              }}
            />
          </div>
        )}
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <FormUI.Label htmlFor={`filter-operator-${index}`}>
            <FormattedMessage
              defaultMessage="Operator"
              description="Label for the operator field in the GenAI Traces Table Filter form"
            />
          </FormUI.Label>
          <SimpleSelect
            aria-label="Operator"
            componentId="mlflow.evaluations_review.table_ui.filter_operator"
            id={'filter-operator-' + index}
            placeholder="Select"
            width={100}
            contentProps={{
              // Set the z-index to be higher than the Popover
              style: { zIndex: theme.options.zIndexBase + 100 },
            }}
            // Currently only executionTime supports other operators
            value={column === '' || column === EXECUTION_DURATION_COLUMN_ID ? operator : '='}
            disabled={column !== '' && column !== EXECUTION_DURATION_COLUMN_ID}
            onChange={(e) => {
              onChange({ ...tableFilter, operator: e.target.value as FilterOperator }, index);
            }}
          >
            {(Object.values(FilterOperator) as string[]).map((op) => (
              <SimpleSelectOption key={op} value={op}>
                {op}
              </SimpleSelectOption>
            ))}
          </SimpleSelect>
        </div>
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <FormUI.Label htmlFor={`filter-value-${index}`}>
            <FormattedMessage
              defaultMessage="Value"
              description="Label for the value field in the GenAI Traces Table Filter form"
            />
          </FormUI.Label>
          <TableFilterItemValueInput
            index={index}
            tableFilter={tableFilter}
            assessmentInfos={assessmentInfos}
            onChange={onChange}
            experimentId={experimentId}
            tableFilterOptions={tableFilterOptions}
          />
        </div>
        <div
          css={{
            alignSelf: 'flex-end',
          }}
        >
          <Button
            componentId="mlflow.evaluations_review.table_ui.filter_delete_button"
            type="tertiary"
            icon={<CloseSmallIcon />}
            onClick={onDelete}
          />
        </div>
      </div>
    </>
  );
};
