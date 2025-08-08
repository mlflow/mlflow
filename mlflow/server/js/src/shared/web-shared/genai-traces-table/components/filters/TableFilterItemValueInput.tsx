import { useCallback, useState, useMemo } from 'react';

import { useDesignSystemTheme, Input } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

import { TableFilterItemTypeahead } from './TableFilterItemTypeahead';
import { ExperimentViewTracesStatusLabels } from '../../cellRenderers/StatusRenderer';
import {
  assessmentValueToSerializedString,
  serializedStringToAssessmentValueV2,
} from '../../hooks/useAssessmentFilters';
import { useExperimentVersionsQuery } from '../../hooks/useExperimentVersionsQuery';
import { useGenAiExperimentRunsForComparison } from '../../hooks/useGenAiExperimentRunsForComparison';
import {
  EXECUTION_DURATION_COLUMN_ID,
  STATE_COLUMN_ID,
  RUN_NAME_COLUMN_ID,
  LOGGED_MODEL_COLUMN_ID,
  SOURCE_COLUMN_ID,
} from '../../hooks/useTableColumns';
import { TracesTableColumnGroup } from '../../types';
import type { AssessmentInfo, TableFilter, TableFilterOption, TableFilterOptions } from '../../types';
import { getAssessmentValueLabel } from '../GenAiEvaluationTracesReview.utils';

export const TableFilterItemValueInput = ({
  index,
  tableFilter,
  assessmentInfos,
  onChange,
  experimentId,
  tableFilterOptions,
}: {
  index: number;
  tableFilter: TableFilter;
  assessmentInfos: AssessmentInfo[];
  onChange: (tableFilter: TableFilter, index: number) => void;
  experimentId: string;
  tableFilterOptions: TableFilterOptions;
}) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const id = `filter-value-${index}`;

  const [localValue, setLocalValue] = useState(tableFilter.value);

  const onValueBlur = useCallback(() => {
    if (localValue !== tableFilter.value) {
      onChange({ ...tableFilter, value: localValue }, index);
    }
  }, [tableFilter, index, onChange, localValue]);

  // Fetch runs data when the run name column is selected
  const runsQuery = useGenAiExperimentRunsForComparison(experimentId);

  // Transform runs data into the format expected by TableFilterItemTypeahead
  const runNameQuery = useMemo(() => {
    const transformedData = runsQuery.runInfos
      ?.filter((run) => run.runUuid && run.runName)
      ?.map((run) => ({
        value: run.runUuid as string,
        renderValue: () => run.runName as string,
      }));

    return {
      data: transformedData,
      isLoading: runsQuery.isLoading,
    };
  }, [runsQuery]);

  // Fetch versions data when the version column is selected
  const versionsDataQuery = useExperimentVersionsQuery(experimentId);

  // Transform versions data into the format expected by TableFilterItemTypeahead
  const versionsQuery = useMemo(() => {
    const transformedData = versionsDataQuery.data?.map((loggedModel) => ({
      value: loggedModel.info.model_id,
      renderValue: () => loggedModel.info.name,
    }));

    return {
      data: transformedData,
      isLoading: versionsDataQuery.isLoading,
    };
  }, [versionsDataQuery]);

  const stateOptions: TableFilterOption[] = [
    { value: 'IN_PROGRESS', renderValue: () => intl.formatMessage(ExperimentViewTracesStatusLabels.IN_PROGRESS) },
    { value: 'OK', renderValue: () => intl.formatMessage(ExperimentViewTracesStatusLabels.OK) },
    { value: 'ERROR', renderValue: () => intl.formatMessage(ExperimentViewTracesStatusLabels.ERROR) },
  ];

  if (tableFilter.column === RUN_NAME_COLUMN_ID) {
    return (
      <TableFilterItemTypeahead
        id={id}
        item={runNameQuery.data?.find((item) => item.value === tableFilter.value)}
        query={runNameQuery}
        onChange={(value: string) => {
          onChange({ ...tableFilter, value }, index);
        }}
        placeholder="Select run"
        width={200}
        canSearchCustomValue={false}
      />
    );
  }

  if (tableFilter.column === LOGGED_MODEL_COLUMN_ID) {
    return (
      <TableFilterItemTypeahead
        id={id}
        item={versionsQuery.data?.find((item) => item.value === tableFilter.value)}
        query={versionsQuery}
        onChange={(value: string) => {
          onChange({ ...tableFilter, value }, index);
        }}
        placeholder="Select version"
        width={200}
        canSearchCustomValue={false}
      />
    );
  }

  if (tableFilter.column === TracesTableColumnGroup.ASSESSMENT) {
    const assessmentInfo = assessmentInfos.find((assessment) => assessment.name === tableFilter.key);
    if (assessmentInfo && assessmentInfo.dtype !== 'numeric' && assessmentInfo.dtype !== 'unknown') {
      const options: TableFilterOption[] = Array.from(assessmentInfo.uniqueValues.values()).map((value) => {
        return {
          value: assessmentValueToSerializedString(value),
          renderValue: () => getAssessmentValueLabel(intl, theme, assessmentInfo, value).content,
        };
      });

      return (
        <TableFilterItemTypeahead
          id={id}
          item={options.find((item) => item.value === assessmentValueToSerializedString(tableFilter.value))}
          options={options}
          onChange={(value: string) => {
            onChange({ ...tableFilter, value: serializedStringToAssessmentValueV2(value) }, index);
          }}
          placeholder="Select"
          width={200}
          canSearchCustomValue={false}
        />
      );
    }
  }

  if (tableFilter.column === STATE_COLUMN_ID) {
    return (
      <TableFilterItemTypeahead
        id={id}
        item={stateOptions.find((item) => item.value === tableFilter.value)}
        options={stateOptions}
        onChange={(value: string) => {
          onChange({ ...tableFilter, value }, index);
        }}
        placeholder="Select"
        width={200}
        canSearchCustomValue={false}
      />
    );
  }

  if (tableFilter.column === SOURCE_COLUMN_ID) {
    const sourceOptions = tableFilterOptions.source;
    return (
      <TableFilterItemTypeahead
        id={id}
        item={sourceOptions.find((item) => item.value === tableFilter.value)}
        options={sourceOptions}
        onChange={(value: string) => {
          onChange({ ...tableFilter, value }, index);
        }}
        placeholder="Select source"
        width={200}
        canSearchCustomValue={false}
      />
    );
  }

  return (
    <Input
      aria-label="Value"
      componentId="mlflow.evaluations_review.table_ui.filter_value"
      id={id}
      placeholder={tableFilter.column === EXECUTION_DURATION_COLUMN_ID ? 'Time in milliseconds' : 'Value'}
      type={tableFilter.column === EXECUTION_DURATION_COLUMN_ID ? 'number' : 'text'}
      value={localValue as string}
      onChange={(e) => {
        setLocalValue(e.target.value);
      }}
      onBlur={onValueBlur}
      css={{ width: 200 }}
      // Disable it for assessment column at this point, since the data type is not supported yet.
      disabled={tableFilter.column === TracesTableColumnGroup.ASSESSMENT}
    />
  );
};
