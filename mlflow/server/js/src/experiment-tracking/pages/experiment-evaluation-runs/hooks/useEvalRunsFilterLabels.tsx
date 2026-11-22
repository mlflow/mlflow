import { useIntl } from 'react-intl';

import {
  MLFLOW_RUN_TYPE_VALUE_EVALUATION,
  MLFLOW_RUN_TYPE_VALUE_GENAI_EVALUATE,
  MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION,
} from '../../../constants';
import {
  EVAL_RUNS_FILTER_OPERATOR_LABELS,
  EvalRunsFilterOperator,
  FIELD_GROUP_METRICS,
  FIELD_GROUP_PARAMS,
  FIELD_GROUP_RUN,
  FIELD_GROUP_TAGS,
  RUN_TYPE_FIELD_ID,
  type EvalRunsFilterFieldDef,
  type EvalRunsFilterFieldOption,
} from '../evalRunsFilterFields';
import { EvalRunsDatePreset } from './useEvalRunsFilters';

/** Display order for the Date presets, ALL last so "no filter" reads as the reset option. */
export const DATE_PRESET_ORDER: EvalRunsDatePreset[] = [
  EvalRunsDatePreset.LAST_HOUR,
  EvalRunsDatePreset.LAST_24_HOURS,
  EvalRunsDatePreset.LAST_7_DAYS,
  EvalRunsDatePreset.LAST_30_DAYS,
  EvalRunsDatePreset.LAST_YEAR,
  EvalRunsDatePreset.ALL,
];

/**
 * Human labels for the Date presets, used by the standalone Date button. Built with inline
 * `formatMessage` (rather than a `defineMessages` map indexed by enum) to keep FormatJS extraction
 * happy with the enum keys.
 */
export const useEvalRunsFilterLabels = () => {
  const intl = useIntl();

  const datePresetLabels: Record<EvalRunsDatePreset, string> = {
    [EvalRunsDatePreset.LAST_HOUR]: intl.formatMessage({
      defaultMessage: 'Last hour',
      description: 'Date range preset in the evaluation runs filters',
    }),
    [EvalRunsDatePreset.LAST_24_HOURS]: intl.formatMessage({
      defaultMessage: 'Last 24 hours',
      description: 'Date range preset in the evaluation runs filters',
    }),
    [EvalRunsDatePreset.LAST_7_DAYS]: intl.formatMessage({
      defaultMessage: 'Last 7 days',
      description: 'Date range preset in the evaluation runs filters',
    }),
    [EvalRunsDatePreset.LAST_30_DAYS]: intl.formatMessage({
      defaultMessage: 'Last 30 days',
      description: 'Date range preset in the evaluation runs filters',
    }),
    [EvalRunsDatePreset.LAST_YEAR]: intl.formatMessage({
      defaultMessage: 'Last year',
      description: 'Date range preset in the evaluation runs filters',
    }),
    [EvalRunsDatePreset.ALL]: intl.formatMessage({
      defaultMessage: 'All time',
      description: 'Date range preset in the evaluation runs filters',
    }),
  };

  const fieldGroupLabels = new Map<string, string>(
    Object.entries({
      [FIELD_GROUP_RUN]: intl.formatMessage({
        defaultMessage: 'Run',
        description: 'Field group heading in the evaluation runs filter field picker',
      }),
      [FIELD_GROUP_METRICS]: intl.formatMessage({
        defaultMessage: 'Metrics',
        description: 'Field group heading in the evaluation runs filter field picker',
      }),
      [FIELD_GROUP_PARAMS]: intl.formatMessage({
        defaultMessage: 'Parameters',
        description: 'Field group heading in the evaluation runs filter field picker',
      }),
      [FIELD_GROUP_TAGS]: intl.formatMessage({
        defaultMessage: 'Tags',
        description: 'Field group heading in the evaluation runs filter field picker',
      }),
    }),
  );

  // Keyed by field id; metric, param and tag fields show their own key.
  const attributeFieldLabels = new Map<string, string>(
    Object.entries({
      'attributes.run_name': intl.formatMessage({
        defaultMessage: 'Run name',
        description: 'Run name field in the evaluation runs filter',
      }),
      'attributes.status': intl.formatMessage({
        defaultMessage: 'Status',
        description: 'Run status field in the evaluation runs filter',
      }),
      [RUN_TYPE_FIELD_ID]: intl.formatMessage({
        defaultMessage: 'Run type',
        description: 'Run type field in the evaluation runs filter',
      }),
      'attributes.user_id': intl.formatMessage({
        defaultMessage: 'User',
        description: 'User field in the evaluation runs filter',
      }),
      'attributes.start_time': intl.formatMessage({
        defaultMessage: 'Start time (epoch ms)',
        description: 'Run start time field in the evaluation runs filter, compared as epoch milliseconds',
      }),
      'dataset.name': intl.formatMessage({
        defaultMessage: 'Dataset',
        description: 'Dataset field in the evaluation runs filter',
      }),
    }),
  );

  const statusOptionLabels = new Map<string, string>(
    Object.entries({
      RUNNING: intl.formatMessage({ defaultMessage: 'Running', description: 'Run status filter value' }),
      SCHEDULED: intl.formatMessage({ defaultMessage: 'Scheduled', description: 'Run status filter value' }),
      FINISHED: intl.formatMessage({ defaultMessage: 'Finished', description: 'Run status filter value' }),
      FAILED: intl.formatMessage({ defaultMessage: 'Failed', description: 'Run status filter value' }),
      KILLED: intl.formatMessage({ defaultMessage: 'Killed', description: 'Run status filter value' }),
    }),
  );
  const runTypeOptionLabels = new Map<string, string>(
    Object.entries({
      [MLFLOW_RUN_TYPE_VALUE_EVALUATION]: intl.formatMessage({
        defaultMessage: 'Evaluation',
        description: 'Run type filter value',
      }),
      [MLFLOW_RUN_TYPE_VALUE_GENAI_EVALUATE]: intl.formatMessage({
        defaultMessage: 'Agent evaluate',
        description: 'Run type filter value',
      }),
      [MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION]: intl.formatMessage({
        defaultMessage: 'Issue detection',
        description: 'Run type filter value',
      }),
    }),
  );
  const optionLabelsByField = new Map<string, Map<string, string>>([
    ['attributes.status', statusOptionLabels],
    [RUN_TYPE_FIELD_ID, runTypeOptionLabels],
  ]);

  // The comparison operators are symbols; only "contains" is a word.
  const operatorLabels: Record<EvalRunsFilterOperator, string> = {
    ...EVAL_RUNS_FILTER_OPERATOR_LABELS,
    [EvalRunsFilterOperator.CONTAINS]: intl.formatMessage({
      defaultMessage: 'contains',
      description: 'Operator in the evaluation runs filter that matches values containing the text',
    }),
  };

  const getFieldLabel = (field: EvalRunsFilterFieldDef) => attributeFieldLabels.get(field.id) ?? field.label;
  const getFieldGroupLabel = (group: string) => fieldGroupLabels.get(group) ?? group;
  const getOptionLabel = (field: EvalRunsFilterFieldDef, option: EvalRunsFilterFieldOption) =>
    optionLabelsByField.get(field.id)?.get(option.value) ?? option.label;

  return { datePresetLabels, operatorLabels, getFieldLabel, getFieldGroupLabel, getOptionLabel };
};
