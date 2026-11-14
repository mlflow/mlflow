import type { ExperimentRunsSelectorResult } from '../../components/experiment-page/utils/experimentRuns.selector';
import { isUserFacingTag } from '../../../common/utils/TagUtils';
import {
  MLFLOW_RUN_TYPE_TAG,
  MLFLOW_RUN_TYPE_VALUE_EVALUATION,
  MLFLOW_RUN_TYPE_VALUE_GENAI_EVALUATE,
  MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION,
} from '../../constants';
import { EvalRunsTableColumnId } from './ExperimentEvaluationRunsTable.constants';
import { parseEvalRunsTableKeyedColumnKey } from './ExperimentEvaluationRunsTable.utils';
import { EvalRunsTableKeyedColumnPrefix } from './ExperimentEvaluationRunsTable.constants';

/**
 * Structured multi-key filter for the eval-runs V2 toolbar, mirroring the Traces filter UX
 * (field / operator / value rows) but targeting the SearchRuns query grammar rather than the
 * traces-search dialect. A clause is compiled to a single SearchRuns filter expression and the
 * clauses are AND-joined with the search text and date/type presets (see `buildEvalRunsFilterString`).
 */
export enum EvalRunsFilterOperator {
  EQUALS = '=',
  NOT_EQUALS = '!=',
  GREATER_THAN = '>',
  LESS_THAN = '<',
  GREATER_THAN_OR_EQUALS = '>=',
  LESS_THAN_OR_EQUALS = '<=',
  // Compiled to a `LIKE '%value%'` expression; kept semantic here so the UI can label it "contains".
  CONTAINS = 'CONTAINS',
}

export interface EvalRunsFilterClause {
  /** Left-hand side of the SearchRuns expression, backtick-quoted where the key is arbitrary
   *  (e.g. `metrics.\`accuracy\``, `attributes.status`, `dataset.name`). */
  field: string;
  operator: EvalRunsFilterOperator;
  value: string;
}

export type EvalRunsFilterFieldType = 'string' | 'number' | 'enum';

export interface EvalRunsFilterFieldOption {
  value: string;
  label: string;
}

export interface EvalRunsFilterFieldDef {
  /** SearchRuns left-hand side, ready to drop into a filter expression. */
  id: string;
  label: string;
  type: EvalRunsFilterFieldType;
  /** Section heading in the field picker. */
  group: string;
  /** Only for `enum` fields — the allowed values. */
  options?: EvalRunsFilterFieldOption[];
}

/**
 * Returns the SearchRuns field represented by an evaluation-runs table column.
 * Columns without a corresponding field in the structured filter model return undefined.
 */
export const getEvalRunsFilterFieldForColumn = (columnId: string): string | undefined => {
  switch (columnId) {
    case EvalRunsTableColumnId.runName:
      return 'attributes.run_name';
    case EvalRunsTableColumnId.status:
      return 'attributes.status';
    case EvalRunsTableColumnId.createdAt:
      return 'attributes.start_time';
    case EvalRunsTableColumnId.dataset:
      return 'dataset.name';
    default:
      break;
  }

  const parsedColumn = parseEvalRunsTableKeyedColumnKey(columnId);
  if (!parsedColumn) {
    return undefined;
  }

  switch (parsedColumn.columnType) {
    case EvalRunsTableKeyedColumnPrefix.METRIC:
      return `metrics.\`${parsedColumn.key}\``;
    case EvalRunsTableKeyedColumnPrefix.PARAM:
      return `params.\`${parsedColumn.key}\``;
    case EvalRunsTableKeyedColumnPrefix.TAG:
      if (!isUserFacingTag(parsedColumn.key)) {
        return undefined;
      }
      return `tags.\`${parsedColumn.key}\``;
    default:
      return undefined;
  }
};

// English fallbacks; the UI shows the localized text from `useEvalRunsFilterLabels`.
export const FIELD_GROUP_RUN = 'Run';
export const FIELD_GROUP_METRICS = 'Metrics';
export const FIELD_GROUP_PARAMS = 'Parameters';
export const FIELD_GROUP_TAGS = 'Tags';
export const RUN_TYPE_FIELD_ID = `tags.\`${MLFLOW_RUN_TYPE_TAG}\``;

// MLflow run lifecycle states, as accepted by the SearchRuns `attributes.status` filter.
const RUN_STATUS_OPTIONS: EvalRunsFilterFieldOption[] = [
  { value: 'RUNNING', label: 'Running' },
  { value: 'SCHEDULED', label: 'Scheduled' },
  { value: 'FINISHED', label: 'Finished' },
  { value: 'FAILED', label: 'Failed' },
  { value: 'KILLED', label: 'Killed' },
];

const STRING_OPERATORS = [
  EvalRunsFilterOperator.EQUALS,
  EvalRunsFilterOperator.NOT_EQUALS,
  EvalRunsFilterOperator.CONTAINS,
];
const NUMBER_OPERATORS = [
  EvalRunsFilterOperator.EQUALS,
  EvalRunsFilterOperator.NOT_EQUALS,
  EvalRunsFilterOperator.GREATER_THAN,
  EvalRunsFilterOperator.LESS_THAN,
  EvalRunsFilterOperator.GREATER_THAN_OR_EQUALS,
  EvalRunsFilterOperator.LESS_THAN_OR_EQUALS,
];
const ENUM_OPERATORS = [EvalRunsFilterOperator.EQUALS, EvalRunsFilterOperator.NOT_EQUALS];

export const getOperatorsForFieldType = (type: EvalRunsFilterFieldType): EvalRunsFilterOperator[] => {
  switch (type) {
    case 'number':
      return NUMBER_OPERATORS;
    case 'enum':
      return ENUM_OPERATORS;
    default:
      return STRING_OPERATORS;
  }
};

export const EVAL_RUNS_FILTER_OPERATOR_LABELS: Record<EvalRunsFilterOperator, string> = {
  [EvalRunsFilterOperator.EQUALS]: '=',
  [EvalRunsFilterOperator.NOT_EQUALS]: '≠',
  [EvalRunsFilterOperator.GREATER_THAN]: '>',
  [EvalRunsFilterOperator.LESS_THAN]: '<',
  [EvalRunsFilterOperator.GREATER_THAN_OR_EQUALS]: '≥',
  [EvalRunsFilterOperator.LESS_THAN_OR_EQUALS]: '≤',
  [EvalRunsFilterOperator.CONTAINS]: 'contains',
};

// The run-type tag values, surfaced as an enum field so run type is filtered like any other field
// (the internal `mlflow.*` tags are otherwise hidden from the Tags group by `isUserFacingTag`).
const RUN_TYPE_OPTIONS: EvalRunsFilterFieldOption[] = [
  { value: MLFLOW_RUN_TYPE_VALUE_EVALUATION, label: 'Evaluation' },
  { value: MLFLOW_RUN_TYPE_VALUE_GENAI_EVALUATE, label: 'Agent evaluate' },
  { value: MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION, label: 'Issue detection' },
];

// Static run attributes filterable via SearchRuns.
const getAttributeFields = (excludeIssueDetectionRuns = false): EvalRunsFilterFieldDef[] => [
  { id: 'attributes.run_name', label: 'Run name', type: 'string', group: FIELD_GROUP_RUN },
  { id: 'attributes.status', label: 'Status', type: 'enum', group: FIELD_GROUP_RUN, options: RUN_STATUS_OPTIONS },
  {
    id: RUN_TYPE_FIELD_ID,
    label: 'Run type',
    type: 'enum',
    group: FIELD_GROUP_RUN,
    options: excludeIssueDetectionRuns
      ? RUN_TYPE_OPTIONS.filter((option) => option.value !== MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION)
      : RUN_TYPE_OPTIONS,
  },
  { id: 'attributes.user_id', label: 'User', type: 'string', group: FIELD_GROUP_RUN },
  { id: 'attributes.start_time', label: 'Start time (epoch ms)', type: 'number', group: FIELD_GROUP_RUN },
  { id: 'dataset.name', label: 'Dataset', type: 'string', group: FIELD_GROUP_RUN },
];

/** Resolves SearchRuns fields even when their key is absent from the currently loaded page. */
export const getEvalRunsFilterField = (
  fieldId: string,
  excludeIssueDetectionRuns = false,
): EvalRunsFilterFieldDef | undefined => {
  const attribute = getAttributeFields(excludeIssueDetectionRuns).find((field) => field.id === fieldId);
  if (attribute) return attribute;
  const match = /^(metrics|params|tags)\.`(.+)`$/.exec(fieldId);
  if (!match) return undefined;
  const [, namespace, key] = match;
  if (namespace === 'tags' && !isUserFacingTag(key)) return undefined;
  return {
    id: fieldId,
    label: key,
    type: namespace === 'metrics' ? 'number' : 'string',
    group:
      namespace === 'metrics' ? FIELD_GROUP_METRICS : namespace === 'params' ? FIELD_GROUP_PARAMS : FIELD_GROUP_TAGS,
  };
};

const getTagKeys = (runsData: ExperimentRunsSelectorResult): string[] =>
  Array.from(new Set(runsData.tagsList.flatMap((record) => Object.keys(record)))).filter(isUserFacingTag);

/**
 * Returns values observed in the loaded runs for a field. These values are suggestions only: the
 * filter editor remains editable so callers can enter a value that is not present on this page.
 */
export const getEvalRunsFilterFieldValues = (fieldId: string, runsData: ExperimentRunsSelectorResult): string[] => {
  const values: unknown[] = [];

  if (fieldId === 'attributes.run_name') {
    values.push(...runsData.runInfos.map((runInfo) => runInfo.runName));
  } else if (fieldId === 'attributes.status') {
    values.push(...runsData.runInfos.map((runInfo) => runInfo.status));
  } else if (fieldId === 'attributes.user_id') {
    // `userId` is present in some SearchRuns responses but is not part of the shared RunInfoEntity
    // type used by the selector. Keep this best-effort so older responses remain supported.
    values.push(
      ...runsData.runInfos.map((runInfo) =>
        runInfo && 'userId' in runInfo && typeof runInfo.userId === 'string' ? runInfo.userId : undefined,
      ),
    );
  } else if (fieldId === 'attributes.start_time') {
    values.push(...runsData.runInfos.map((runInfo) => runInfo.startTime));
  } else if (fieldId === 'dataset.name') {
    values.push(
      ...runsData.datasetsList.flatMap((datasets) => datasets.map((datasetWithTags) => datasetWithTags.dataset.name)),
    );
  } else {
    const match = /^(metrics|params|tags)\.`(.+)`$/.exec(fieldId);
    if (match) {
      const [, namespace, key] = match;
      if (namespace === 'metrics') {
        values.push(
          ...runsData.metricsList.flatMap((metrics) =>
            metrics.filter((metric) => metric.key === key).map((metric) => metric.value),
          ),
        );
      } else if (namespace === 'params') {
        values.push(
          ...runsData.paramsList.flatMap((params) =>
            params.filter((param) => param.key === key).map((param) => param.value),
          ),
        );
      } else {
        values.push(...runsData.tagsList.flatMap((tags) => (tags[key] ? [tags[key].value] : [])));
      }
    }
  }

  return Array.from(
    new Set(
      values
        .filter((value): value is string | number => typeof value === 'string' || typeof value === 'number')
        .map(String)
        .filter((value) => value.length > 0),
    ),
  ).sort((left, right) => left.localeCompare(right, undefined, { numeric: true }));
};

/**
 * Builds the filterable field set: static run attributes plus metric/param/tag keys discovered from
 * the loaded runs (so the picker reflects what this experiment actually logged). Keys with arbitrary
 * characters are backtick-quoted in the field id so it drops straight into a SearchRuns expression.
 */
export const getEvalRunsFilterFields = (
  runsData: ExperimentRunsSelectorResult,
  excludeIssueDetectionRuns = false,
): EvalRunsFilterFieldDef[] => [
  ...getAttributeFields(excludeIssueDetectionRuns),
  ...runsData.metricKeyList.map(
    (key): EvalRunsFilterFieldDef => ({
      id: `metrics.\`${key}\``,
      label: key,
      type: 'number',
      group: FIELD_GROUP_METRICS,
    }),
  ),
  ...runsData.paramKeyList.map(
    (key): EvalRunsFilterFieldDef => ({
      id: `params.\`${key}\``,
      label: key,
      type: 'string',
      group: FIELD_GROUP_PARAMS,
    }),
  ),
  ...getTagKeys(runsData).map(
    (key): EvalRunsFilterFieldDef => ({
      id: `tags.\`${key}\``,
      label: key,
      type: 'string',
      group: FIELD_GROUP_TAGS,
    }),
  ),
];

// Numeric fields emit unquoted values; everything else is single-quoted. Derived from the field id
// (not the field list) so compilation stays correct even if a key drops out of the loaded runs.
const isNumericField = (field: string): boolean =>
  field.startsWith('metrics.') || field === 'attributes.start_time' || field === 'attributes.end_time';

/**
 * Quotes a SearchRuns string value. The search parser has no escape for a quote inside a literal, so
 * a value containing `'` is wrapped in double quotes. A value with both quote kinds can't be written;
 * it keeps the doubled-quote form, which matches nothing rather than breaking the expression.
 */
export const quoteFilterValue = (value: string): string => {
  if (!value.includes("'")) {
    return `'${value}'`;
  }
  return value.includes('"') ? `'${value.replace(/'/g, "''")}'` : `"${value}"`;
};

const compileClause = (clause: EvalRunsFilterClause): string | null => {
  const value = clause.value.trim();
  if (!clause.field || !value) {
    return null;
  }
  if (clause.operator === EvalRunsFilterOperator.CONTAINS) {
    return `${clause.field} LIKE ${quoteFilterValue(`%${value}%`)}`;
  }
  if (isNumericField(clause.field)) {
    // Drop clauses whose value isn't a number rather than emit an expression the parser rejects.
    if (!Number.isFinite(Number(value))) {
      return null;
    }
    return `${clause.field} ${clause.operator} ${value}`;
  }
  return `${clause.field} ${clause.operator} ${quoteFilterValue(value)}`;
};

/** Compiles the structured clauses to SearchRuns expressions, skipping incomplete/invalid ones. */
export const buildClausesFilterString = (clauses: EvalRunsFilterClause[]): string =>
  clauses
    .map(compileClause)
    .filter((clause): clause is string => clause !== null)
    .join(' AND ');
