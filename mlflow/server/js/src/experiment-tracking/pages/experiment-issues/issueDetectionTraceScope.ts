import { FilterOp, type TraceFilterModel } from '@databricks/web-shared/traces-table';
import { compileFilterModel } from '../../components/experiment-page/components/traces-v4/utils/filterModel';
import { buildFilter } from '../../components/experiment-page/components/traces-v4/utils/buildTracesV4SearchParams';
import { MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER_TAG } from '../../constants';
import { MlflowService } from '../../sdk/MlflowService';

export interface IssueDetectionTraceScopeValue {
  searchQuery: string;
  filters: TraceFilterModel;
  // Keep the persisted expression stable across filter UI flag changes.
  filterString?: string;
}

const SEARCH_FIELD = 'issue_detection_search_query';
const MAX_COMPILED_FILTER_LENGTH = 10000;
export class InvalidIssueDetectionTraceScopeError extends Error {}

export const emptyIssueDetectionTraceScope = (): IssueDetectionTraceScopeValue => ({ searchQuery: '', filters: [] });

export const readIssueDetectionTraceScope = (
  tags: readonly { key: string; value?: string | null }[] | undefined | null,
): IssueDetectionTraceScopeValue => {
  const scope = emptyIssueDetectionTraceScope();
  const compiled: string[] = [];
  try {
    const entries: unknown = JSON.parse(
      tags?.find(({ key }) => key === MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER_TAG)?.value || '[]',
    );
    if (!Array.isArray(entries) || entries.length > 20) throw new Error('Invalid trace scope.');
    for (const entry of entries) {
      if (!entry || typeof entry !== 'object') throw new Error('Invalid trace filter.');
      if (typeof entry.value === 'string' && entry.value.includes("'")) {
        throw new Error('Single quotes are not supported in saved trace filter values.');
      }
      if (typeof entry.key === 'string' && entry.key.includes('`')) {
        throw new Error('Backticks are not supported in saved trace filter keys.');
      }
      if (entry.field === SEARCH_FIELD && typeof entry.value === 'string') {
        scope.searchQuery = entry.value;
      } else if (typeof entry.key === 'string' && entry.operator === undefined) {
        if (!entry.key.trim()) throw new Error('Invalid trace filter key.');
        scope.filters.push({
          field: entry.field === 'tag' ? 'tag' : 'metadata',
          operator: FilterOp.EQUALS,
          key: entry.key.trim(),
          value: typeof entry.value === 'string' ? entry.value : '',
        });
      } else if (
        typeof entry.field === 'string' &&
        typeof entry.value === 'string' &&
        Object.values(FilterOp).includes(entry.operator)
      ) {
        scope.filters.push({
          field: entry.field,
          operator: entry.operator,
          value: entry.value,
          ...(typeof entry.key === 'string' ? { key: entry.key } : {}),
        });
      } else {
        throw new Error('Unsupported trace filter.');
      }
      if (entry.compiled !== undefined) {
        if (
          typeof entry.compiled !== 'string' ||
          !entry.compiled.trim() ||
          entry.compiled.length > MAX_COMPILED_FILTER_LENGTH
        ) {
          throw new Error('Invalid compiled trace filter.');
        }
        compiled.push(entry.compiled);
      } else if (
        entry.field === SEARCH_FIELD ||
        (entry.operator !== undefined && entry.operator !== FilterOp.EQUALS) ||
        !entry.key
      ) {
        throw new Error('Rich trace filters require a saved expression.');
      } else {
        compiled.push(...compileFilterModel(scope.filters.slice(-1)));
      }
    }
  } catch {
    throw new InvalidIssueDetectionTraceScopeError(
      'Could not read the saved trace scope. Reset it to all traces, then configure new filters before running analysis.',
    );
  }
  if (compileFilterModel(scope.filters).length !== scope.filters.length) {
    throw new InvalidIssueDetectionTraceScopeError(
      'The saved trace scope contains an unsupported filter. Reset it before running analysis.',
    );
  }
  return compiled.length ? { ...scope, filterString: compiled.join(' AND ') } : scope;
};

export const compileIssueDetectionTraceScope = (scope: IssueDetectionTraceScopeValue): string =>
  scope.filterString ??
  [...compileFilterModel(scope.filters), buildFilter({ searchQuery: scope.searchQuery })].filter(Boolean).join(' AND ');

export const loadIssueDetectionTraceScope = async (experimentId: string) => {
  const { experiment } = await MlflowService.getExperiment({ experiment_id: experimentId });
  return readIssueDetectionTraceScope(experiment?.tags);
};

export const serializeIssueDetectionTraceScope = (scope: IssueDetectionTraceScopeValue): string => {
  const normalized = { filters: scope.filters, searchQuery: scope.searchQuery.trim() };
  if (normalized.searchQuery.includes("'") || normalized.filters.some((clause) => clause.value.includes("'"))) {
    throw new Error('Single quotes are not supported in trace filter values or text search.');
  }
  const entries = normalized.filters.map((clause) => {
    if (clause.key?.includes('`')) {
      throw new Error('Backticks are not supported in trace filter keys.');
    }
    if (['tag', 'metadata', 'assessment', 'expectation'].includes(clause.field) && !clause.key?.trim()) {
      throw new Error('Complete or remove each trace filter before saving.');
    }
    if (clause.field === 'duration' && !/^-?\d+(\.\d+)?$/.test(clause.value.trim())) {
      throw new Error('Enter a numeric duration before saving the trace scope.');
    }
    const compiled = compileFilterModel([clause]);
    if (compiled.length !== 1) throw new Error('Complete or remove each trace filter before saving.');
    return { ...clause, compiled: compiled[0] };
  });
  const search = buildFilter({ searchQuery: normalized.searchQuery });
  const saved = [
    ...entries,
    ...(search ? [{ field: SEARCH_FIELD, value: normalized.searchQuery, compiled: search }] : []),
  ];
  if (saved.some((entry) => entry.compiled.length > MAX_COMPILED_FILTER_LENGTH)) {
    throw new Error('Each compiled trace filter must be at most 10000 characters.');
  }
  if (saved.length > 20) throw new Error('Use at most 20 trace filters, including text search.');
  return saved.length ? JSON.stringify(saved) : '';
};

export const saveIssueDetectionTraceScope = async (experimentId: string, scope: IssueDetectionTraceScopeValue) => {
  const value = serializeIssueDetectionTraceScope(scope);
  await MlflowService.setExperimentTag({
    experiment_id: experimentId,
    key: MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER_TAG,
    value,
  });
  return readIssueDetectionTraceScope([{ key: MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER_TAG, value }]);
};
