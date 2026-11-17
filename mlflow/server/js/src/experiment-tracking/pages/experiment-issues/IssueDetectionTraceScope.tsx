import { TraceFilterButton } from '@databricks/web-shared/traces-table';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { TracesV4Search } from '../../components/experiment-page/components/traces-v4/components/TracesV4Search';
import { useMlflowTraceFilterFields } from '../../components/experiment-page/components/traces-v4/utils/filterModel';
import { parseTraceSearchQuery } from '../../components/experiment-page/components/traces-v4/utils/traceSearchQuery';
import { buildTraceSearchFields } from '../../components/experiment-page/components/traces-v4/utils/traceSearchSuggestions';
import { emptyIssueDetectionTraceScope, type IssueDetectionTraceScopeValue } from './issueDetectionTraceScope';

const NO_TRACES: ModelTraceInfoV3[] = [];

export const useNormalizeIssueDetectionTraceScope = () => {
  const fields = useMlflowTraceFilterFields();
  return (scope: IssueDetectionTraceScopeValue): IssueDetectionTraceScopeValue => {
    const query = parseTraceSearchQuery(scope.searchQuery, fields, buildTraceSearchFields(fields, NO_TRACES));
    return query.status === 'complete' && query.clause
      ? { searchQuery: '', filters: [...scope.filters, query.clause] }
      : scope;
  };
};

export const IssueDetectionTraceScope = ({
  scope,
  onChange,
  disabled = false,
}: {
  scope: IssueDetectionTraceScopeValue;
  onChange: (scope: IssueDetectionTraceScopeValue) => void;
  disabled?: boolean;
}) => {
  const fields = useMlflowTraceFilterFields();
  const normalize = useNormalizeIssueDetectionTraceScope();
  const update = (value: IssueDetectionTraceScopeValue) =>
    onChange({ searchQuery: value.searchQuery, filters: value.filters });
  const clear = () => onChange(emptyIssueDetectionTraceScope());
  return (
    <fieldset disabled={disabled} css={{ border: 0, margin: 0, padding: 0, minWidth: 0 }}>
      <TraceFilterButton
        fields={fields}
        filterModel={scope.filters}
        onChange={(filters) => update({ ...scope, filters })}
        onClearAll={clear}
        activeCount={scope.filters.length}
        renderTrigger={(openEditor) => (
          <TracesV4Search
            traces={NO_TRACES}
            filterModel={scope.filters}
            value={scope.searchQuery}
            onChange={(searchQuery) => update({ ...scope, searchQuery })}
            onClear={clear}
            onSubmit={() => update(normalize(scope))}
            onRunTextSearch={(searchQuery) => update({ ...scope, searchQuery })}
            onApplyFilter={(clause) => update({ searchQuery: '', filters: [...scope.filters, clause] })}
            onRemoveFilter={(index) => update({ ...scope, filters: scope.filters.filter((_, i) => i !== index) })}
            onOpenFilter={openEditor}
          />
        )}
      />
    </fieldset>
  );
};
