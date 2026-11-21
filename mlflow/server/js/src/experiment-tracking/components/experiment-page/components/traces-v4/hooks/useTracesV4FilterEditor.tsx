import { lazy } from 'react';
// eslint-disable-next-line no-restricted-imports
import { Suspense } from 'react';
import { useIntl } from 'react-intl';
import { FilterOp, type FilterFieldDef, type TraceFilterModel } from '@databricks/web-shared/traces-table';
import type { TracesV4UrlState } from './useTracesV4UrlState';
import type { useTracesV4UnifiedSearch } from './useTracesV4UnifiedSearch';
import { buildFilter } from '../utils/buildTracesV4SearchParams';
import {
  compileFilterModel,
  compileTagFilters,
  isSqlFilterClause,
  isSafeFilterKey,
  isValidDuration,
  SQL_FILTER_FIELD,
} from '../utils/filterModel';
import type { TracesV4ToolbarParams } from '../components/TracesV4Toolbar';
import { parseBasicTraceFilter } from '../utils/traceSqlFilters';

const SqlFilterEditor = lazy(() =>
  import('../components/SqlFilterEditor').then(({ SqlFilterEditor }) => ({ default: SqlFilterEditor })),
);

interface FilterEditorOptions {
  filterModel: TraceFilterModel;
  unifiedSearch: Pick<ReturnType<typeof useTracesV4UnifiedSearch>, 'searchFilterModel' | 'setSearchFilters'>;
  filterFields: FilterFieldDef[];
  url: Pick<TracesV4UrlState, 'tagFilters' | 'setFilterState'>;
  numericAssessmentKeys: ReadonlySet<string>;
  error: unknown;
  isFetching: boolean;
}

/** Both trace tables use the same scoped edits, conversions and rejected-draft recovery. */
export const useTracesV4FilterEditor = ({
  filterModel,
  unifiedSearch,
  filterFields,
  url,
  numericAssessmentKeys,
  error,
  isFetching,
}: FilterEditorOptions): Pick<
  TracesV4ToolbarParams,
  'filterModel' | 'onFilterChange' | 'filterFields' | 'validateFilters' | 'sqlFilter'
> => {
  const intl = useIntl();
  const basicFilterModel = filterModel.filter((clause) => !isSqlFilterClause(clause));
  const sqlClauses = filterModel.filter(isSqlFilterClause);
  const basicEditorModel = unifiedSearch.searchFilterModel.filter((clause) => !isSqlFilterClause(clause));
  const formatBasic = (model: typeof filterModel) =>
    buildFilter({ extraClauses: compileFilterModel(model, numericAssessmentKeys) }) ?? '';
  const parseBasic = (value: string) => parseBasicTraceFilter(value, filterFields, formatBasic);
  const sqlValue = formatBasic(sqlClauses);
  return {
    filterModel: basicEditorModel,
    onFilterChange: (next) => unifiedSearch.setSearchFilters([...next, ...sqlClauses]),
    filterFields,
    validateFilters: (model) => {
      if (model.some(({ key }) => key !== undefined && !isSafeFilterKey(key))) {
        return intl.formatMessage({
          defaultMessage: 'Filter keys cannot contain backticks, double quotes, backslashes, or control characters.',
          description: 'Invalid custom key in trace filter',
        });
      }
      return model.some(({ field, value }) => field === 'duration' && value.trim() && !isValidDuration(value))
        ? intl.formatMessage({
            defaultMessage: 'Enter a whole number of milliseconds.',
            description: 'Invalid trace filter duration',
          })
        : undefined;
    },
    sqlFilter: {
      value:
        buildFilter({
          extraClauses: [
            ...compileFilterModel(filterModel, numericAssessmentKeys),
            ...compileTagFilters(url.tagFilters),
          ],
        }) ?? '',
      onChange: (value) =>
        url.setFilterState(parseBasic(value) ?? [{ field: SQL_FILTER_FIELD, operator: FilterOp.EQUALS, value }], []),
      appliedClause: {
        value: sqlValue,
        onChange: (value) =>
          url.setFilterState(
            [
              ...basicFilterModel,
              ...(value.trim() ? [{ field: SQL_FILTER_FIELD, operator: FilterOp.EQUALS, value: value.trim() }] : []),
            ],
            url.tagFilters,
          ),
      },
      replaceClause: (index, value) => {
        const replacesTag = index >= basicFilterModel.length;
        const model = replacesTag ? basicEditorModel : basicFilterModel;
        const parsed = parseBasic(value);
        const nextBasic = model.flatMap((clause, clauseIndex) => (clauseIndex === index ? (parsed ?? []) : [clause]));
        const nextSql = parsed ? sqlValue : (buildFilter({ extraClauses: [sqlValue, value] }) ?? '');
        const next = [
          ...nextBasic,
          ...(nextSql ? [{ field: SQL_FILTER_FIELD, operator: FilterOp.EQUALS, value: nextSql }] : []),
        ];
        url.setFilterState(next, replacesTag ? [] : url.tagFilters);
        return (
          buildFilter({
            extraClauses: [formatBasic(next), ...(replacesTag ? [] : compileTagFilters(url.tagFilters))],
          }) ?? ''
        );
      },
      formatBasic,
      parseBasic,
      error: error instanceof Error ? error : undefined,
      isFetching,
      preferSql: sqlClauses.length > 0,
      renderEditor: (value, onChange) => (
        <Suspense
          fallback={
            <div role="status">
              {intl.formatMessage({
                defaultMessage: 'Loading SQL editor…',
                description: 'Loading trace SQL editor',
              })}
            </div>
          }
        >
          <SqlFilterEditor value={value} onChange={onChange} fields={filterFields} />
        </Suspense>
      ),
    },
  };
};
