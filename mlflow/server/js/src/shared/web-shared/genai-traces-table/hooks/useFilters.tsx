import { useCallback, useMemo } from 'react';

import { assessmentValueToSerializedString, serializedStringToAssessmentValueV2 } from './useAssessmentFilters';
import { type TableFilter, type TableFilterValue, type FilterOperator, TracesTableColumnGroup } from '../types';
import { useSearchParams } from '../utils/RoutingUtils';

const QUERY_PARAM_KEY = 'filter';
const VALUE_SEPARATOR = '::';

/**
 * Query param-powered hook that manages both generic and assessment filters.
 * Each filter is stored in the URL as: key::operator::value::type
 */
export const useFilters = () => {
  const [searchParams, setSearchParams] = useSearchParams();

  const filters: TableFilter[] = useMemo(() => {
    const filtersUrl = searchParams.getAll(QUERY_PARAM_KEY) ?? [];

    return filtersUrl.reduce<TableFilter[]>((filters, urlFilter) => {
      const [column, urlOperator, value, key] = urlFilter.split(VALUE_SEPARATOR);
      if (!column || !urlOperator || !value) return filters;

      const operator = urlOperator as FilterOperator;

      const isAssessmentFilter = column === TracesTableColumnGroup.ASSESSMENT;
      let filterValue: TableFilterValue = value;
      if (isAssessmentFilter) {
        filterValue = serializedStringToAssessmentValueV2(value);
      }

      filters.push({
        column,
        key,
        operator,
        value: filterValue,
      });

      return filters;
    }, []);
  }, [searchParams]);

  const setFilters = useCallback(
    (newFilters: TableFilter[] | undefined, replace = false) => {
      setSearchParams(
        (params: URLSearchParams) => {
          params.delete(QUERY_PARAM_KEY);

          if (newFilters) {
            newFilters.forEach((filter) => {
              let filterValue = filter.value;
              if (filter.column === TracesTableColumnGroup.ASSESSMENT) {
                filterValue = assessmentValueToSerializedString(filter.value);
              }
              params.append(
                QUERY_PARAM_KEY,
                [filter.column, filter.operator, filterValue, filter.key].join(VALUE_SEPARATOR),
              );
            });
          }

          return params;
        },
        { replace },
      );
    },
    [setSearchParams],
  );

  return [filters, setFilters] as const;
};
