import { useCallback, useEffect, useMemo } from 'react';

import { useLocalStorage } from '../../hooks/useLocalStorage';

import { assessmentValueToSerializedString, serializedStringToAssessmentValueV2 } from './useAssessmentFilters';
import {
  type TableFilter,
  type TableFilterValue,
  type FilterOperator,
  TracesTableColumnGroup,
  isNullOperator,
} from '../types';
import { useSearchParams } from '../utils/RoutingUtils';

const QUERY_PARAM_KEY = 'filter';
const VALUE_SEPARATOR = '::';

/**
 * Query param-powered hook that manages both generic and assessment filters.
 * Each filter is stored in the URL as: key::operator::value::type
 */
export const useFilters = ({
  persist = false,
  loadPersistedValues = false,
  persistKey,
}: { persist?: boolean; loadPersistedValues?: boolean; persistKey?: string } = {}) => {
  const [searchParams, setSearchParams] = useSearchParams();

  const [localStorageFilters, setLocalStorageFilters] = useLocalStorage<TableFilter[] | undefined>({
    initialValue: undefined,
    key: `traces_useFilters_${persistKey}`,
    version: 1,
  });

  const isEmptySearchParams = useMemo(() => !searchParams.get(QUERY_PARAM_KEY), [searchParams]);

  const filters: TableFilter[] = useMemo(() => {
    const filtersUrl = searchParams.getAll(QUERY_PARAM_KEY) ?? [];

    return filtersUrl.reduce<TableFilter[]>((filters, urlFilter) => {
      const [column, urlOperator, value, key] = urlFilter.split(VALUE_SEPARATOR);
      // IS NULL / IS NOT NULL operators don't require a value
      if (!column || !urlOperator || (!value && !isNullOperator(urlOperator))) return filters;

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
      if (persist) {
        setLocalStorageFilters(newFilters);
      }
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
    [setSearchParams, setLocalStorageFilters, persist],
  );

  useEffect(() => {
    if (isEmptySearchParams && loadPersistedValues && localStorageFilters) {
      setFilters(localStorageFilters, true);
    }
    // Rehydrate from local storage only once
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadPersistedValues]);

  return [filters, setFilters] as const;
};
