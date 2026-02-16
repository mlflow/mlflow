import { useCallback, useMemo } from 'react';

import type { AssessmentFilter, AssessmentInfo, AssessmentValueType } from '../types';
import { useSearchParams } from '../utils/RoutingUtils';

const QUERY_PARAM_KEY = 'assessmentFilter';
const VALUE_SEPARATOR = '::';

/**
 * Query param-powered hook that returns the compare to run uuid when comparison is enabled.
 */
export const useAssessmentFilters = (assessmentInfos: AssessmentInfo[]) => {
  const [searchParams, setSearchParams] = useSearchParams();

  const assessmentFilters: AssessmentFilter[] = useMemo(() => {
    const assessmentFiltersUrl = searchParams.getAll(QUERY_PARAM_KEY) ?? [];

    return assessmentFiltersUrl.reduce<AssessmentFilter[]>((filters, urlFilter) => {
      const [run, assessmentName, filterValueString, filterType] = urlFilter.split(VALUE_SEPARATOR);
      const assessmentInfo = assessmentInfos?.find((info) => info.name === assessmentName);
      if (assessmentInfo) {
        const filterValue = serializedStringToAssessmentValue(assessmentInfo, filterValueString);
        filters.push({
          run,
          assessmentName,
          filterValue,
          filterType: filterType === '' ? undefined : filterType,
        } as AssessmentFilter);
      }
      return filters;
    }, []);
  }, [assessmentInfos, searchParams]);

  const setAssessmentFilters = useCallback(
    (filters: AssessmentFilter[] | undefined, replace = false) => {
      setSearchParams(
        (params: URLSearchParams) => {
          params.delete(QUERY_PARAM_KEY);

          if (filters) {
            filters.forEach((filter) => {
              params.append(
                QUERY_PARAM_KEY,
                [
                  filter.run,
                  filter.assessmentName,
                  assessmentValueToSerializedString(filter.filterValue),
                  filter.filterType,
                ].join(VALUE_SEPARATOR),
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

  return [assessmentFilters, setAssessmentFilters] as const;
};

export function serializedStringToAssessmentValueV2(value: string): AssessmentValueType {
  if (value === 'undefined') {
    return undefined;
  }

  // Handle boolean values
  if (value === 'true') {
    return true;
  }
  if (value === 'false') {
    return false;
  }

  // TODO(nsthorat): handle float / int types here.
  return value;
}

export function serializedStringToAssessmentValue(assessmentInfo: AssessmentInfo, value: string): AssessmentValueType {
  if (assessmentInfo.dtype === 'pass-fail') {
    if (value === 'undefined') {
      return undefined;
    }
    return value;
  } else if (assessmentInfo.dtype === 'boolean') {
    if (value === 'true') {
      return true;
    } else if (value === 'false') {
      return false;
    } else {
      return undefined;
    }
  }
  // TODO(nsthorat): handle float / int types here.
  return value;
}

export function assessmentValueToSerializedString(value: AssessmentValueType): string {
  if (value === undefined || value === null) return 'undefined';
  return `${value}`;
}
