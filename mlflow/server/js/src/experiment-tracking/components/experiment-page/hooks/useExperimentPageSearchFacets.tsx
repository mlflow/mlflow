import { assign, entries, isNil, keys, omitBy, pick } from 'lodash';
import { useMemo } from 'react';
import type { NavigateOptions } from '../../../../common/utils/RoutingUtils';
import { useParams, useSearchParams } from '../../../../common/utils/RoutingUtils';
import type { ExperimentPageSearchFacetsState } from '../models/ExperimentPageSearchFacetsState';
import { createExperimentPageSearchFacetsState } from '../models/ExperimentPageSearchFacetsState';
import {
  deserializeFieldsFromQueryString,
  serializeFieldsToQueryString,
} from '../utils/persistSearchFacets.serializers';

export const EXPERIMENT_PAGE_QUERY_PARAM_KEYS = [
  'searchFilter',
  'orderByKey',
  'orderByAsc',
  'startTime',
  'lifecycleFilter',
  'modelVersionFilter',
  'datasetsFilter',
];

export const EXPERIMENT_PAGE_QUERY_PARAM_IS_PREVIEW = 'isPreview';

export type ExperimentPageQueryParams = any;

export type ExperimentQueryParamsSearchFacets = ExperimentPageSearchFacetsState & {
  experimentIds?: string[];
};

const getComparedExperimentIds = (comparedExperimentIds: string): string[] => {
  try {
    return comparedExperimentIds ? JSON.parse(comparedExperimentIds) : [];
  } catch {
    return [];
  }
};

export const useExperimentPageSearchFacets = (): [ExperimentQueryParamsSearchFacets | null, string[], boolean] => {
  const [queryParams] = useSearchParams();

  // Pick only the keys we care about
  const pickedValues = useMemo(
    () => pick(Object.fromEntries(queryParams.entries()), EXPERIMENT_PAGE_QUERY_PARAM_KEYS),
    [queryParams],
  );

  // Check if the page is in preview mode. If so, it should not be persisted until explicitly changed
  const isPreview = queryParams.get(EXPERIMENT_PAGE_QUERY_PARAM_IS_PREVIEW) === 'true';

  // Destructure to get raw values
  const { searchFilter, orderByKey, orderByAsc, startTime, lifecycleFilter, modelVersionFilter, datasetsFilter } =
    pickedValues;

  const areValuesEmpty = keys(pickedValues).length < 1;

  const { experimentId } = useParams<{ experimentId: string }>();
  const queryParamsExperimentIds = queryParams.get('experiments');

  // Calculate experiment IDs
  const experimentIds = useMemo(() => {
    if (experimentId) {
      return [experimentId];
    }
    if (queryParamsExperimentIds) {
      return getComparedExperimentIds(queryParamsExperimentIds);
    }
    return [];
  }, [experimentId, queryParamsExperimentIds]);

  // Calculate and memoize search facets
  const searchFacets = useMemo(() => {
    if (areValuesEmpty) {
      return null;
    }
    const deserializedFields = deserializeFieldsFromQueryString(
      omitBy(
        {
          searchFilter,
          orderByKey,
          orderByAsc,
          startTime,
          lifecycleFilter,
          modelVersionFilter,
          datasetsFilter,
        },
        isNil,
      ),
    ) as ExperimentPageSearchFacetsState;

    // If not all fields are provided, fill the gaps with default values
    return assign(createExperimentPageSearchFacetsState(), deserializedFields);
  }, [
    // Use exact values to avoid unnecessary re-renders
    searchFilter,
    orderByKey,
    orderByAsc,
    startTime,
    lifecycleFilter,
    modelVersionFilter,
    datasetsFilter,
    areValuesEmpty,
  ]);

  return [searchFacets, experimentIds, isPreview];
};

export const useUpdateExperimentPageSearchFacets = () => {
  const [, setParams] = useSearchParams();

  return (partialFacets: Partial<ExperimentPageSearchFacetsState>, options?: NavigateOptions) => {
    const newParams = serializeFieldsToQueryString(partialFacets);
    setParams((currentParams) => {
      entries(newParams).forEach(([key, value]) => {
        currentParams.set(key, value);
      });
      currentParams.delete(EXPERIMENT_PAGE_QUERY_PARAM_IS_PREVIEW);
      return currentParams;
    }, options);
  };
};
