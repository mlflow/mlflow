import { assign, entries, isNil, keys, omitBy, pick } from 'lodash';
import { useMemo } from 'react';
import { NavigateOptions, useParams, useSearchParams } from '../../../../common/utils/RoutingUtils';
import {
  ExperimentPageSearchFacetsStateV2,
  createExperimentPageSearchFacetsStateV2,
} from '../models/ExperimentPageSearchFacetsStateV2';
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

export type ExperimentPageQueryParams = any;

export type ExperimentQueryParamsSearchFacets = ExperimentPageSearchFacetsStateV2 & {
  experimentIds?: string[];
};

const getComparedExperimentIds = (comparedExperimentIds: string): string[] => {
  try {
    return comparedExperimentIds ? JSON.parse(comparedExperimentIds) : [];
  } catch {
    return [];
  }
};

export const useExperimentPageSearchFacets = (): [ExperimentQueryParamsSearchFacets | null, string[]] => {
  const [queryParams] = useSearchParams();

  // Pick only the keys we care about
  const pickedValues = useMemo(
    () => pick(Object.fromEntries(queryParams.entries()), EXPERIMENT_PAGE_QUERY_PARAM_KEYS),
    [queryParams],
  );

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
    ) as ExperimentPageSearchFacetsStateV2;

    // If not all fields are provided, fill the gaps with default values
    return assign(createExperimentPageSearchFacetsStateV2(), deserializedFields);
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

  return [searchFacets, experimentIds];
};

export const useUpdateExperimentPageSearchFacets = () => {
  const [, setParams] = useSearchParams();

  return (partialFacets: Partial<ExperimentPageSearchFacetsStateV2>, options?: NavigateOptions) => {
    const newParams = serializeFieldsToQueryString(partialFacets);
    setParams((currentParams) => {
      entries(newParams).forEach(([key, value]) => {
        currentParams.set(key, value);
      });
      return currentParams;
    }, options);
  };
};
