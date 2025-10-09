import { useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { useState } from 'react';

export const OPERATORS = ['IS', 'IS NOT', 'CONTAINS'] as const;
type Operator = typeof OPERATORS[number];
export type TagFilter = { key: string; operator: Operator; value: string };

function isOperator(value: string): value is Operator {
  return (OPERATORS as readonly string[]).includes(value);
}

function serialize(tagFilter: TagFilter) {
  return [tagFilter.key, tagFilter.operator, tagFilter.value].join('-');
}

function deserialize(value: string) {
  const split = value.split('-');
  if (split.length >= 3 && isOperator(split[1])) {
    // NOTE: key may not have dashes in it, but value may, so we'll join the rest
    const [key, operator, ...valueParts] = split;
    return { key, operator, value: valueParts.join('-') } satisfies TagFilter;
  } else {
    return null;
  }
}

export function useTagsFilter() {
  const [isTagsFilterOpen, setIsTagsFilterOpen] = useState(false);

  const name = 'experimentTagsFilter';
  const [searchParams, setSearchParams] = useSearchParams();

  const tagsFilter = (searchParams.getAll(name) ?? []).map(deserialize).filter((tagFilter) => tagFilter !== null);

  function setTagsFilter(tagsFilter: TagFilter[]) {
    searchParams.delete(name);

    const filtered = tagsFilter.filter((tagFilter) => tagFilter.key !== '' && tagFilter.value !== '');

    if (filtered.length !== 0) {
      for (const tagFilter of filtered) {
        searchParams.append(name, serialize(tagFilter));
      }
    }
    setSearchParams(searchParams);
    setIsTagsFilterOpen(false);
  }

  return { tagsFilter, setTagsFilter, isTagsFilterOpen, setIsTagsFilterOpen };
}
