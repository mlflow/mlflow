import { useMemo } from 'react';
import { useSearchParamSelector, type Location } from '@mlflow/mlflow/src/common/utils/RoutingUtils';

const withoutSpan = (params: URLSearchParams) => {
  params.delete('spanId');
  return params.toString();
};

export const selectTracesV4PageLocation = ({ pathname, search, hash }: Location) => ({
  pathname,
  search: `?${withoutSpan(new URLSearchParams(search))}`,
  hash,
});

/** Span selection belongs to the drawer; the table and saved views do not consume it. */
export const useTracesV4SearchParams = () => {
  const [query, setParams] = useSearchParamSelector(withoutSpan);
  return [useMemo(() => new URLSearchParams(query), [query]), setParams] as const;
};
