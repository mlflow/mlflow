import { useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';

export function useSearchFilter() {
  const name = 'experimentSearchFilter';
  const [searchParams, setSearchParams] = useSearchParams();

  const searchFilter = searchParams.get(name) ?? '';

  function setSearchFilter(searchFilter: string) {
    if (!searchFilter) {
      searchParams.delete(name);
    } else {
      searchParams.set(name, searchFilter);
    }
    setSearchParams(searchParams);
  }

  return [searchFilter, setSearchFilter] as const;
}
