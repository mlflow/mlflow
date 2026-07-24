import { useSearchParams as useRouterSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';

/**
 * Universe-ported v2 calls `useSearchParams((params) => selector)` as a read-tuple. OSS's
 * `useSearchParams` is the unmodified react-router one (returns `[URLSearchParams, setter]`),
 * so v2 needs this shim to keep the same call sites working. Returns `[selectedValue,
 * setSearchParams]` where the setter is the same react-router setter.
 */
export function useSearchParamSelector<T>(
  selector: (params: URLSearchParams) => T,
): [T, ReturnType<typeof useRouterSearchParams>[1]] {
  const [params, setParams] = useRouterSearchParams();
  return [selector(params), setParams];
}
