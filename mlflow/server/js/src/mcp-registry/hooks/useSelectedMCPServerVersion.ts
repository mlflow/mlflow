import { useCallback } from 'react';
import { useSearchParams } from '../../common/utils/RoutingUtils';

const VERSION_QUERY_PARAM = 'version';

export const useSelectedMCPServerVersion = () => {
  const [searchParams, setSearchParams] = useSearchParams();

  const selectedVersion = searchParams.get(VERSION_QUERY_PARAM) ?? undefined;

  const setSelectedVersion = useCallback(
    (version: string | undefined) => {
      setSearchParams(
        (params) => {
          if (version === undefined) {
            params.delete(VERSION_QUERY_PARAM);
          } else {
            params.set(VERSION_QUERY_PARAM, version);
          }
          return params;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  return [selectedVersion, setSelectedVersion] as const;
};
