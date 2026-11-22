import { useCallback } from 'react';
import { useSearchParams } from '../../common/utils/RoutingUtils';
import { parseSkillVersionParam } from '../utils';

const VERSION_QUERY_PARAM = 'version';

export const useSelectedSkillVersion = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedVersion = parseSkillVersionParam(searchParams.get(VERSION_QUERY_PARAM));

  const setSelectedVersion = useCallback(
    (version: number | undefined) => {
      setSearchParams(
        (params) => {
          if (version == null) {
            params.delete(VERSION_QUERY_PARAM);
          } else {
            params.set(VERSION_QUERY_PARAM, String(version));
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
