import { Link } from '../../../../common/utils/RoutingUtils';
import Utils from '../../../../common/utils/Utils';
import Routes from '../../../routes';
import type { RunInfoEntity } from '../../../types';
import type { KeyValueEntity } from '../../../../common/types';
import type { UseGetRunQueryResponseRunInfo } from '../hooks/useGetRunQuery';

export const RunViewUserLinkBox = ({
  runInfo,
  tags,
}: {
  runInfo: RunInfoEntity | UseGetRunQueryResponseRunInfo;
  tags: Record<string, KeyValueEntity>;
}) => {
  const user = Utils.getUser(runInfo, tags);
  return <Link to={Routes.searchRunsByUser(runInfo?.experimentId ?? '', user)}>{user}</Link>;
};
