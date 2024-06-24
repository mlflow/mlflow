import { Link } from '../../../../common/utils/RoutingUtils';
import Utils from '../../../../common/utils/Utils';
import Routes from '../../../routes';
import type { KeyValueEntity, RunInfoEntity } from '../../../types';

export const RunViewUserLinkBox = ({
  runInfo,
  tags,
}: {
  runInfo: RunInfoEntity;
  tags: Record<string, KeyValueEntity>;
}) => {
  const user = Utils.getUser(runInfo, tags);
  return <Link to={Routes.searchRunsByUser(runInfo.experimentId, user)}>{user}</Link>;
};
