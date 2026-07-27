import { Typography, useDesignSystemTheme } from '@databricks/design-system';

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
  const { theme } = useDesignSystemTheme();
  const user = Utils.getUser(runInfo, tags);
  if (!user) {
    return <Typography.Hint css={{ padding: `${theme.spacing.xs}px 0px` }}>—</Typography.Hint>;
  }
  return (
    <Link
      componentId="mlflow.run_page.overview.user_link"
      to={Routes.searchRunsByUser(runInfo?.experimentId ?? '', user)}
    >
      {user}
    </Link>
  );
};
