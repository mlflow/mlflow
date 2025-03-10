import { ParagraphSkeleton, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { Link } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';
import { usePromptSourceRunsInfo } from '../hooks/usePromptSourceRunsInfo';
import { RegisteredPromptVersion } from '../types';

export const PromptDetailsSourceRunsBox = ({ promptVersions }: { promptVersions?: RegisteredPromptVersion[] }) => {
  const { isLoading, sourceRunInfos } = usePromptSourceRunsInfo(promptVersions);
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ display: 'inline-flex', flexWrap: 'wrap', gap: theme.spacing.xs }}>
      {isLoading ? (
        <ParagraphSkeleton css={{ width: 200 }} />
      ) : (
        sourceRunInfos?.map((sourceRunInfo, index) => (
          <Typography.Text key={sourceRunInfo?.runUuid}>
            {sourceRunInfo?.experimentId && sourceRunInfo?.runUuid && sourceRunInfo?.runName ? (
              <Link to={Routes.getRunPageRoute(sourceRunInfo.experimentId, sourceRunInfo.runUuid)}>
                {sourceRunInfo.runName}
              </Link>
            ) : (
              <>{sourceRunInfo?.runName || sourceRunInfo?.runUuid}</>
            )}
            {index < sourceRunInfos.length - 1 && ','}
          </Typography.Text>
        ))
      )}
    </div>
  );
};
