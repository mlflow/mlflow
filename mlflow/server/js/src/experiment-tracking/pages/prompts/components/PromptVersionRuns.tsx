import { useState } from 'react';
import { Button, ParagraphSkeleton, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';
import { isNil } from 'lodash';

export const PromptVersionRuns = ({
  isLoadingRuns,
  runIds,
  runInfoMap,
}: {
  isLoadingRuns: boolean;
  runIds: string[];
  runInfoMap: Record<string, any>;
}) => {
  const [showAll, setShowAll] = useState(false);
  const { theme } = useDesignSystemTheme();

  const displayThreshold = 3;
  const visibleCount = showAll ? runIds.length : Math.min(displayThreshold, runIds.length || 0);
  const hasMore = runIds.length > displayThreshold;

  return (
    <>
      <Typography.Text bold>
        <FormattedMessage
          defaultMessage="MLflow runs:"
          description="A label for the associated MLflow runs in the prompt details page"
        />
      </Typography.Text>

      <div>
        {isLoadingRuns ? (
          <ParagraphSkeleton css={{ width: 100 }} />
        ) : (
          <>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.sm }}>
              {runIds.slice(0, visibleCount).map((runId, index) => {
                const runInfo = runInfoMap[runId];

                if (!isNil(runInfo?.experimentId) && runInfo?.runUuid && runInfo?.runName) {
                  const { experimentId, runUuid, runName } = runInfo;
                  return (
                    // eslint-disable-next-line react/jsx-key
                    <Typography.Text>
                      <Link to={Routes.getRunPageRoute(experimentId, runUuid)}>{runName}</Link>
                      {index < visibleCount - 1 && ','}
                    </Typography.Text>
                  );
                } else {
                  // eslint-disable-next-line react/jsx-key
                  return <span>{runInfo?.runName || runInfo?.runUuid}</span>;
                }
              })}
              {hasMore && (
                <Button
                  componentId="mlflow.prompts.details.runs.show_more"
                  size="small"
                  type="link"
                  onClick={() => setShowAll(!showAll)}
                >
                  {showAll ? (
                    <FormattedMessage
                      defaultMessage="Show less"
                      description="Label for a link that shows less tags when clicked"
                    />
                  ) : (
                    <FormattedMessage
                      defaultMessage="{count} more..."
                      description="Label for a link that renders the remaining tags when clicked"
                      values={{ count: runIds.length - visibleCount }}
                    />
                  )}
                </Button>
              )}
            </div>
          </>
        )}
      </div>
    </>
  );
};
