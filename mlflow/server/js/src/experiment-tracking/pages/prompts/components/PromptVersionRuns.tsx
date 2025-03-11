import { useState } from 'react';
import { Button, ParagraphSkeleton, Typography } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link } from 'react-router-dom';
import Routes from '../../../routes';



export const PromptVersionRuns = ({
  isLoadingRuns,
  runIds,
  runInfoMap
}: {
  isLoadingRuns: boolean;
  runIds: string[];
  runInfoMap: Record<string, any>;
}) => {
  const [showAll, setShowAll] = useState(false);

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

      <div >
        {isLoadingRuns ? (
          <ParagraphSkeleton css={{ width: 100 }} />
        ) : (
          <>
            <div style={{ display: 'flex', flexWrap: 'wrap' }}>
              {runIds.slice(0, visibleCount).map((runId) => {
                const runInfo = runInfoMap[runId];

                if (runInfo?.experimentId && runInfo?.runUuid && runInfo?.runName) {
                  return <PromptVersionRunCell runInfo={runInfo}/>
                } else {
                  return <span>{runInfo?.runName || runInfo?.runUuid}</span>
                }
              })}
              {hasMore && (
                <Button
                  componentId="mlflow.prompts.details.runs.show_more"
                  size="small"
                  type="link"
                  onClick={() => setShowAll(!showAll)}
                >
                  {showAll ? 'Show less' : `${runIds.length - visibleCount} more...`}
                </Button>
              )}
            </div>
          </>
        )}
      </div>
    </>
  );
};

const PromptVersionRunCell = ({
  runInfo,
} : {
  runInfo: any;
}) => {
  const style = {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '0px 8px',
    borderRadius: '12px',
    marginRight: '6px',
    marginLeft: '-2px',
    margin: '0px 8px 4px -4px',
    fontSize: '13px',
    border: '1px solid #D1D9E1',
  };

  const { runName, runUuid, experimentId } = runInfo;
  const to = Routes.getRunPageRoute(experimentId, runUuid);
  return (
    <Link to={to} style={style}>
      {runName}
    </Link>
  );
}