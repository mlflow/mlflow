import { useCallback, useEffect, useState } from 'react';
import { Button, ParagraphSkeleton, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';
import { MlflowService } from '../../../sdk/MlflowService';
import type { RunInfoEntity } from '../../../types';
import { EXPERIMENT_PARENT_ID_TAG } from '../../experiment-page/utils/experimentPage.common-utils';

const EmptyValue = () => <Typography.Hint>—</Typography.Hint>;

const PAGE_SIZE = 10;

export const RunViewChildRunsBox = ({ runUuid, experimentId }: { runUuid: string; experimentId: string }) => {
  const { theme } = useDesignSystemTheme();
  const [childRuns, setChildRuns] = useState<RunInfoEntity[] | undefined>();
  const [nextPageToken, setNextPageToken] = useState<string | undefined>();
  const [isLoading, setIsLoading] = useState(false);
  const [hasError, setHasError] = useState(false);

  const loadChildRuns = useCallback(
    async (pageToken?: string) => {
      setIsLoading(true);
      try {
        const res = await MlflowService.searchRuns({
          experiment_ids: [experimentId],
          filter: `tags.\`${EXPERIMENT_PARENT_ID_TAG}\` = '${runUuid}'`,
          order_by: ['attributes.start_time DESC'],
          max_results: PAGE_SIZE,
          page_token: pageToken,
        });
        const infos = res.runs?.map((r: any) => r.info) || [];
        setChildRuns((prev = []) => [...prev, ...infos]);
        setNextPageToken(res.next_page_token);
        setHasError(false);
      } catch {
        setHasError(true);
      } finally {
        setIsLoading(false);
      }
    },
    [experimentId, runUuid],
  );

  useEffect(() => {
    setChildRuns(undefined);
    setNextPageToken(undefined);
    loadChildRuns();
  }, [loadChildRuns]);

  if (hasError) {
    return (
      <Typography.Text color="error">
        <FormattedMessage
          defaultMessage="Failed to load child runs"
          description="Run page > Overview > Child runs error"
        />
      </Typography.Text>
    );
  }

  if (childRuns === undefined) {
    return (
      <ParagraphSkeleton
        loading
        label={
          <FormattedMessage
            defaultMessage="Child runs loading"
            description="Run page > Overview > Child runs loading"
          />
        }
      />
    );
  }

  if (childRuns.length === 0) {
    return <EmptyValue />;
  }

  return (
    <div>
      <div
        css={{
          display: 'flex',
          flexDirection: 'row',
          flexWrap: 'wrap',
          gap: theme.spacing.sm,
          padding: `${theme.spacing.sm}px 0`,
        }}
      >
        {childRuns.map((info, index) => (
          <Typography.Text key={info.runUuid} css={{ whiteSpace: 'nowrap' }}>
            <Link to={Routes.getRunPageRoute(info.experimentId, info.runUuid)}>{info.runName}</Link>
            {index < childRuns.length - 1 && ','}
          </Typography.Text>
        ))}
      </div>
      {nextPageToken && (
        <div css={{ marginBottom: theme.spacing.sm }}>
          <Button
            componentId="mlflow.run_details.overview.child_runs.load_more_button"
            size="small"
            type="primary"
            onClick={() => loadChildRuns(nextPageToken)}
            loading={isLoading}
          >
            <FormattedMessage defaultMessage="Load more" description="Run page > Overview > Child runs load more" />
          </Button>
        </div>
      )}
    </div>
  );
};
