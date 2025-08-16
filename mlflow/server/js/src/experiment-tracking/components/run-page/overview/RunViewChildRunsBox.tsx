import { useEffect, useState } from 'react';
import { ParagraphSkeleton, Typography, useDesignSystemTheme } from '@databricks/design-system';
import Routes from '../../../routes';
import { FormattedMessage } from 'react-intl';
import { MlflowService } from '../../../sdk/MlflowService';
import { ViewType } from '../../../sdk/MlflowEnums';
import { Link } from '../../../../common/utils/RoutingUtils';

export const RunViewChildRunsBox = ({
  parentRunUuid,
  experimentId,
}: {
  parentRunUuid: string;
  experimentId: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const [childRuns, setChildRuns] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchChildRuns = async () => {
      setLoading(true);
      setError(null);

      try {
        // Search for child runs using the parent run ID tag
        const response = await MlflowService.searchRuns({
          experiment_ids: [experimentId],
          filter: `tags.mlflow.parentRunId = '${parentRunUuid}'`,
          run_view_type: ViewType.ACTIVE_ONLY,
          max_results: 10, // Show most recent 10 runs
          order_by: ['attributes.start_time DESC'], // Most recent first
        });

        if (response.runs) {
          setChildRuns(response.runs);
        }
      } catch (err) {
        console.error('Error fetching child runs:', err);
        setError('Failed to load child runs');
      } finally {
        setLoading(false);
      }
    };

    fetchChildRuns();
  }, [parentRunUuid, experimentId]);

  if (loading) {
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

  if (error) {
    return (
      <Typography.Text color="error" size="sm">
        {error}
      </Typography.Text>
    );
  }

  if (!childRuns || childRuns.length === 0) {
    return (
      <Typography.Text color="secondary" size="sm">
        <FormattedMessage
          defaultMessage="No child runs found"
          description="Run page > Overview > No child runs found"
        />
      </Typography.Text>
    );
  }

  return (
    <div>
      <Typography.Text color="secondary" size="sm" css={{ marginBottom: theme.spacing.sm, display: 'block' }}>
        <FormattedMessage
          defaultMessage="Found {count} child run{count, plural, one {} other {s}}"
          description="Run page > Overview > Child runs count"
          values={{ count: childRuns.length }}
        />
      </Typography.Text>
      {childRuns.map((childRun, index) => (
        <div key={childRun.info?.runUuid || index} css={{ marginBottom: theme.spacing.xs }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <Link to={Routes.getRunPageRoute(experimentId, childRun.info?.runUuid)}>
              {childRun.info?.runName || childRun.info?.runUuid}
            </Link>
            <Typography.Text color="secondary" size="sm">
              ({childRun.info?.status || 'UNKNOWN'})
            </Typography.Text>
          </div>
        </div>
      ))}
      {childRuns.length === 10 && (
        <Typography.Text color="secondary" size="sm" css={{ marginTop: theme.spacing.xs }}>
          <FormattedMessage
            defaultMessage="Showing most recent 10 child runs (may be more available)"
            description="Run page > Overview > Child runs limit note"
          />
        </Typography.Text>
      )}
    </div>
  );
};
