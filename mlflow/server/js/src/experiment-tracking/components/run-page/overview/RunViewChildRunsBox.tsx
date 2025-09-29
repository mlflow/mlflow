import { useCallback, useEffect, useState } from 'react';
import { ParagraphSkeleton, Typography } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';
import { ExperimentPageTabName } from '../../../constants';
import { MlflowService } from '../../../sdk/MlflowService';
import { EXPERIMENT_PARENT_ID_TAG } from '../../experiment-page/utils/experimentPage.common-utils';

const EmptyValue = () => <Typography.Hint>—</Typography.Hint>;

const CHILD_RUN_CHECK_LIMIT = 1;

export const RunViewChildRunsBox = ({ runUuid, experimentId }: { runUuid: string; experimentId: string }) => {
  const [hasChildRuns, setHasChildRuns] = useState<boolean | undefined>();
  const [hasError, setHasError] = useState(false);

  const loadChildRuns = useCallback(async () => {
    try {
      const res = await MlflowService.searchRuns({
        experiment_ids: [experimentId],
        filter: `tags.\`${EXPERIMENT_PARENT_ID_TAG}\` = '${runUuid}'`,
        order_by: ['attributes.start_time DESC'],
        max_results: CHILD_RUN_CHECK_LIMIT,
      });
      const hasResults = Boolean(res.runs?.length);
      setHasChildRuns(hasResults);
      setHasError(false);
    } catch {
      setHasError(true);
    }
  }, [experimentId, runUuid]);

  useEffect(() => {
    setHasChildRuns(undefined);
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

  if (hasChildRuns === undefined) {
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

  if (!hasChildRuns) {
    return <EmptyValue />;
  }

  const filter = `tags.\`${EXPERIMENT_PARENT_ID_TAG}\` = '${runUuid}'`;
  const childRunsSearchLink = `${Routes.getExperimentPageTabRoute(
    experimentId,
    ExperimentPageTabName.Runs,
  )}?searchFilter=${encodeURIComponent(filter)}`;

  return (
    <Typography.Text>
      <Link to={childRunsSearchLink}>
        <FormattedMessage
          defaultMessage="View all child runs in the experiment page"
          description="Run page > Overview > Child runs link"
        />
      </Link>
    </Typography.Text>
  );
};
