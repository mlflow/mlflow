import { useCallback, useMemo } from 'react';
import { FormattedMessage } from 'react-intl';
import { Link, useNavigate, useParams, useSearchParams } from '../../../common/utils/RoutingUtils';
import { RunPage } from '../../components/run-page/RunPage';
import { ExperimentPageTabName, RunPageTabName, MLFLOW_ISSUE_DETECTION_JOB_ID_TAG } from '../../constants';
import Routes from '../../routes';
import { getTimeRangeQueryString } from '../experiment-page-tabs/side-nav/utils';
import { useGetExperimentQuery } from '../../hooks/useExperimentQuery';
import { IssueDetectionRunOverview } from '../../components/run-page/overview/IssueDetectionRunOverview';

/**
 * Thin wrapper around RunPage for issue detection run details.
 * Customizes breadcrumbs and tab navigation to stay within /evaluation-runs/ routes.
 */
export const IssueDetectionRunDetailsPage = () => {
  const { experimentId } = useParams<{ experimentId: string }>();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const safeExperimentId = experimentId as string;
  const timeRangeSearch = useMemo(() => getTimeRangeQueryString(searchParams.toString()), [searchParams]);

  // Fetch experiment data for breadcrumb
  const { data: experiment } = useGetExperimentQuery({ experimentId: safeExperimentId });

  // Helper to append time-range search params to a route
  const withSearchParams = useCallback(
    (route: string) => (timeRangeSearch ? `${route}${timeRangeSearch}` : route),
    [timeRangeSearch],
  );

  const customBreadcrumbs = useMemo(() => {
    const experimentName = experiment?.name ?? safeExperimentId;
    const evalRunsRoute = Routes.getExperimentPageTabRoute(safeExperimentId, ExperimentPageTabName.EvaluationRuns);
    return [
      <Link
        componentId="mlflow.experiment_tracking.issue_detection.breadcrumb_experiments_link"
        key="experiments"
        to={Routes.experimentsObservatoryRoute}
      >
        <FormattedMessage
          defaultMessage="Experiments"
          description="Issue detection run details > Breadcrumb > Experiments"
        />
      </Link>,
      <Link
        componentId="mlflow.experiment_tracking.issue_detection.breadcrumb_experiment_link"
        key="experiment"
        to={Routes.getExperimentPageRoute(safeExperimentId)}
      >
        {experimentName}
      </Link>,
      <Link
        componentId="mlflow.experiment_tracking.issue_detection.breadcrumb_evaluation_runs_link"
        key="evaluation-runs"
        to={{ pathname: evalRunsRoute, search: timeRangeSearch }}
      >
        <FormattedMessage
          defaultMessage="Evaluation runs"
          description="Issue detection run details > Breadcrumb > Evaluation runs"
        />
      </Link>,
    ];
  }, [experiment?.name, safeExperimentId, timeRangeSearch]);

  const handleDeleteSuccess = useCallback(
    (expId: string) => {
      navigate(withSearchParams(Routes.getExperimentPageTabRoute(expId, ExperimentPageTabName.EvaluationRuns)));
    },
    [navigate, withSearchParams],
  );

  return (
    <RunPage
      customBreadcrumbs={customBreadcrumbs}
      tabSwitchProps={{
        getBaseRoute: Routes.getIssueDetectionRunDetailsRoute,
        getTabRoute: Routes.getIssueDetectionRunDetailsTabRoute,
        visibleTabs: [RunPageTabName.OVERVIEW, RunPageTabName.ISSUES, RunPageTabName.TRACES],
      }}
      onDeleteSuccess={handleDeleteSuccess}
      hideTracesCompareSelector
      renderCustomOverview={({ runInfo, tags, onRunDataUpdated }) => (
        <IssueDetectionRunOverview
          runInfo={runInfo}
          tags={tags}
          jobId={tags[MLFLOW_ISSUE_DETECTION_JOB_ID_TAG]?.value}
          onRunDataUpdated={onRunDataUpdated}
        />
      )}
    />
  );
};

export default IssueDetectionRunDetailsPage;
