import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Alert, Button, Empty, Typography, TableSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { NotFoundError, PermissionError } from '@databricks/web-shared/errors';
import { FormattedMessage } from 'react-intl';
import { PageNotFoundView } from '../../../common/components/PageNotFoundView';
import { shouldEnableExperimentIssuesTab } from '../../../common/utils/FeatureUtils';
import { useParams, useSearchParams } from '../../../common/utils/RoutingUtils';

import { useActiveWorkspace } from '../../../workspaces/utils/WorkspaceUtils';
import { IssueDetectionModal } from '../../components/experiment-page/components/traces-v3/IssueDetectionModal';
import { JobStatus, isJobComplete } from '../../components/run-page/hooks/useFetchJobStatus';
import { useCancelJob } from '../../components/run-page/hooks/useCancelJob';
import type { Issue } from '../../components/run-page/hooks/useSearchIssuesQuery';
import { SELECTED_EVALUATION_ID_PARAM } from '../../../shared/web-shared/genai-traces-table';
import { SELECTED_ISSUE_ID_PARAM, SELECTED_ISSUE_SPAN_ID_PARAM } from '../../constants';
import { SERVER_FEATURE_KEYS, useFeatureEnabled, useServerInfo } from '../../hooks/useServerInfo';
import { ExperimentIssueDetail } from './ExperimentIssueDetail';
import { ExperimentIssueRow } from './ExperimentIssueRow';
import { ExperimentIssuesFilterChips, ExperimentIssuesSortControl } from './ExperimentIssuesToolbar';
import { useIssueListControls } from './useExperimentIssuesListControls';
import { ExperimentIssuesActionBar } from './ExperimentIssuesActionBar';
import { IssueSplitPane } from './IssueSplitPane';
import { markDetectIssuesGuidanceSeen } from '../../../shared/web-shared/genai-traces-table/components/DetectIssuesButton';
import { ExperimentIssuesDetectionProgress } from './ExperimentIssuesDetectionProgress';
import { ExperimentIssuesEmptyState } from './ExperimentIssuesEmptyState';
import { useDetectionTraceIds, useLatestIssueDetection } from './useIssueDetection';
import { useExperimentIssue, useExperimentIssues, useHasExperimentIssues } from './useExperimentIssues';
import { IssueTraceDrawerProviders } from './IssueTraceDrawer';

const CenteredState = ({ children }: { children: ReactNode }) => (
  <div
    css={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: 400,
      height: '100%',
      width: '100%',
      flex: 1,
      '& > div': {
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
      },
    }}
  >
    {children}
  </div>
);

const ReadError = ({ error, retry, message }: { error?: Error; retry: () => void; message?: ReactNode }) => (
  <CenteredState>
    <div>
      <Alert
        componentId="mlflow.experiment-issues.read-error"
        type="error"
        closable={false}
        message={
          message ??
          (error instanceof PermissionError ? (
            <FormattedMessage
              defaultMessage="You do not have permission to view these issues"
              description="Experiment Issues permission error"
            />
          ) : error instanceof NotFoundError ? (
            <FormattedMessage
              defaultMessage="Issue not found in this experiment"
              description="Experiment Issue missing selection error"
            />
          ) : (
            <FormattedMessage defaultMessage="Unable to load issues" description="Experiment Issues load error" />
          ))
        }
        description={error?.message}
      />
      <Button componentId="mlflow.experiment-issues.retry" onClick={retry}>
        <FormattedMessage defaultMessage="Retry" description="Experiment Issues read retry action" />
      </Button>
    </div>
  </CenteredState>
);

const ExperimentIssuesContent = ({ experimentId }: { experimentId: string }) => {
  const { theme } = useDesignSystemTheme();
  const controls = useIssueListControls();
  const search = useExperimentIssues(experimentId, controls.filters);
  const issues = useMemo(() => {
    // Later pages update an Issue's data while keeping its first-seen position, so loading more
    // does not reshuffle existing rows. Refreshing the list establishes a new sort order.
    const byId = new Map<string, Issue>();
    search.data?.pages.forEach((page) => page.issues?.forEach((issue) => byId.set(issue.issue_id, issue)));
    return Array.from(byId.values());
  }, [search.data]);
  const hasRestrictiveFilters = Boolean(
    controls.filters.name ||
    controls.filters.statuses.length ||
    controls.filters.severities.length ||
    controls.filters.categories.length,
  );
  const needsExistenceCheck = search.isSuccess && !issues.length && hasRestrictiveFilters;
  const anyIssue = useHasExperimentIssues(experimentId, needsExistenceCheck);
  const existenceLoading = needsExistenceCheck && anyIssue.isLoading;
  const existenceError = needsExistenceCheck ? anyIssue.error : null;
  const detection = useLatestIssueDetection(experimentId);
  const isDetecting = Boolean(detection.run) && !isJobComplete(detection.status);
  const detectionEnabled = useFeatureEnabled(SERVER_FEATURE_KEYS.GATEWAY);
  const [showDetection, setShowDetection] = useState(false);
  const traces = useDetectionTraceIds(experimentId, showDetection);
  const isDetectionButtonLoading = isDetecting || (showDetection && traces.isFetching);
  const { cancelJobAsync } = useCancelJob();
  // Gated on a run actually being in flight, not merely on a job ID existing: a completed run
  // keeps its job ID, so gating on that alone would offer to cancel a terminal job whenever the
  // button went busy again -- which it does while the modal loads traces for the next run.
  const onCancelDetection =
    isDetecting && detection.jobId
      ? async () => {
          try {
            // The run is passed so the hook terminates it too: the server cancels the job
            // without touching the run it belongs to, which would otherwise stay RUNNING.
            await cancelJobAsync({ jobId: detection.jobId as string, runUuid: detection.run?.info.run_id });
          } finally {
            // Refreshed even when the cancel fails, for the reason given in
            // `TracesV4PageContent`: "already finalized" is precisely the case where this
            // client's job status is stale, so the refresh is what recovers the page.
            await detection.refetch();
          }
        }
      : undefined;
  const [params, setParams] = useSearchParams();
  const requestedIssueId = params.get(SELECTED_ISSUE_ID_PARAM) ?? undefined;
  const firstIssueId = issues[0]?.issue_id;
  const selectedId = requestedIssueId ?? firstIssueId;
  const detail = useExperimentIssue(experimentId, selectedId);

  useEffect(() => {
    if (!search.isLoading && !search.error && !requestedIssueId && firstIssueId) {
      setParams(
        (current) => {
          current.set(SELECTED_ISSUE_ID_PARAM, firstIssueId);
          current.delete(SELECTED_EVALUATION_ID_PARAM);
          current.delete(SELECTED_ISSUE_SPAN_ID_PARAM);
          return current;
        },
        { replace: true },
      );
    }
  }, [search.isLoading, search.error, requestedIssueId, firstIssueId, setParams]);

  // A new title search selects its first match. Filters retain the selected Issue's detail.
  const searchedQueryRef = useRef(controls.filters.name);
  useEffect(() => {
    if (search.isLoading || search.error || searchedQueryRef.current === controls.filters.name) return;
    searchedQueryRef.current = controls.filters.name;
    if (selectedId && !issues.some((issue) => issue.issue_id === selectedId)) {
      setParams(
        (current) => {
          if (firstIssueId) current.set(SELECTED_ISSUE_ID_PARAM, firstIssueId);
          else current.delete(SELECTED_ISSUE_ID_PARAM);
          current.delete(SELECTED_EVALUATION_ID_PARAM);
          current.delete(SELECTED_ISSUE_SPAN_ID_PARAM);
          return current;
        },
        { replace: true },
      );
    }
  }, [search.isLoading, search.error, controls.filters.name, issues, selectedId, firstIssueId, setParams]);

  // A trace selection travels in the URL, next to the selected Issue, so a copied link reopens
  // the same trace on the same span.
  const viewTrace = (linkableTraceId: string, spanId?: string) =>
    setParams((current) => {
      current.set(SELECTED_EVALUATION_ID_PARAM, linkableTraceId);
      if (spanId) {
        current.set(SELECTED_ISSUE_SPAN_ID_PARAM, spanId);
      } else {
        current.delete(SELECTED_ISSUE_SPAN_ID_PARAM);
      }
      return current;
    });

  const selectIssue = (issueId: string) =>
    setParams((current) => {
      current.set(SELECTED_ISSUE_ID_PARAM, issueId);
      // Another Issue's evidence names other traces, so a trace left open from the previous one
      // would keep its drawer mounted over a selection the new Issue does not contain.
      current.delete(SELECTED_EVALUATION_ID_PARAM);
      current.delete(SELECTED_ISSUE_SPAN_ID_PARAM);
      return current;
    });
  const total = search.data?.pages[0]?.total_count;
  const hasAny = issues.length > 0 || (needsExistenceCheck && Boolean(anyIssue.data?.issues?.length));
  const cleanScan = detection.status === JobStatus.SUCCEEDED && detection.result?.issues === 0;
  const isEmptyState =
    !requestedIssueId &&
    !search.isLoading &&
    !search.error &&
    !issues.length &&
    !hasAny &&
    !existenceLoading &&
    !existenceError;
  const isResolvingEmptyState =
    !requestedIssueId &&
    !issues.length &&
    !hasAny &&
    !search.error &&
    !existenceError &&
    (search.isLoading || existenceLoading || detection.isLoading);
  const hasShownIssues = useRef(false);
  if (hasAny) hasShownIssues.current = true;
  const detectionError = detection.error ?? (!isJobComplete(detection.status) ? detection.job.error : undefined);
  const startDetection = () => {
    markDetectIssuesGuidanceSeen();
    setShowDetection(true);
    if (traces.error) traces.refetch();
  };
  const list = (
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0, minHeight: 0, overflow: 'hidden' }}>
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          flexShrink: 0,
          gap: theme.spacing.sm,
          padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
        }}
      >
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <Typography.Text bold>
            <FormattedMessage
              defaultMessage="Issues"
              description="Experiment issues tab > Heading of the issue list column"
            />
          </Typography.Text>
          {total !== undefined && (
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="{count, plural, one {# result} other {# results}}"
                description="Experiment issues tab > Number of issues matching the search and filters"
                values={{ count: total }}
              />
            </Typography.Text>
          )}
          <div css={{ marginLeft: 'auto' }}>
            <ExperimentIssuesSortControl controls={controls} labeled />
          </div>
        </div>
        <ExperimentIssuesFilterChips controls={controls} />
      </div>
      <div css={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
        {search.isLoading ? (
          <TableSkeleton lines={5} />
        ) : search.error && !search.data ? (
          <ReadError error={search.error} retry={() => search.refetch()} />
        ) : (
          <>
            {issues.length === 0 ? (
              <CenteredState>
                {existenceError ? (
                  <ReadError error={existenceError} retry={() => anyIssue.refetch()} />
                ) : (
                  <Empty
                    description={
                      <FormattedMessage
                        defaultMessage="No issues match these filters."
                        description="Experiment Issues no matching results"
                      />
                    }
                  />
                )}
              </CenteredState>
            ) : (
              <ul
                aria-busy={search.isFetching && !search.isFetchingNextPage}
                css={{ listStyle: 'none', margin: 0, padding: 0 }}
              >
                {issues.map((issue) => (
                  <li key={issue.issue_id}>
                    <ExperimentIssueRow
                      issue={issue}
                      isSelected={selectedId === issue.issue_id}
                      onSelect={() => selectIssue(issue.issue_id)}
                    />
                  </li>
                ))}
              </ul>
            )}
            {search.error && search.data && (
              <Alert
                componentId="mlflow.experiment-issues.page-error"
                type="error"
                closable={false}
                message={
                  <FormattedMessage
                    defaultMessage="Unable to refresh or load more issues"
                    description="Experiment Issues pagination error"
                  />
                }
                description={
                  <div>
                    {search.error.message}
                    <Button componentId="mlflow.experiment-issues.retry-refresh" onClick={() => search.refetch()}>
                      <FormattedMessage defaultMessage="Retry" description="Experiment Issues read retry action" />
                    </Button>
                  </div>
                }
              />
            )}
            {search.hasNextPage && (
              <Button
                componentId="mlflow.experiment-issues.load-more"
                loading={search.isFetchingNextPage}
                onClick={() => search.fetchNextPage()}
              >
                <FormattedMessage defaultMessage="Load more" description="Experiment Issues next page action" />
              </Button>
            )}
          </>
        )}
      </div>
    </div>
  );
  const detailPane = !selectedId ? undefined : detail.error ? (
    <ReadError error={detail.error} retry={() => detail.refetch()} />
  ) : detail.isLoading ? (
    <div css={{ flex: 1, padding: theme.spacing.lg }}>
      <TableSkeleton lines={5} />
    </div>
  ) : detail.data ? (
    <ExperimentIssueDetail key={selectedId} issue={detail.data} onViewTrace={viewTrace} />
  ) : undefined;
  return (
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, height: '100%' }}>
      {(hasShownIssues.current || (!isEmptyState && !isResolvingEmptyState)) && (
        <div css={{ flexShrink: 0, paddingBottom: theme.spacing.md, borderBottom: `1px solid ${theme.colors.border}` }}>
          <ExperimentIssuesActionBar
            experimentId={experimentId}
            controls={controls}
            searchInput={controls.search.input}
            onSearchInputChange={controls.search.setInput}
            onSearchSubmit={controls.search.flush}
            onSearchClear={controls.search.clear}
            isDetecting={isDetectionButtonLoading}
            detectionEnabled={detectionEnabled}
            onRunDetection={startDetection}
            onCancelDetection={onCancelDetection}
          />
        </div>
      )}
      {detectionError && (
        <Alert
          componentId="mlflow.experiment-issues.detection-error"
          type="warning"
          closable={false}
          message={
            detection.error ? (
              <FormattedMessage
                defaultMessage="Unable to load detection history"
                description="Issues detection history read failure"
              />
            ) : (
              <FormattedMessage
                defaultMessage="Unable to update detection progress"
                description="Issues job status error"
              />
            )
          }
          description={
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.md, flexWrap: 'wrap' }}>
              {detectionError.message}
              <Button
                componentId="mlflow.experiment-issues.retry-detection"
                size="small"
                loading={detection.isFetching || detection.job.isFetching}
                onClick={() => (detection.error ? detection.refetch() : detection.job.refetch())}
              >
                <FormattedMessage
                  defaultMessage="Retry status"
                  description="Issues detection job status retry action"
                />
              </Button>
            </div>
          }
        />
      )}
      {showDetection && traces.error && (
        <Alert
          componentId="mlflow.experiment-issues.trace-error"
          type="error"
          closable={false}
          message={
            <FormattedMessage
              defaultMessage="Unable to load traces for detection"
              description="Experiment Issues trace selection load error"
            />
          }
          description={traces.error.message}
        />
      )}
      {isResolvingEmptyState ? (
        <TableSkeleton lines={5} />
      ) : isEmptyState && isDetecting ? (
        <CenteredState>
          <ExperimentIssuesDetectionProgress experimentId={experimentId} onCancelDetection={onCancelDetection} />
        </CenteredState>
      ) : isEmptyState ? (
        <CenteredState>
          <ExperimentIssuesEmptyState
            hasCompletedDetectionRuns={detection.status === JobStatus.SUCCEEDED}
            isCleanScan={cleanScan}
            hasTimedOutDetectionRun={detection.status === JobStatus.TIMEOUT}
            hasFailedDetectionRun={detection.status === JobStatus.FAILED}
            isDetecting={isDetectionButtonLoading}
            detectionEnabled={detectionEnabled}
            onRunIssueDetection={startDetection}
          />
        </CenteredState>
      ) : (
        <IssueSplitPane list={list} detail={detailPane} />
      )}
      {showDetection && traces.data && (
        <IssueDetectionModal
          experimentId={experimentId}
          availableTraceIds={traces.data}
          onClose={() => setShowDetection(false)}
        />
      )}
    </div>
  );
};

const ExperimentIssuesPage = () => {
  const { experimentId } = useParams<{ experimentId: string }>();
  const workspace = useActiveWorkspace();
  const server = useServerInfo();
  if (server.isLoading) return <TableSkeleton lines={5} />;
  if (!shouldEnableExperimentIssuesTab()) return <PageNotFoundView />;
  if (server.isError || server.data?.store_type === undefined || server.data.store_type === '') {
    return (
      <ReadError
        error={server.error instanceof Error ? server.error : undefined}
        message={
          <FormattedMessage
            defaultMessage="Unable to load server information"
            description="Experiment Issues server information load error"
          />
        }
        retry={() => server.refetch()}
      />
    );
  }
  if (server.data?.store_type !== 'SqlStore') {
    return (
      <CenteredState>
        <Empty
          description={
            <FormattedMessage
              defaultMessage="The Issues tab requires an MLflow server with SQL storage."
              description="Experiment Issues unavailable on unsupported storage"
            />
          }
        />
      </CenteredState>
    );
  }
  if (!experimentId) return null;
  return (
    <IssueTraceDrawerProviders experimentId={experimentId}>
      <ExperimentIssuesContent key={`${workspace}:${experimentId}`} experimentId={experimentId} />
    </IssueTraceDrawerProviders>
  );
};

export default ExperimentIssuesPage;
