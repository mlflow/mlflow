import { useState, type ReactNode } from 'react';
import { Alert, Button, Empty, Typography, TableSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { NotFoundError, PermissionError } from '@databricks/web-shared/errors';
import { FormattedMessage } from 'react-intl';
import { PageNotFoundView } from '../../../common/components/PageNotFoundView';
import { shouldEnableExperimentIssuesTab } from '../../../common/utils/FeatureUtils';
import { useParams, useSearchParams } from '../../../common/utils/RoutingUtils';
import { useActiveWorkspace } from '../../../workspaces/utils/WorkspaceUtils';
import { IssueDetectionModal } from '../../components/experiment-page/components/traces-v3/IssueDetectionModal';
import { JobStatus, isJobComplete } from '../../components/run-page/hooks/useFetchJobStatus';
import { SELECTED_ISSUE_ID_PARAM } from '../../constants';
import { useExperimentPermission } from '../../hooks/useExperimentPermission';
import { SERVER_FEATURE_KEYS, useFeatureEnabled, useServerInfo } from '../../hooks/useServerInfo';
import { ExperimentIssueDetail } from './ExperimentIssueDetail';
import { ExperimentIssueRow } from './ExperimentIssueRow';
import { ExperimentIssuesFilterChips, ExperimentIssuesSortControl } from './ExperimentIssuesToolbar';
import { useIssueListControls } from './useExperimentIssuesListControls';
import { ExperimentIssuesActionBar } from './ExperimentIssuesActionBar';
import { IssueSplitPane } from './IssueSplitPane';
import { ExperimentIssuesEmptyState } from './ExperimentIssuesEmptyState';
import { IssueDetectionStatus } from './IssueDetectionStatus';
import { useDetectionTraceIds, useLatestIssueDetection } from './useIssueDetection';
import { useExperimentIssue, useExperimentIssues, useHasExperimentIssues } from './useExperimentIssues';

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

const ReadError = ({ error, retry }: { error: Error; retry: () => void }) => (
  <CenteredState>
    <div>
      <Alert
        componentId="mlflow.experiment-issues.read-error"
        type="error"
        closable={false}
        message={
          error instanceof PermissionError ? (
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
          )
        }
        description={error.message}
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
  const anyIssue = useHasExperimentIssues(experimentId);
  const detection = useLatestIssueDetection(experimentId);
  const isDetecting = Boolean(detection.run) && !isJobComplete(detection.status);
  const canEdit = useExperimentPermission(experimentId, ['EDIT', 'MANAGE']);
  const detectionEnabled = useFeatureEnabled(SERVER_FEATURE_KEYS.GATEWAY);
  const [showDetection, setShowDetection] = useState(false);
  const traces = useDetectionTraceIds(experimentId, showDetection);
  const isDetectionButtonLoading = isDetecting || (showDetection && traces.isFetching);
  const [params, setParams] = useSearchParams();
  const issues = search.data?.pages.flatMap((page) => page.issues ?? []) ?? [];
  const selectedId = params.get(SELECTED_ISSUE_ID_PARAM) ?? issues[0]?.issue_id;
  const detail = useExperimentIssue(experimentId, selectedId);
  const selectIssue = (issueId: string) =>
    setParams((current) => {
      current.set(SELECTED_ISSUE_ID_PARAM, issueId);
      return current;
    });
  const total = search.data?.pages[0]?.total_count;
  const hasAny = Boolean(anyIssue.data?.issues?.length);
  const cleanScan = detection.status === JobStatus.SUCCEEDED && detection.result?.issues === 0;
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
                {!hasAny &&
                !anyIssue.isLoading &&
                !detection.isLoading &&
                !anyIssue.error &&
                !detection.error &&
                (!detection.run || cleanScan) ? (
                  <ExperimentIssuesEmptyState hasCompletedDetectionRuns={cleanScan} />
                ) : (
                  <Empty
                    description={
                      hasAny ? (
                        <FormattedMessage
                          defaultMessage="No issues match these filters."
                          description="Experiment Issues no matching results"
                        />
                      ) : anyIssue.error || detection.error ? (
                        <FormattedMessage
                          defaultMessage="No issues are available for this search."
                          description="Experiment Issues empty search with unavailable history"
                        />
                      ) : detection.isLoading || anyIssue.isLoading ? (
                        <FormattedMessage
                          defaultMessage="Loading detection history…"
                          description="Experiment Issues empty state while checking history"
                        />
                      ) : cleanScan ? (
                        <FormattedMessage
                          defaultMessage="The latest detection found no issues in the selected traces."
                          description="Experiment Issues completed empty scan"
                        />
                      ) : !detection.run ? (
                        <FormattedMessage
                          defaultMessage="Detect issues to find problems in your traces."
                          description="Experiment Issues before the first detection"
                        />
                      ) : (
                        <FormattedMessage
                          defaultMessage="No issues have been recorded."
                          description="Experiment Issues empty issue store"
                        />
                      )
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
    <ExperimentIssueDetail key={selectedId} issue={detail.data} canEdit={canEdit} />
  ) : undefined;
  return (
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, height: '100%' }}>
      <div css={{ flexShrink: 0, paddingBottom: theme.spacing.md, borderBottom: `1px solid ${theme.colors.border}` }}>
        <ExperimentIssuesActionBar
          controls={{ ...controls, totalCount: total }}
          searchInput={controls.search.input}
          onSearchInputChange={controls.search.setInput}
          onSearchSubmit={controls.search.flush}
          onSearchClear={controls.search.clear}
          isDetecting={isDetectionButtonLoading}
          canDetect={canEdit && !isDetecting}
          detectionEnabled={detectionEnabled}
          onRunDetection={() => {
            setShowDetection(true);
            if (traces.error) traces.refetch();
          }}
        />
      </div>
      <IssueDetectionStatus detection={detection} experimentId={experimentId} />
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
      <IssueSplitPane list={list} detail={detailPane} />
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
  return <ExperimentIssuesContent key={`${workspace}:${experimentId}`} experimentId={experimentId} />;
};

export default ExperimentIssuesPage;
