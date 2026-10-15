import { useState, type ReactNode } from 'react';
import {
  Alert,
  Button,
  Empty,
  PreviewCard,
  SparkleIcon,
  TableSkeleton,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { NotFoundError, PermissionError } from '@databricks/web-shared/errors';
import { FormattedMessage, useIntl } from 'react-intl';
import { useParams, useSearchParams } from '../../../common/utils/RoutingUtils';
import { useActiveWorkspace } from '../../../workspaces/utils/WorkspaceUtils';
import { ModelTraceExplorerResizablePane } from '../../../shared/web-shared/model-trace-explorer';
import { useLocalStorage } from '../../../shared/web-shared/hooks';
import { IssueDetectionModal } from '../../components/experiment-page/components/traces-v3/IssueDetectionModal';
import { STATUS_TAG_CONFIG, SEVERITY_TAG_CONFIG } from '../../components/run-page/IssueCard';
import { JobStatus } from '../../components/run-page/hooks/useFetchJobStatus';
import { SELECTED_ISSUE_ID_PARAM } from '../../constants';
import { useExperimentPermission } from '../../hooks/useExperimentPermission';
import { SERVER_FEATURE_KEYS, useFeatureEnabled, useServerInfo } from '../../hooks/useServerInfo';
import { ExperimentIssueDetail } from './ExperimentIssueDetail';
import { ExperimentIssuesToolbar, useIssueListControls } from './ExperimentIssuesToolbar';
import { IssueDetectionStatus } from './IssueDetectionStatus';
import { useDetectionTraceIds, useLatestIssueDetection } from './useIssueDetection';
import { useExperimentIssue, useExperimentIssues, useHasExperimentIssues } from './useExperimentIssues';

const CenteredState = ({ children }: { children: ReactNode }) => (
  <div
    css={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: 320,
      height: '100%',
      width: '100%',
      flex: 1,
      '& > div': { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' },
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
  const intl = useIntl();
  const controls = useIssueListControls();
  const search = useExperimentIssues(experimentId, controls.filters);
  const anyIssue = useHasExperimentIssues(experimentId);
  const detection = useLatestIssueDetection(experimentId);
  const canEdit = useExperimentPermission(experimentId, ['EDIT', 'MANAGE']);
  const detectionEnabled = useFeatureEnabled(SERVER_FEATURE_KEYS.GATEWAY);
  const [showDetection, setShowDetection] = useState(false);
  const traces = useDetectionTraceIds(experimentId, showDetection);
  const [params, setParams] = useSearchParams();
  const issues = search.data?.pages.flatMap((page) => page.issues ?? []) ?? [];
  const selectedId = params.get(SELECTED_ISSUE_ID_PARAM) ?? issues[0]?.issue_id;
  const detail = useExperimentIssue(experimentId, selectedId);
  const [storedRatio, setStoredRatio] = useLocalStorage<number>({
    key: 'mlflow.issues.list-ratio',
    version: 1,
    initialValue: 1 / 3,
  });
  const [paneWidth, setPaneWidth] = useState(320);
  const selectIssue = (issueId: string) =>
    setParams((current) => {
      current.set(SELECTED_ISSUE_ID_PARAM, issueId);
      return current;
    });
  const total = search.data?.pages[0]?.total_count;
  const hasAny = Boolean(anyIssue.data?.issues?.length);
  const cleanScan = detection.status === JobStatus.SUCCEEDED && detection.result?.issues === 0;
  const list = (
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0, overflow: 'auto' }}>
      {search.isLoading ? (
        <TableSkeleton lines={5} />
      ) : search.error && !search.data ? (
        <ReadError error={search.error} retry={() => search.refetch()} />
      ) : (
        <>
          {issues.length === 0 ? (
            <CenteredState>
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
            </CenteredState>
          ) : (
            issues.map((issue) => (
              <PreviewCard
                key={issue.issue_id}
                componentId="mlflow.experiment-issues.row"
                selected={selectedId === issue.issue_id}
                onClick={() => selectIssue(issue.issue_id)}
                title={<Typography.Text bold>{issue.name}</Typography.Text>}
                titleActions={
                  <Typography.Text color="secondary">
                    {intl.formatMessage(STATUS_TAG_CONFIG[issue.status].label)}
                  </Typography.Text>
                }
                dangerouslyAppendEmotionCSS={{
                  flexShrink: 0,
                  marginBottom: theme.spacing.xs,
                  padding: theme.spacing.md,
                }}
              >
                <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.sm }}>
                  {issue.severity && (
                    <Typography.Text>{intl.formatMessage(SEVERITY_TAG_CONFIG[issue.severity].label)}</Typography.Text>
                  )}
                  <Typography.Text color="secondary">{issue.categories?.join(', ')}</Typography.Text>
                  {issue.trace_count !== undefined && (
                    <Typography.Text color="secondary">
                      <FormattedMessage
                        defaultMessage="{count, plural, one {# trace} other {# traces}}"
                        description="Experiment Issue affected trace count"
                        values={{ count: issue.trace_count }}
                      />
                    </Typography.Text>
                  )}
                </div>
              </PreviewCard>
            ))
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
      <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: theme.spacing.md }}>
        <Typography.Title level={3}>
          <FormattedMessage defaultMessage="Issues" description="Experiment Issues page heading" />
          {total !== undefined && <Typography.Text color="secondary"> ({total})</Typography.Text>}
        </Typography.Title>
        {detectionEnabled && (
          <Button
            componentId="mlflow.experiment-issues.detect"
            type="primary"
            icon={<SparkleIcon />}
            disabled={!canEdit}
            loading={showDetection && traces.isFetching}
            onClick={() => {
              setShowDetection(true);
              if (traces.error) traces.refetch();
            }}
          >
            <FormattedMessage defaultMessage="Detect issues" description="Experiment Issues launch detection action" />
          </Button>
        )}
      </div>
      <IssueDetectionStatus detection={detection} experimentId={experimentId} canEdit={canEdit} />
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
      <ExperimentIssuesToolbar controls={controls} />
      <div css={{ display: 'flex', flex: 1, minHeight: 0, minWidth: 0, overflow: 'hidden' }}>
        {detailPane ? (
          <ModelTraceExplorerResizablePane
            initialRatio={storedRatio > 0 && storedRatio < 1 ? storedRatio : 1 / 3}
            paneWidth={paneWidth}
            setPaneWidth={setPaneWidth}
            onRatioChange={setStoredRatio}
            leftMinWidth={280}
            rightMinWidth={400}
            leftChild={list}
            rightChild={detailPane}
          />
        ) : (
          list
        )}
      </div>
      {showDetection && traces.data && !traces.isFetching && !traces.error && (
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
