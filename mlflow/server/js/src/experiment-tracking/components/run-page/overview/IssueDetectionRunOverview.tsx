import { useEffect, useRef } from 'react';
import { Tag, useDesignSystemTheme } from '@databricks/design-system';
import { KeyValueProperty, NoneCell } from '@databricks/web-shared/utils';
import { useIntl } from 'react-intl';
import { Link } from '../../../../common/utils/RoutingUtils';
import Utils from '../../../../common/utils/Utils';
import { DetailsPageLayout } from '../../../../common/components/details-page-layout/DetailsPageLayout';
import { DetailsOverviewCopyableIdBox } from '../../DetailsOverviewCopyableIdBox';
import { RunViewStatusBox } from './RunViewStatusBox';
import { RunViewUserLinkBox } from './RunViewUserLinkBox';
import { IssueDetectionProgress, type IssueJobResult } from './IssueDetectionProgress';
import { useFetchJobStatus, isJobComplete, JobStatus } from '../hooks/useFetchJobStatus';
import Routes from '../../../routes';
import type { RunInfoEntity } from '../../../types';
import type { KeyValueEntity } from '../../../../common/types';
import type { UseGetRunQueryResponseRunInfo } from '../hooks/useGetRunQuery';
import { runStatusToJobStatus } from '../../../utils/statusMapping';
import {
  MLFLOW_ISSUE_DETECTION_RESULT_ISSUES_TAG,
  MLFLOW_ISSUE_DETECTION_RESULT_TOTAL_TRACES_TAG,
  MLFLOW_ISSUE_DETECTION_RESULT_SUMMARY_TAG,
} from '../../../constants';

export interface IssueDetectionRunOverviewProps {
  runInfo: RunInfoEntity | UseGetRunQueryResponseRunInfo;
  tags: Record<string, KeyValueEntity>;
  /** Job ID for fetching issue detection job status */
  jobId?: string;
  /** Callback when run data is updated (e.g., job completes) */
  onRunDataUpdated?: () => void;
}

export const IssueDetectionRunOverview = ({
  runInfo,
  tags,
  jobId,
  onRunDataUpdated,
}: IssueDetectionRunOverviewProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const {
    status: jobStatus,
    result: rawResult,
    status_details: jobStatusDetails,
    isLoading: isLoadingJobStatus,
    error: jobStatusError,
  } = useFetchJobStatus({
    jobId,
    enabled: !!jobId,
  });

  // Parse issue-specific result format from job if available
  const isFailed = jobStatus === JobStatus.FAILED || jobStatus === JobStatus.TIMEOUT;
  const jobErrorMessage = isFailed && typeof rawResult === 'string' ? rawResult : undefined;
  const jobResult =
    !isFailed && typeof rawResult === 'object' && rawResult !== null ? (rawResult as IssueJobResult) : undefined;

  // Fall back to reading result from run tags if no job exists
  const resultFromTags: IssueJobResult | undefined = !jobId
    ? (() => {
        const issuesTag = tags[MLFLOW_ISSUE_DETECTION_RESULT_ISSUES_TAG]?.value;
        const tracesTag = tags[MLFLOW_ISSUE_DETECTION_RESULT_TOTAL_TRACES_TAG]?.value;
        const summaryTag = tags[MLFLOW_ISSUE_DETECTION_RESULT_SUMMARY_TAG]?.value;

        if (issuesTag && tracesTag) {
          return {
            issues: parseInt(issuesTag, 10),
            total_traces_analyzed: parseInt(tracesTag, 10),
            summary: summaryTag,
          };
        }
        return undefined;
      })()
    : undefined;

  const result = jobResult || resultFromTags;

  // Derive job status from run status if no job exists
  const effectiveJobStatus = jobStatus || (!jobId && runInfo.status ? runStatusToJobStatus(runInfo.status) : undefined);

  const model = tags['model']?.value;
  const categoriesStr = tags['categories']?.value;
  const categories = categoriesStr ? categoriesStr.split(',').map((c) => c.trim()) : undefined;
  // Use total_traces_analyzed from result if available, otherwise fall back to total_traces tag
  const totalTraces =
    result?.total_traces_analyzed ??
    (tags['total_traces']?.value ? parseInt(tags['total_traces'].value, 10) : undefined);

  const jobComplete = isJobComplete(effectiveJobStatus) || !!jobStatusError;
  const prevJobCompleteRef = useRef(jobComplete);

  useEffect(() => {
    if (jobComplete && !prevJobCompleteRef.current) {
      onRunDataUpdated?.();
    }
    prevJobCompleteRef.current = jobComplete;
  }, [jobComplete, onRunDataUpdated]);

  const detailsSection = {
    id: 'DETAILS',
    title: intl.formatMessage({
      defaultMessage: 'About this run',
      description: 'Title for the details/metadata section on the run details page',
    }),
    content: runInfo && (
      <>
        <KeyValueProperty
          keyValue={intl.formatMessage({
            defaultMessage: 'Created at',
            description: 'Run page > Overview > Run start time section label',
          })}
          value={runInfo.startTime ? Utils.formatTimestamp(runInfo.startTime, intl) : <NoneCell />}
        />
        <KeyValueProperty
          keyValue={intl.formatMessage({
            defaultMessage: 'Created by',
            description: 'Run page > Overview > Run author section label',
          })}
          value={<RunViewUserLinkBox runInfo={runInfo} tags={tags} />}
        />
        {model && (
          <KeyValueProperty
            keyValue={intl.formatMessage({
              defaultMessage: 'Model',
              description: 'Run page > Overview > Model used for issue detection',
            })}
            value={model}
          />
        )}
        {categories && categories.length > 0 && (
          <KeyValueProperty
            keyValue={intl.formatMessage({
              defaultMessage: 'Categories',
              description: 'Run page > Overview > Issue categories being detected',
            })}
            value={
              <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.xs }}>
                {categories.map((category) => (
                  <Tag key={category} componentId="mlflow.issue-detection.category-tag">
                    {category}
                  </Tag>
                ))}
              </div>
            }
          />
        )}
        <KeyValueProperty
          keyValue={intl.formatMessage({
            defaultMessage: 'Experiment ID',
            description: 'Run page > Overview > experiment ID section label',
          })}
          value={
            <DetailsOverviewCopyableIdBox
              value={runInfo?.experimentId ?? ''}
              element={
                runInfo?.experimentId ? (
                  <Link
                    componentId="mlflow.run_page.overview.issue_detection_experiment_link"
                    to={Routes.getExperimentPageRoute(runInfo.experimentId)}
                  >
                    {runInfo?.experimentId}
                  </Link>
                ) : undefined
              }
            />
          }
        />
        <KeyValueProperty
          keyValue={intl.formatMessage({
            defaultMessage: 'Status',
            description: 'Run page > Overview > Run status section label',
          })}
          value={<RunViewStatusBox status={runInfo.status} useSpinner />}
        />
        <KeyValueProperty
          keyValue={intl.formatMessage({
            defaultMessage: 'Run ID',
            description: 'Run page > Overview > Run ID section label',
          })}
          value={<DetailsOverviewCopyableIdBox value={runInfo.runUuid ?? ''} />}
        />
        {jobComplete && (
          <KeyValueProperty
            keyValue={intl.formatMessage({
              defaultMessage: 'Duration',
              description: 'Run page > Overview > Run duration section label',
            })}
            value={Utils.getDuration(runInfo.startTime, runInfo.endTime)}
          />
        )}
      </>
    ),
  };

  return (
    <DetailsPageLayout
      css={{ flex: 1, alignSelf: 'flex-start' }}
      usingSidebarLayout
      secondarySections={[detailsSection]}
    >
      <IssueDetectionProgress
        jobId={jobId}
        jobStatus={effectiveJobStatus}
        jobStage={jobStatusDetails?.stage}
        totalTraces={totalTraces}
        result={result}
        isLoadingJobStatus={jobId ? isLoadingJobStatus : false}
        jobStatusError={jobStatusError}
        jobErrorMessage={jobErrorMessage}
      />
    </DetailsPageLayout>
  );
};
