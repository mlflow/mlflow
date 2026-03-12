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
import { IssueDetectionProgress } from './IssueDetectionProgress';
import { useFetchIssueJobStatus, isJobComplete } from '../hooks/useFetchIssueJobStatus';
import Routes from '../../../routes';
import type { RunInfoEntity } from '../../../types';
import type { KeyValueEntity } from '../../../../common/types';
import type { UseGetRunQueryResponseRunInfo } from '../hooks/useGetRunQuery';

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
    result,
    isLoading: isLoadingJobStatus,
    error: jobStatusError,
  } = useFetchIssueJobStatus({
    jobId,
    enabled: !!jobId,
  });

  const model = tags['model']?.value;
  const provider = tags['provider']?.value;
  const categoriesStr = tags['categories']?.value;
  const categories = categoriesStr ? categoriesStr.split(',').map((c) => c.trim()) : undefined;
  const totalTraces = tags['total_traces']?.value ? parseInt(tags['total_traces'].value, 10) : undefined;
  const modelDisplay = model && provider ? `${provider}:/${model}` : undefined;

  const jobComplete = isJobComplete(jobStatus) || !!jobStatusError;
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
        {modelDisplay && (
          <KeyValueProperty
            keyValue={intl.formatMessage({
              defaultMessage: 'Model',
              description: 'Run page > Overview > Model used for issue detection',
            })}
            value={modelDisplay}
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
        jobStatus={jobStatus}
        totalTraces={totalTraces}
        result={result}
        isLoadingJobStatus={isLoadingJobStatus}
        jobStatusError={jobStatusError}
      />
    </DetailsPageLayout>
  );
};
