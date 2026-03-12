import { useDesignSystemTheme } from '@databricks/design-system';
import { KeyValueProperty, NoneCell } from '@databricks/web-shared/utils';
import { useIntl } from 'react-intl';
import { Link } from '../../../../common/utils/RoutingUtils';
import Utils from '../../../../common/utils/Utils';
import { DetailsPageLayout } from '../../../../common/components/details-page-layout/DetailsPageLayout';
import { DetailsOverviewCopyableIdBox } from '../../DetailsOverviewCopyableIdBox';
import { RunViewStatusBox } from './RunViewStatusBox';
import { RunViewUserLinkBox } from './RunViewUserLinkBox';
import { IssueDetectionProgress, type IssueDetectionProgressProps } from './IssueDetectionProgress';
import { useFetchIssueJobStatus, isJobComplete } from '../hooks/useFetchIssueJobStatus';
import Routes from '../../../routes';
import type { RunInfoEntity } from '../../../types';
import type { KeyValueEntity } from '../../../../common/types';
import type { UseGetRunQueryResponseRunInfo } from '../hooks/useGetRunQuery';

export interface IssueDetectionRunOverviewProps {
  runInfo: RunInfoEntity | UseGetRunQueryResponseRunInfo;
  tags: Record<string, KeyValueEntity>;
  progressProps: IssueDetectionProgressProps;
}

export const IssueDetectionRunOverview = ({ runInfo, tags, progressProps }: IssueDetectionRunOverviewProps) => {
  const intl = useIntl();

  const { status: jobStatus, error: jobStatusError } = useFetchIssueJobStatus({
    jobId: progressProps.jobId,
    enabled: !!progressProps.jobId,
  });

  const jobComplete = isJobComplete(jobStatus) || !!jobStatusError;

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
      <IssueDetectionProgress {...progressProps} />
    </DetailsPageLayout>
  );
};
