import { FormattedMessage } from 'react-intl';
import { useSelector } from 'react-redux';
import { useMemo } from 'react';

import { Button, FileIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';

import Utils from '../../../common/utils/Utils';
import type { ReduxState } from '../../../redux-types';
import { useLocation } from '../../../common/utils/RoutingUtils';
import { EXPERIMENT_PARENT_ID_TAG } from '../experiment-page/utils/experimentPage.common-utils';

import { RunViewStatusBox } from './overview/RunViewStatusBox';
import { RunViewUserLinkBox } from './overview/RunViewUserLinkBox';
import { RunViewParamsTable } from './overview/RunViewParamsTable';
import { RunViewMetricsTable } from './overview/RunViewMetricsTable';
import { RunViewDatasetBox } from './overview/RunViewDatasetBox';
import { RunViewParentRunBox } from './overview/RunViewParentRunBox';
import { RunViewTagsBox } from './overview/RunViewTagsBox';
import { RunViewDescriptionBox } from './overview/RunViewDescriptionBox';
import { RunViewMetadataRow } from './overview/RunViewMetadataRow';
import { RunViewRegisteredModelsBox } from './overview/RunViewRegisteredModelsBox';
import { RunViewLoggedModelsBox } from './overview/RunViewLoggedModelsBox';
import { RunViewSourceBox } from './overview/RunViewSourceBox';
import { RunViewMetadataTable } from '@mlflow/mlflow/src/experiment-tracking/components/run-page/overview/RunViewMetadataTable';
import { RunViewCopyableIdBox } from './overview/RunViewCopyableIdBox';
import type { RunInfoEntity } from '../../types';
import type { UseGetRunQueryResponseRunInfo } from './hooks/useGetRunQuery';
import type { KeyValueEntity, MetricEntitiesByName, RunDatasetWithTags } from '../../types';

const EmptyValue = () => <Typography.Hint>—</Typography.Hint>;

export const RunViewOverview = ({
  runUuid,
  onRunDataUpdated,
  tags,
  runInfo,
  datasets,
  params,
  latestMetrics,
}: {
  runUuid: string;
  onRunDataUpdated: () => void | Promise<any>;
  runInfo: RunInfoEntity | UseGetRunQueryResponseRunInfo;
  tags: Record<string, KeyValueEntity>;
  latestMetrics: MetricEntitiesByName;
  datasets?: RunDatasetWithTags[];
  params: Record<string, KeyValueEntity>;
}) => {
  const { theme } = useDesignSystemTheme();
  const { search } = useLocation();

  const { registeredModels } = useSelector(({ entities }: ReduxState) => ({
    registeredModels: entities.modelVersionsByRunUuid[runUuid],
  }));

  const loggedModels = useMemo(() => Utils.getLoggedModelsFromTags(tags), [tags]);
  const parentRunIdTag = tags[EXPERIMENT_PARENT_ID_TAG];

  const renderDetails = () => {
    return (
      <RunViewMetadataTable>
        <RunViewMetadataRow
          title={
            <FormattedMessage
              defaultMessage="Created at"
              description="Run page > Overview > Run start time section label"
            />
          }
          value={runInfo.startTime ? Utils.formatTimestamp(runInfo.startTime) : <EmptyValue />}
        />
        <RunViewMetadataRow
          title={
            <FormattedMessage
              defaultMessage="Created by"
              description="Run page > Overview > Run author section label"
            />
          }
          value={<RunViewUserLinkBox runInfo={runInfo} tags={tags} />}
        />
        <RunViewMetadataRow
          title={
            <FormattedMessage
              defaultMessage="Experiment ID"
              description="Run page > Overview > experiment ID section label"
            />
          }
          value={<RunViewCopyableIdBox value={runInfo?.experimentId ?? ''} />}
        />
        <RunViewMetadataRow
          title={
            <FormattedMessage defaultMessage="Status" description="Run page > Overview > Run status section label" />
          }
          value={<RunViewStatusBox status={runInfo.status} />}
        />
        <RunViewMetadataRow
          title={<FormattedMessage defaultMessage="Run ID" description="Run page > Overview > Run ID section label" />}
          value={<RunViewCopyableIdBox value={runInfo.runUuid ?? ''} />}
        />
        <RunViewMetadataRow
          title={
            <FormattedMessage
              defaultMessage="Duration"
              description="Run page > Overview > Run duration section label"
            />
          }
          value={Utils.getDuration(runInfo.startTime, runInfo.endTime)}
        />
        {parentRunIdTag && (
          <RunViewMetadataRow
            title={<FormattedMessage defaultMessage="Parent run" description="Run page > Overview > Parent run" />}
            value={<RunViewParentRunBox parentRunUuid={parentRunIdTag.value} />}
          />
        )}
        <RunViewMetadataRow
          title={
            <FormattedMessage
              defaultMessage="Datasets used"
              description="Run page > Overview > Run datasets section label"
            />
          }
          value={
            datasets?.length ? <RunViewDatasetBox tags={tags} runInfo={runInfo} datasets={datasets} /> : <EmptyValue />
          }
        />
        <RunViewMetadataRow
          title={<FormattedMessage defaultMessage="Tags" description="Run page > Overview > Run tags section label" />}
          value={<RunViewTagsBox runUuid={runInfo.runUuid ?? ''} tags={tags} onTagsUpdated={onRunDataUpdated} />}
        />
        <RunViewMetadataRow
          title={
            <FormattedMessage defaultMessage="Source" description="Run page > Overview > Run source section label" />
          }
          value={<RunViewSourceBox tags={tags} search={search} runUuid={runUuid} />}
        />
        <RunViewMetadataRow
          title={
            <FormattedMessage
              defaultMessage="Logged models"
              description="Run page > Overview > Run models section label"
            />
          }
          value={
            loggedModels?.length > 0 ? (
              <RunViewLoggedModelsBox runInfo={runInfo} loggedModels={loggedModels} />
            ) : (
              <EmptyValue />
            )
          }
        />
        <RunViewMetadataRow
          title={
            <FormattedMessage
              defaultMessage="Registered models"
              description="Run page > Overview > Run models section label"
            />
          }
          value={
            registeredModels?.length > 0 ? (
              <RunViewRegisteredModelsBox runInfo={runInfo} registeredModels={registeredModels} />
            ) : (
              <EmptyValue />
            )
          }
        />
      </RunViewMetadataTable>
    );
  };

  const renderParams = () => {
    return <RunViewParamsTable params={params} runUuid={runUuid} />;
  };

  return (
    <div css={{ flex: '1' }}>
      <RunViewDescriptionBox runUuid={runUuid} tags={tags} onDescriptionChanged={onRunDataUpdated} />
      <Typography.Title level={4}>
        <FormattedMessage defaultMessage="Details" description="Run page > Overview > Details section title" />
      </Typography.Title>
      {renderDetails()}
      <div css={{ display: 'flex', gap: theme.spacing.lg, minHeight: 360, maxHeight: 760, overflow: 'hidden' }}>
        {renderParams()}
        <RunViewMetricsTable latestMetrics={latestMetrics} runInfo={runInfo} />
      </div>
    </div>
  );
};
