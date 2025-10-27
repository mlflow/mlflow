import { FormattedMessage, useIntl } from 'react-intl';
import { useMemo } from 'react';

import { Button, FileIcon, Spacer, Spinner, Typography, useDesignSystemTheme } from '@databricks/design-system';

import Utils from '../../../common/utils/Utils';
import { useLocation } from '../../../common/utils/RoutingUtils';
import { EXPERIMENT_PARENT_ID_TAG } from '../experiment-page/utils/experimentPage.common-utils';

import { RunViewStatusBox } from './overview/RunViewStatusBox';
import { RunViewUserLinkBox } from './overview/RunViewUserLinkBox';
import { DetailsOverviewParamsTable } from '../DetailsOverviewParamsTable';
import { RunViewMetricsTable } from './overview/RunViewMetricsTable';
import { RunViewDatasetBox } from './overview/RunViewDatasetBox';
import { RunViewParentRunBox } from './overview/RunViewParentRunBox';
import { RunViewTagsBox } from './overview/RunViewTagsBox';
import { RunViewDescriptionBox } from './overview/RunViewDescriptionBox';
import { DetailsOverviewMetadataRow } from '../DetailsOverviewMetadataRow';
import { RunViewRegisteredModelsBox } from './overview/RunViewRegisteredModelsBox';
import { RunViewRegisteredPromptsBox } from './overview/RunViewRegisteredPromptsBox';
import { RunViewLoggedModelsBox } from './overview/RunViewLoggedModelsBox';
import { RunViewSourceBox } from './overview/RunViewSourceBox';
import { DetailsOverviewMetadataTable } from '@mlflow/mlflow/src/experiment-tracking/components/DetailsOverviewMetadataTable';
import type { LoggedModelProto } from '../../types';
import { ExperimentKind } from '../../constants';
import { useExperimentLoggedModelRegisteredVersions } from '../experiment-logged-models/hooks/useExperimentLoggedModelRegisteredVersions';
import { DetailsOverviewCopyableIdBox } from '../DetailsOverviewCopyableIdBox';
import type { RunInfoEntity } from '../../types';
import type {
  UseGetRunQueryResponseInputs,
  UseGetRunQueryResponseOutputs,
  UseGetRunQueryResponseRunInfo,
} from './hooks/useGetRunQuery';
import type { MetricEntitiesByName, RunDatasetWithTags } from '../../types';
import type { KeyValueEntity } from '../../../common/types';
import { type RunPageModelVersionSummary } from './hooks/useUnifiedRegisteredModelVersionsSummariesForRun';
import { isEmpty, uniqBy } from 'lodash';
import { RunViewLoggedModelsTable } from './overview/RunViewLoggedModelsTable';
import { DetailsPageLayout } from '../../../common/components/details-page-layout/DetailsPageLayout';
import { useRunDetailsPageOverviewSectionsV2 } from './hooks/useRunDetailsPageOverviewSectionsV2';

const EmptyValue = () => <Typography.Hint>—</Typography.Hint>;

export const RunViewOverview = ({
  runUuid,
  onRunDataUpdated,
  tags,
  runInfo,
  datasets,
  params,
  latestMetrics,
  runInputs,
  runOutputs,
  registeredModelVersionSummaries: registeredModelVersionSummariesForRun,
  loggedModelsV3 = [],
  isLoadingLoggedModels = false,
  loggedModelsError,
  experimentKind,
}: {
  runUuid: string;
  onRunDataUpdated: () => void | Promise<any>;
  runInfo: RunInfoEntity | UseGetRunQueryResponseRunInfo;
  tags: Record<string, KeyValueEntity>;
  latestMetrics: MetricEntitiesByName;
  runInputs?: UseGetRunQueryResponseInputs;
  runOutputs?: UseGetRunQueryResponseOutputs;
  datasets?: RunDatasetWithTags[];
  params: Record<string, KeyValueEntity>;
  registeredModelVersionSummaries: RunPageModelVersionSummary[];
  loggedModelsV3?: LoggedModelProto[];
  isLoadingLoggedModels?: boolean;
  loggedModelsError?: Error;
  experimentKind?: ExperimentKind;
}) => {
  const { theme } = useDesignSystemTheme();
  const { search } = useLocation();
  const intl = useIntl();

  const loggedModelsFromTags = useMemo(() => Utils.getLoggedModelsFromTags(tags), [tags]);
  const parentRunIdTag = tags[EXPERIMENT_PARENT_ID_TAG];
  const containsLoggedModelsFromInputsOutputs = !isEmpty(runInputs?.modelInputs) || !isEmpty(runOutputs?.modelOutputs);
  const shouldRenderLoggedModelsBox = !containsLoggedModelsFromInputsOutputs;
  const shouldRenderLinkedPromptsTable = experimentKind === ExperimentKind.GENAI_DEVELOPMENT;

  // We have two flags for controlling the visibility of the "logged models" section:
  // - `shouldRenderLoggedModelsBox` determines if "logged models" section should be rendered.
  //   It is hidden if any IAv3 logged models are detected in inputs/outputs, in this case we're
  //   displaying a big table instead.
  // - `shouldDisplayContentsOfLoggedModelsBox` determines if the contents of the "logged models"
  //   section should be displayed. It is hidden if there are no logged models to display.
  const shouldDisplayContentsOfLoggedModelsBox = loggedModelsFromTags?.length > 0 || loggedModelsV3?.length > 0;
  const { modelVersions: loggedModelsV3RegisteredModels } = useExperimentLoggedModelRegisteredVersions({
    loggedModels: loggedModelsV3,
  });

  /**
   * We have to query multiple sources for registered model versions (logged models API, models API, UC)
   * and it's possible to end up with duplicates.
   * We can dedupe them using `link` field, which should be unique for each model.
   */
  const registeredModelVersionSummaries = uniqBy(
    [...registeredModelVersionSummariesForRun, ...loggedModelsV3RegisteredModels],
    (model) => model?.link,
  );

  const renderPromptMetadataRow = () => {
    return (
      <DetailsOverviewMetadataRow
        title={
          <FormattedMessage
            defaultMessage="Registered prompts"
            description="Run page > Overview > Run prompts section label"
          />
        }
        value={<RunViewRegisteredPromptsBox tags={tags} runUuid={runUuid} />}
      />
    );
  };

  const renderDetails = () => {
    return (
      <DetailsOverviewMetadataTable>
        <DetailsOverviewMetadataRow
          title={
            <FormattedMessage
              defaultMessage="Created at"
              description="Run page > Overview > Run start time section label"
            />
          }
          value={runInfo.startTime ? Utils.formatTimestamp(runInfo.startTime, intl) : <EmptyValue />}
        />
        <DetailsOverviewMetadataRow
          title={
            <FormattedMessage
              defaultMessage="Created by"
              description="Run page > Overview > Run author section label"
            />
          }
          value={<RunViewUserLinkBox runInfo={runInfo} tags={tags} />}
        />
        <DetailsOverviewMetadataRow
          title={
            <FormattedMessage
              defaultMessage="Experiment ID"
              description="Run page > Overview > experiment ID section label"
            />
          }
          value={<DetailsOverviewCopyableIdBox value={runInfo?.experimentId ?? ''} />}
        />
        <DetailsOverviewMetadataRow
          title={
            <FormattedMessage defaultMessage="Status" description="Run page > Overview > Run status section label" />
          }
          value={<RunViewStatusBox status={runInfo.status} />}
        />
        <DetailsOverviewMetadataRow
          title={<FormattedMessage defaultMessage="Run ID" description="Run page > Overview > Run ID section label" />}
          value={<DetailsOverviewCopyableIdBox value={runInfo.runUuid ?? ''} />}
        />
        <DetailsOverviewMetadataRow
          title={
            <FormattedMessage
              defaultMessage="Duration"
              description="Run page > Overview > Run duration section label"
            />
          }
          value={Utils.getDuration(runInfo.startTime, runInfo.endTime)}
        />
        {parentRunIdTag && (
          <DetailsOverviewMetadataRow
            title={<FormattedMessage defaultMessage="Parent run" description="Run page > Overview > Parent run" />}
            value={<RunViewParentRunBox parentRunUuid={parentRunIdTag.value} />}
          />
        )}
        <DetailsOverviewMetadataRow
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
        <DetailsOverviewMetadataRow
          title={<FormattedMessage defaultMessage="Tags" description="Run page > Overview > Run tags section label" />}
          value={<RunViewTagsBox runUuid={runInfo.runUuid ?? ''} tags={tags} onTagsUpdated={onRunDataUpdated} />}
        />
        <DetailsOverviewMetadataRow
          title={
            <FormattedMessage defaultMessage="Source" description="Run page > Overview > Run source section label" />
          }
          value={<RunViewSourceBox tags={tags} search={search} runUuid={runUuid} />}
        />
        {shouldRenderLoggedModelsBox && (
          <DetailsOverviewMetadataRow
            title={
              <FormattedMessage
                defaultMessage="Logged models"
                description="Run page > Overview > Run models section label"
              />
            }
            value={
              isLoadingLoggedModels ? (
                <Spinner />
              ) : shouldDisplayContentsOfLoggedModelsBox ? (
                <RunViewLoggedModelsBox
                  // Pass the run info and logged models
                  runInfo={runInfo}
                  loggedModels={loggedModelsFromTags}
                  // Provide loggedModels from IA v3
                  loggedModelsV3={loggedModelsV3}
                />
              ) : (
                <EmptyValue />
              )
            }
          />
        )}
        <DetailsOverviewMetadataRow
          title={
            <FormattedMessage
              defaultMessage="Registered models"
              description="Run page > Overview > Run models section label"
            />
          }
          value={
            registeredModelVersionSummaries?.length > 0 ? (
              <RunViewRegisteredModelsBox registeredModelVersionSummaries={registeredModelVersionSummaries} />
            ) : (
              <EmptyValue />
            )
          }
        />
        {renderPromptMetadataRow()}
      </DetailsOverviewMetadataTable>
    );
  };

  const renderParams = () => {
    return <DetailsOverviewParamsTable params={params} />;
  };

  const detailsSectionsV2 = useRunDetailsPageOverviewSectionsV2({
    runUuid,
    runInfo,
    tags,
    onTagsUpdated: onRunDataUpdated,
    datasets,
    loggedModelsV3,
    shouldRenderLoggedModelsBox,
    registeredModelVersionSummaries,
  });
  const usingSidebarLayout = true;
  return (
    <DetailsPageLayout
      css={{ flex: 1, alignSelf: 'flex-start' }}
      // Enable sidebar layout based on feature flag
      usingSidebarLayout={usingSidebarLayout}
      secondarySections={detailsSectionsV2}
    >
      <RunViewDescriptionBox runUuid={runUuid} tags={tags} onDescriptionChanged={onRunDataUpdated} />
      {!usingSidebarLayout && (
        <>
          <Typography.Title level={4}>
            <FormattedMessage defaultMessage="Details" description="Run page > Overview > Details section title" />
          </Typography.Title>
          {renderDetails()}
        </>
      )}
      <div
        // Use different grid setup for unified details page layout
        css={[
          usingSidebarLayout ? { flexDirection: 'column' } : { minHeight: 360, maxHeight: 760 },
          { display: 'flex', gap: theme.spacing.lg, overflow: 'hidden' },
        ]}
      >
        <RunViewMetricsTable latestMetrics={latestMetrics} runInfo={runInfo} loggedModels={loggedModelsV3} />
        {renderParams()}
      </div>
      {containsLoggedModelsFromInputsOutputs && (
        <>
          {!usingSidebarLayout && <Spacer />}
          <div css={{ minHeight: 360, maxHeight: 760, overflow: 'hidden', display: 'flex' }}>
            <RunViewLoggedModelsTable
              loggedModelsV3={loggedModelsV3}
              isLoadingLoggedModels={isLoadingLoggedModels}
              inputs={runInputs}
              outputs={runOutputs}
              runInfo={runInfo}
              loggedModelsError={loggedModelsError}
            />
          </div>
        </>
      )}
      {!usingSidebarLayout && <Spacer />}
      {/* Add a spacer so the page doesn't jump when searching params / metrics */}
      <div css={{ height: 500 }} />
    </DetailsPageLayout>
  );
};
