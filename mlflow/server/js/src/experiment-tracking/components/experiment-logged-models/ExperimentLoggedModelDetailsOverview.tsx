import { Alert, GenericSkeleton, Spacer, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { LoggedModelProto } from '../../types';
import type { KeyValueEntity } from '../../../common/types';
import { DetailsOverviewMetadataTable } from '../DetailsOverviewMetadataTable';
import { DetailsOverviewMetadataRow } from '../DetailsOverviewMetadataRow';
import { FormattedMessage } from 'react-intl';
import { ExperimentLoggedModelTableDateCell } from './ExperimentLoggedModelTableDateCell';
import { ExperimentLoggedModelStatusIndicator } from './ExperimentLoggedModelStatusIndicator';
import { DetailsOverviewCopyableIdBox } from '../DetailsOverviewCopyableIdBox';
import { ExperimentLoggedModelDescription } from './ExperimentLoggedModelDescription';
import { DetailsOverviewParamsTable } from '../DetailsOverviewParamsTable';
import { useMemo } from 'react';
import { isEmpty, keyBy } from 'lodash';
import { ExperimentLoggedModelDetailsMetricsTable } from './ExperimentLoggedModelDetailsMetricsTable';
import { ExperimentLoggedModelDetailsPageRunsTable } from './ExperimentLoggedModelDetailsRunsTable';
import { ExperimentLoggedModelDetailsPageLinkedPromptsTable } from './ExperimentLoggedModelDetailsPageLinkedPromptsTable';
import { useRelatedRunsDataForLoggedModels } from '../../hooks/logged-models/useRelatedRunsDataForLoggedModels';
import { Link } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { ExperimentLoggedModelAllDatasetsList } from './ExperimentLoggedModelAllDatasetsList';
import { ExperimentLoggedModelOpenDatasetDetailsContextProvider } from './hooks/useExperimentLoggedModelOpenDatasetDetails';
import { ExperimentLoggedModelDetailsModelVersionsList } from './ExperimentLoggedModelDetailsModelVersionsList';
import { ExperimentLoggedModelSourceBox } from './ExperimentLoggedModelSourceBox';
import { DetailsPageLayout } from '../../../common/components/details-page-layout/DetailsPageLayout';
import { useExperimentLoggedModelDetailsMetadataV2 } from './hooks/useExperimentLoggedModelDetailsMetadataV2';
import { ExperimentKind, MLFLOW_LOGGED_MODEL_USER_TAG } from '../../constants';

export const ExperimentLoggedModelDetailsOverview = ({
  onDataUpdated,
  loggedModel,
  experimentKind,
}: {
  onDataUpdated: () => void | Promise<any>;
  loggedModel?: LoggedModelProto;
  experimentKind?: ExperimentKind;
}) => {
  const { theme } = useDesignSystemTheme();
  const shouldRenderLinkedPromptsTable = experimentKind === ExperimentKind.GENAI_DEVELOPMENT;

  // Fetch related runs data for the logged model
  const {
    data: relatedRunsData,
    loading: relatedRunsLoading,
    error: relatedRunsDataError,
  } = useRelatedRunsDataForLoggedModels({ loggedModels: loggedModel ? [loggedModel] : [] });

  const relatedSourceRun = useMemo(
    () => relatedRunsData?.find((r) => r.info?.runUuid === loggedModel?.info?.source_run_id),
    [loggedModel?.info?.source_run_id, relatedRunsData],
  );

  const paramsDictionary = useMemo(
    () =>
      keyBy(
        (loggedModel?.data?.params ?? []).filter(({ key, value }) => !isEmpty(key) && !isEmpty(value)),
        'key',
      ) as Record<string, KeyValueEntity>,
    [loggedModel?.data?.params],
  );

  const renderDetails = () => {
    if (!loggedModel) {
      return null;
    }
    return (
      <DetailsOverviewMetadataTable>
        <DetailsOverviewMetadataRow
          title={
            <FormattedMessage
              defaultMessage="Created at"
              description="Label for the creation timestamp of a logged model on the logged model details page"
            />
          }
          value={<ExperimentLoggedModelTableDateCell value={loggedModel.info?.creation_timestamp_ms} />}
        />
        <DetailsOverviewMetadataRow
          title={
            <FormattedMessage
              defaultMessage="Created by"
              description="Label for the creator of a logged model on the logged model details page"
            />
          }
          value={loggedModel.info?.tags?.find((tag) => tag.key === MLFLOW_LOGGED_MODEL_USER_TAG)?.value ?? '-'}
        />
        <DetailsOverviewMetadataRow
          title={
            <FormattedMessage
              defaultMessage="Status"
              description="Label for the status of a logged model on the logged model details page"
            />
          }
          value={<ExperimentLoggedModelStatusIndicator data={loggedModel} />}
        />
        <DetailsOverviewMetadataRow
          title={
            <FormattedMessage
              defaultMessage="Model ID"
              description="Label for the model ID of a logged model on the logged model details page"
            />
          }
          value={<DetailsOverviewCopyableIdBox value={loggedModel.info?.model_id ?? ''} />}
        />
        {/* If the logged model has a source run, display the source run name after its loaded */}
        {loggedModel.info?.source_run_id &&
          loggedModel.info?.experiment_id &&
          (relatedRunsLoading || relatedSourceRun) && (
            <DetailsOverviewMetadataRow
              title={
                <FormattedMessage
                  defaultMessage="Source run"
                  description="Label for the source run name of a logged model on the logged model details page"
                />
              }
              value={
                // Display a skeleton while loading
                relatedRunsLoading ? (
                  <GenericSkeleton css={{ width: 200, height: theme.spacing.md }} />
                ) : (
                  <Link to={Routes.getRunPageRoute(loggedModel.info?.experiment_id, loggedModel.info?.source_run_id)}>
                    {relatedSourceRun?.info?.runName}
                  </Link>
                )
              }
            />
          )}
        {loggedModel.info?.source_run_id && (
          <DetailsOverviewMetadataRow
            title={
              <FormattedMessage
                defaultMessage="Source run ID"
                description="Label for the source run ID of a logged model on the logged model details page"
              />
            }
            value={<DetailsOverviewCopyableIdBox value={loggedModel.info?.source_run_id ?? ''} />}
          />
        )}
        <DetailsOverviewMetadataRow
          title={
            <FormattedMessage
              defaultMessage="Logged from"
              description="Label for the source (where it was logged from) of a logged model on the logged model details page. It can be e.g. a notebook or a file."
            />
          }
          value={<ExperimentLoggedModelSourceBox loggedModel={loggedModel} displayDetails />}
        />
        <DetailsOverviewMetadataRow
          title={
            <FormattedMessage
              defaultMessage="Datasets used"
              description="Label for the datasets used by a logged model on the logged model details page"
            />
          }
          value={<ExperimentLoggedModelAllDatasetsList loggedModel={loggedModel} />}
        />
        <DetailsOverviewMetadataRow
          title={
            <FormattedMessage
              defaultMessage="Model versions"
              description="Label for the model versions of a logged model on the logged model details page"
            />
          }
          value={<ExperimentLoggedModelDetailsModelVersionsList loggedModel={loggedModel} />}
        />
      </DetailsOverviewMetadataTable>
    );
  };

  const detailsSectionsV2 = useExperimentLoggedModelDetailsMetadataV2({
    loggedModel,
    relatedRunsLoading,
    relatedSourceRun,
  });

  return (
    <ExperimentLoggedModelOpenDatasetDetailsContextProvider>
      <DetailsPageLayout css={{ flex: 1 }} usingSidebarLayout secondarySections={detailsSectionsV2}>
        <ExperimentLoggedModelDescription loggedModel={loggedModel} onDescriptionChanged={onDataUpdated} />

        {relatedRunsDataError?.message && (
          <>
            <Alert
              closable={false}
              message={
                <FormattedMessage
                  defaultMessage="Error when fetching related runs data: {error}"
                  description="Error message displayed when logged model details page couldn't fetch related runs data"
                  values={{
                    error: relatedRunsDataError.message,
                  }}
                />
              }
              type="error"
              componentId="mlflow.logged_model.details.related_runs.error"
            />
            <Spacer size="md" />
          </>
        )}
        <div
          css={[
            {
              display: 'flex',
              flexDirection: 'column',
            },
            {
              gap: theme.spacing.lg,
              overflow: 'hidden',
              // add some bottom padding so the user can interact with the
              // last table closer to the center of the page
              paddingBottom: theme.spacing.lg * 3,
            },
          ]}
        >
          <ExperimentLoggedModelDetailsMetricsTable
            loggedModel={loggedModel}
            relatedRunsLoading={relatedRunsLoading}
            relatedRunsData={relatedRunsData ?? undefined}
          />
          <DetailsOverviewParamsTable params={paramsDictionary} />
          <ExperimentLoggedModelDetailsPageRunsTable
            loggedModel={loggedModel}
            relatedRunsLoading={relatedRunsLoading}
            relatedRunsData={relatedRunsData ?? undefined}
          />
          {shouldRenderLinkedPromptsTable && (
            <ExperimentLoggedModelDetailsPageLinkedPromptsTable loggedModel={loggedModel} />
          )}
        </div>
      </DetailsPageLayout>
    </ExperimentLoggedModelOpenDatasetDetailsContextProvider>
  );
};
