import { Alert, GenericSkeleton, Spacer, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { KeyValueEntity, LoggedModelProto } from '../../types';
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
import { useRelatedRunsDataForLoggedModels } from '../../hooks/logged-models/useRelatedRunsDataForLoggedModels';
import { Link } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { ExperimentLoggedModelAllDatasetsList } from './ExperimentLoggedModelAllDatasetsList';
import { ExperimentLoggedModelOpenDatasetDetailsContextProvider } from './hooks/useExperimentLoggedModelOpenDatasetDetails';
import { ExperimentLoggedModelDetailsModelVersionsList } from './ExperimentLoggedModelDetailsModelVersionsList';

export const ExperimentLoggedModelDetailsOverview = ({
  onDataUpdated,
  loggedModel,
}: {
  onDataUpdated: () => void | Promise<any>;
  loggedModel?: LoggedModelProto;
}) => {
  const { theme } = useDesignSystemTheme();

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
        {/* TODO(ML-47205): Re-enable this when creator name/email is available */}
        {/* <DetailsOverviewMetadataRow
          title={
            <FormattedMessage
              defaultMessage="Created by"
              description="Label for the creator of a logged model on the logged model details page"
            />
          }
          value={loggedModel.info?.creator_id}
        /> */}
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

  return (
    <ExperimentLoggedModelOpenDatasetDetailsContextProvider>
      <div css={{ flex: '1' }}>
        <ExperimentLoggedModelDescription loggedModel={loggedModel} onDescriptionChanged={onDataUpdated} />
        <Typography.Title level={4}>
          <FormattedMessage
            defaultMessage="Details"
            description="Title for the details section on the logged model details page"
          />
        </Typography.Title>
        {renderDetails()}
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
          css={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gridTemplateRows: '400px 400px',
            gap: theme.spacing.lg,
            overflow: 'hidden',
            marginBottom: theme.spacing.md,
          }}
        >
          <DetailsOverviewParamsTable params={paramsDictionary} />
          <ExperimentLoggedModelDetailsMetricsTable
            loggedModel={loggedModel}
            relatedRunsLoading={relatedRunsLoading}
            relatedRunsData={relatedRunsData ?? undefined}
          />
          <ExperimentLoggedModelDetailsPageRunsTable
            loggedModel={loggedModel}
            relatedRunsLoading={relatedRunsLoading}
            relatedRunsData={relatedRunsData ?? undefined}
          />
          <div>{/* TODO: inference tables list */}</div>
        </div>
      </div>
    </ExperimentLoggedModelOpenDatasetDetailsContextProvider>
  );
};
