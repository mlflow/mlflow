import { GenericSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import type { LoggedModelProto, RunEntity } from '../../../types';
import { useIntl } from 'react-intl';
import { ExperimentLoggedModelTableDateCell } from '../ExperimentLoggedModelTableDateCell';
import { ExperimentLoggedModelStatusIndicator } from '../ExperimentLoggedModelStatusIndicator';
import { DetailsOverviewCopyableIdBox } from '../../DetailsOverviewCopyableIdBox';
import { Link } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';
import type { AsideSections } from '@databricks/web-shared/utils';
import { KeyValueProperty, NoneCell } from '@databricks/web-shared/utils';
import { ExperimentLoggedModelSourceBox } from '../ExperimentLoggedModelSourceBox';
import { ExperimentLoggedModelAllDatasetsList } from '../ExperimentLoggedModelAllDatasetsList';
import { ExperimentLoggedModelDetailsModelVersionsList } from '../ExperimentLoggedModelDetailsModelVersionsList';
import { MLFLOW_LOGGED_MODEL_USER_TAG } from '../../../constants';

enum ExperimentLoggedModelDetailsMetadataSections {
  DETAILS = 'DETAILS',
  DATASETS = 'DATASETS',
  MODEL_VERSIONS = 'MODEL_VERSIONS',
}

export const useExperimentLoggedModelDetailsMetadataV2 = ({
  loggedModel,
  relatedRunsLoading,
  relatedSourceRun,
}: {
  loggedModel?: LoggedModelProto;
  relatedRunsLoading?: boolean;
  relatedSourceRun?: RunEntity;
}): AsideSections => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const detailsContent = loggedModel && (
    <>
      <KeyValueProperty
        keyValue={intl.formatMessage({
          defaultMessage: 'Created at',
          description: 'Label for the creation timestamp of a logged model on the logged model details page',
        })}
        value={<ExperimentLoggedModelTableDateCell value={loggedModel?.info?.creation_timestamp_ms} />}
      />
      <KeyValueProperty
        keyValue={intl.formatMessage({
          defaultMessage: 'Created by',
          description: 'Label for the creator of a logged model on the logged model details page',
        })}
        value={loggedModel.info?.tags?.find((tag) => tag.key === MLFLOW_LOGGED_MODEL_USER_TAG)?.value ?? '-'}
      />
      <KeyValueProperty
        keyValue={intl.formatMessage({
          defaultMessage: 'Status',
          description: 'Label for the status of a logged model on the logged model details page',
        })}
        value={<ExperimentLoggedModelStatusIndicator data={loggedModel} />}
      />
      <KeyValueProperty
        keyValue={intl.formatMessage({
          defaultMessage: 'Model ID',
          description: 'Label for the model ID of a logged model on the logged model details page',
        })}
        value={
          <DetailsOverviewCopyableIdBox
            value={loggedModel.info?.model_id ?? ''}
            css={{
              whiteSpace: 'nowrap',
            }}
          />
        }
      />
      {loggedModel.info?.source_run_id &&
        loggedModel.info?.experiment_id &&
        (relatedRunsLoading || relatedSourceRun) && (
          <KeyValueProperty
            keyValue={intl.formatMessage({
              defaultMessage: 'Source run',
              description: 'Label for the source run name of a logged model on the logged model details page',
            })}
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
        <KeyValueProperty
          keyValue={intl.formatMessage({
            defaultMessage: 'Source run ID',
            description: 'Label for the source run ID of a logged model on the logged model details page',
          })}
          value={
            <DetailsOverviewCopyableIdBox
              value={loggedModel.info?.source_run_id ?? ''}
              element={
                loggedModel.info?.experiment_id ? (
                  <Link to={Routes.getRunPageRoute(loggedModel.info?.experiment_id, loggedModel.info?.source_run_id)}>
                    {loggedModel.info?.source_run_id}
                  </Link>
                ) : undefined
              }
            />
          }
        />
      )}
      <KeyValueProperty
        keyValue={intl.formatMessage({
          defaultMessage: 'Logged from',
          description:
            'Label for the source (where it was logged from) of a logged model on the logged model details page. It can be e.g. a notebook or a file.',
        })}
        value={
          <ExperimentLoggedModelSourceBox
            loggedModel={loggedModel}
            displayDetails
            css={{ paddingTop: theme.spacing.xs, paddingBottom: theme.spacing.xs, wordBreak: 'break-all' }}
          />
        }
      />
    </>
  );

  return [
    {
      id: ExperimentLoggedModelDetailsMetadataSections.DETAILS,
      title: intl.formatMessage({
        defaultMessage: 'About this logged model',
        description: 'Title for the details sidebar of a logged model on the logged model details page',
      }),
      content: detailsContent,
    },
    {
      id: ExperimentLoggedModelDetailsMetadataSections.DATASETS,
      title: intl.formatMessage({
        defaultMessage: 'Datasets used',
        description: 'Label for the datasets used by a logged model on the logged model details page',
      }),
      content: loggedModel && <ExperimentLoggedModelAllDatasetsList loggedModel={loggedModel} empty={<NoneCell />} />,
    },
    {
      id: ExperimentLoggedModelDetailsMetadataSections.MODEL_VERSIONS,
      title: intl.formatMessage({
        defaultMessage: 'Model versions',
        description: 'Label for the model versions of a logged model on the logged model details page',
      }),
      content: loggedModel && (
        <ExperimentLoggedModelDetailsModelVersionsList empty={<NoneCell />} loggedModel={loggedModel} />
      ),
    },
  ];
};
