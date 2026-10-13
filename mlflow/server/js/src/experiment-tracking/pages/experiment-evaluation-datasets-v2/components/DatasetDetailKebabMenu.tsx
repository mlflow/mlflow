import { useCallback, useState } from 'react';
import {
  Alert,
  Button,
  DangerModal,
  DropdownMenu,
  NewWindowIcon,
  OverflowIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useNavigate } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import { ExperimentPageTabName } from '@mlflow/mlflow/src/experiment-tracking/constants';

import type { Dataset } from '../hooks/useDatasetsQueries';
import { useDatasetDelete } from '../hooks/useDatasetDelete';
import type { DatasetNotifyApi } from '../hooks/useDatasetNotifications';
import { DatasetMetadataModal } from './DatasetMetadataModal';

interface DatasetDetailKebabMenuProps {
  experimentId: string;
  dataset: Dataset;
  notify: DatasetNotifyApi;
}

/**
 * Page-level overflow menu for the V2 dataset detail page. Hosts:
 * - "View in Unity Catalog" — deep-links to /explore/data/{catalog}/{schema}/{table}
 *   when `dataset.name` is a 3-part UC path. Hidden otherwise.
 * - "View dataset metadata" — opens a modal with digest/schema/profile. Disabled when
 *   all three fields are empty.
 * - "Delete dataset" — destructive confirm via DangerModal.
 */
export const DatasetDetailKebabMenu = ({ experimentId, dataset, notify }: DatasetDetailKebabMenuProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const navigate = useNavigate();

  const handleAfterDelete = useCallback(() => {
    navigate(Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Datasets));
  }, [navigate, experimentId]);

  const datasetDelete = useDatasetDelete({
    experimentId,
    notify,
    onAfterDelete: handleAfterDelete,
  });

  const handleRequestDelete = useCallback(() => {
    datasetDelete.requestDelete(dataset);
  }, [datasetDelete, dataset]);

  // OSS has no Unity Catalog link target — hide the "View in Unity Catalog" menu item.
  const ucPath: string | undefined = undefined;
  const hasMetadata = Boolean(dataset.digest || dataset.schema || dataset.profile);
  const [metadataModalOpen, setMetadataModalOpen] = useState(false);

  return (
    <>
      <DropdownMenu.Root>
        <DropdownMenu.Trigger asChild>
          <Button
            componentId="mlflow.eval-datasets-v2.detail.actions"
            icon={<OverflowIcon />}
            aria-label={intl.formatMessage({
              defaultMessage: 'Dataset actions',
              description: 'Aria label for the V2 evaluation dataset detail page overflow-menu trigger',
            })}
          />
        </DropdownMenu.Trigger>
        <DropdownMenu.Content align="end">
          {ucPath && (
            <DropdownMenu.Item componentId="mlflow.eval-datasets-v2.detail.actions.view-in-unity-catalog" asChild>
              <Typography.Link
                componentId="mlflow.eval-datasets-v2.detail.actions.view-in-unity-catalog-link"
                href={ucPath}
                target="_blank"
                rel="noopener noreferrer"
              >
                <FormattedMessage
                  defaultMessage="View in Unity Catalog"
                  description="Overflow-menu item linking to the V2 evaluation dataset's table page in Unity Catalog"
                />
                <DropdownMenu.HintColumn>
                  <NewWindowIcon />
                </DropdownMenu.HintColumn>
              </Typography.Link>
            </DropdownMenu.Item>
          )}
          <DropdownMenu.Item
            componentId="mlflow.eval-datasets-v2.detail.actions.view-metadata"
            disabled={!hasMetadata}
            onClick={() => setMetadataModalOpen(true)}
          >
            <FormattedMessage
              defaultMessage="View dataset metadata"
              description="Overflow-menu item that opens the V2 evaluation dataset metadata modal"
            />
          </DropdownMenu.Item>
          <DropdownMenu.Item
            componentId="mlflow.eval-datasets-v2.detail.actions.delete-dataset"
            onClick={handleRequestDelete}
          >
            <FormattedMessage
              defaultMessage="Delete dataset"
              description="Overflow-menu item to delete the current V2 evaluation dataset from its detail page"
            />
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Root>

      <DatasetMetadataModal dataset={dataset} visible={metadataModalOpen} onClose={() => setMetadataModalOpen(false)} />

      <DangerModal
        componentId="mlflow.eval-datasets-v2.detail.delete-confirm-modal"
        visible={datasetDelete.pendingDataset !== null}
        title={
          <FormattedMessage
            defaultMessage="Delete dataset"
            description="Title for the V2 evaluation dataset delete confirmation modal on the detail page"
          />
        }
        okText={intl.formatMessage({
          defaultMessage: 'Delete',
          description: 'Confirm-button text for the V2 evaluation dataset delete modal on the detail page',
        })}
        cancelText={intl.formatMessage({
          defaultMessage: 'Cancel',
          description: 'Cancel-button text for the V2 evaluation dataset delete modal on the detail page',
        })}
        okButtonProps={{ loading: datasetDelete.isDeleting || datasetDelete.isPolling }}
        cancelButtonProps={{ disabled: datasetDelete.isDeleting || datasetDelete.isPolling }}
        onOk={datasetDelete.confirmDelete}
        onCancel={datasetDelete.cancelDelete}
      >
        <FormattedMessage
          defaultMessage='Are you sure you want to delete the dataset "{name}"? This action cannot be undone.'
          description="Body for the V2 evaluation dataset delete confirmation modal on the detail page"
          values={{ name: datasetDelete.pendingDataset?.name ?? datasetDelete.pendingDataset?.dataset_id ?? '' }}
        />
        {datasetDelete.error && (
          <Alert
            componentId="mlflow.eval-datasets-v2.detail.delete-error"
            type="error"
            message={datasetDelete.error.message}
            css={{ marginTop: theme.spacing.sm }}
            closable={false}
          />
        )}
      </DangerModal>
    </>
  );
};
