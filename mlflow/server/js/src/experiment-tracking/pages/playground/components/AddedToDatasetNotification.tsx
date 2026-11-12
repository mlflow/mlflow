import { Notification } from '@databricks/design-system';
import { Fragment } from 'react';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';
import type { AddedToDatasetResult } from './AddToDatasetModal';

// Long enough to reach the dataset links; the toast also pauses while hovered or focused.
const NOTIFICATION_DURATION_MS = 10000;

interface Props {
  experimentId: string;
  datasets: AddedToDatasetResult[];
  onClose: () => void;
}

/** Success toast after adding a playground prompt to datasets, linking to each dataset. */
export const AddedToDatasetNotification = ({ experimentId, datasets, onClose }: Props) => {
  const updatedCount = datasets.filter((dataset) => dataset.updated).length;
  const addedCount = datasets.length - updatedCount;

  // A new tab, because the playground conversation only lives in this page's state.
  const renderDatasetLink = (dataset: AddedToDatasetResult) => (
    <Link
      componentId="mlflow.playground.add_to_dataset.added.dataset_link"
      to={Routes.getExperimentPageDatasetDetailRoute(experimentId, dataset.datasetId)}
      target="_blank"
    >
      {dataset.name}
    </Link>
  );

  const renderTitle = () => {
    if (datasets.length === 1) {
      const [dataset] = datasets;
      return dataset.updated ? (
        <FormattedMessage
          defaultMessage="Updated the existing record in {name}"
          description="Success toast on the playground when the dataset already had a record with the same inputs, so adding the prompt updated that record"
          values={{ name: renderDatasetLink(dataset) }}
        />
      ) : (
        <FormattedMessage
          defaultMessage="Added to {name}"
          description="Success toast on the playground after adding the current prompt to one evaluation dataset"
          values={{ name: renderDatasetLink(dataset) }}
        />
      );
    }
    if (updatedCount === 0) {
      return (
        <FormattedMessage
          defaultMessage="Added to {count, plural, one {# dataset} other {# datasets}}"
          description="Success toast on the playground after adding the current prompt to several evaluation datasets"
          values={{ count: datasets.length }}
        />
      );
    }
    if (addedCount === 0) {
      return (
        <FormattedMessage
          defaultMessage="Updated the existing record in {count, plural, one {# dataset} other {# datasets}}"
          description="Success toast on the playground when every selected dataset already had a record with the same inputs, so those records were updated"
          values={{ count: datasets.length }}
        />
      );
    }
    // Mixed outcome; the dataset list in the description marks which ones were updated.
    return (
      <FormattedMessage
        defaultMessage="Added or updated records in {count, plural, one {# dataset} other {# datasets}}"
        description="Success toast on the playground when adding the prompt created a record in some datasets and updated a record with the same inputs in others"
        values={{ count: datasets.length }}
      />
    );
  };

  const renderDescription = () => {
    if (datasets.length === 1) {
      return datasets[0].updated ? (
        <FormattedMessage
          defaultMessage="A record with the same inputs was already in this dataset, so it was updated instead of duplicated."
          description="Explains on the playground success toast why adding the prompt updated an existing dataset record"
        />
      ) : null;
    }
    return datasets.map((dataset, index) => (
      <Fragment key={dataset.datasetId}>
        {index > 0 && ', '}
        {dataset.updated ? (
          <FormattedMessage
            defaultMessage="{name} (updated)"
            description="A dataset in the playground success toast's dataset list whose existing record with the same inputs was updated"
            values={{ name: renderDatasetLink(dataset) }}
          />
        ) : (
          renderDatasetLink(dataset)
        )}
      </Fragment>
    ));
  };

  const description = renderDescription();

  return (
    <Notification.Provider>
      <Notification.Root
        severity="success"
        componentId="mlflow.playground.add_to_dataset.added"
        duration={NOTIFICATION_DURATION_MS}
        open
        onOpenChange={(open) => {
          if (!open) onClose();
        }}
      >
        <Notification.Title>{renderTitle()}</Notification.Title>
        {description && <Notification.Description>{description}</Notification.Description>}
        <Notification.Close componentId="mlflow.playground.add_to_dataset.added.close" />
      </Notification.Root>
      <Notification.Viewport />
    </Notification.Provider>
  );
};
