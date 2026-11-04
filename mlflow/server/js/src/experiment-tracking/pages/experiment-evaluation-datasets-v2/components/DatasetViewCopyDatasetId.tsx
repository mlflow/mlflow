import { CopyActionButton } from '@databricks/web-shared/copy';
import { useIntl } from 'react-intl';
import type { Dataset } from '../hooks/useDatasetsQueries';

/**
 * Dataset detail page component responsible for copying
 * the datasetId after clicking on the icon
 */
export const DatasetViewCopyDatasetId = ({ dataset }: { dataset: Dataset }) => {
  const intl = useIntl();

  return (
    <CopyActionButton
      componentId="mlflow.datasets.header.copy-dataset-id"
      copyText={dataset.dataset_id}
      copyTooltip={intl.formatMessage({
        defaultMessage: 'Copy dataset id',
        description: 'Copy tooltip to copy dataset id from dataset detail page header',
      })}
    />
  );
};
