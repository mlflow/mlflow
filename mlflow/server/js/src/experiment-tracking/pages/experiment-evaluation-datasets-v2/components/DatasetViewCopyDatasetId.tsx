import { CopyIcon, Typography } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import type { Dataset } from '../hooks/useDatasetsQueries';

/**
 * Dataset detail page component responsible for copying
 * the datasetId after clicking on the icon
 */
export const DatasetViewCopyDatasetId = ({ dataset }: { dataset: Dataset }) => {
  const intl = useIntl();

  return (
    <Typography.Text
      size="md"
      dangerouslySetAntdProps={{
        copyable: {
          text: dataset.dataset_id,
          icon: <CopyIcon />,
          tooltips: [
            intl.formatMessage({
              defaultMessage: 'Copy dataset id',
              description: 'Copy tooltip to copy dataset id from dataset detail page header',
            }),
            intl.formatMessage({
              defaultMessage: 'Dataset id copied',
              description: 'Tooltip displayed after dataset id was successfully copied to clipboard',
            }),
          ],
        },
      }}
    />
  );
};
