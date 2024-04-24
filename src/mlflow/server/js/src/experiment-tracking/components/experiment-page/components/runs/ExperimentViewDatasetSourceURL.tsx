import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { DatasetSourceTypes, RunDatasetWithTags } from '../../../../types';
import { getDatasetSourceUrl } from '../../../../utils/DatasetUtils';

export interface ExperimentViewDatasetSourceProps {
  datasetWithTags: RunDatasetWithTags;
}

export const ExperimentViewDatasetSourceURL = ({ datasetWithTags }: ExperimentViewDatasetSourceProps) => {
  const { dataset } = datasetWithTags;
  const { theme } = useDesignSystemTheme();

  const sourceType = dataset.source_type;

  if (
    sourceType === DatasetSourceTypes.HTTP ||
    sourceType === DatasetSourceTypes.EXTERNAL ||
    sourceType === DatasetSourceTypes.HUGGING_FACE
  ) {
    const url = getDatasetSourceUrl(datasetWithTags);
    if (url) {
      return (
        <div
          css={{
            whiteSpace: 'nowrap',
            display: 'flex',
            fontSize: theme.typography.fontSizeSm,
            color: theme.colors.textSecondary,
            columnGap: theme.spacing.xs,
          }}
          title={url}
        >
          URL:{' '}
          <Typography.Link openInNewTab href={url} css={{ display: 'flex', overflow: 'hidden' }}>
            <span css={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{url}</span>
          </Typography.Link>
        </div>
      );
    }
  }
  if (sourceType === DatasetSourceTypes.S3) {
    try {
      const { uri } = JSON.parse(dataset.source);
      if (uri) {
        return (
          <Typography.Hint
            title={uri}
            css={{
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}
          >
            S3 URI: {uri}
          </Typography.Hint>
        );
      }
    } catch {
      return null;
    }
  }
  return null;
};
