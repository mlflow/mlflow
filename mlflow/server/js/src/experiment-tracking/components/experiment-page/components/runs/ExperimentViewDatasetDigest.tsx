import { Typography } from '@databricks/design-system';
import type { RunDatasetWithTags } from '../../../../types';
import { FormattedMessage } from 'react-intl';

export const ExperimentViewDatasetDigest = ({ datasetWithTags }: { datasetWithTags: RunDatasetWithTags }) => {
  const { dataset } = datasetWithTags;
  return (
    <Typography.Hint>
      <FormattedMessage
        defaultMessage="Digest: {digest}"
        description="Experiment dataset drawer > digest > label and value"
        values={{ digest: <code>{dataset.digest}</code> }}
      />
    </Typography.Hint>
  );
};
