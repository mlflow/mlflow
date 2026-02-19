import { Typography } from '@databricks/design-system';
import type { RunDatasetWithTags } from '../../../../types';
import { DatasetSourceTypes } from '../../../../types';
import React from 'react';
import { FormattedMessage } from 'react-intl';

export interface ExperimentViewDatasetSourceTypeProps {
  datasetWithTags: RunDatasetWithTags;
}

export const ExperimentViewDatasetSourceType = ({ datasetWithTags }: ExperimentViewDatasetSourceTypeProps) => {
  const { dataset } = datasetWithTags;

  const sourceType = dataset.sourceType;

  const getSourceTypeLabel = () => {
    if (sourceType === DatasetSourceTypes.HTTP || sourceType === DatasetSourceTypes.EXTERNAL) {
      return (
        <FormattedMessage
          defaultMessage="HTTP"
          description="Experiment dataset drawer > source type > HTTP source type label"
        />
      );
    }
    if (sourceType === DatasetSourceTypes.S3) {
      return (
        <FormattedMessage
          defaultMessage="S3"
          description="Experiment dataset drawer > source type > S3 source type label"
        />
      );
    }
    if (sourceType === DatasetSourceTypes.HUGGING_FACE) {
      return (
        <FormattedMessage
          defaultMessage="Hugging Face"
          description="Experiment dataset drawer > source type > Hugging Face source type label"
        />
      );
    }
    return null;
  };

  const typeLabel = getSourceTypeLabel();

  if (typeLabel) {
    return (
      <Typography.Hint>
        <FormattedMessage
          defaultMessage="Source type: {typeLabel}"
          description="Experiment dataset drawer > source type > label"
          values={{ typeLabel }}
        />
      </Typography.Hint>
    );
  }

  return null;
};
