import { Button, NewWindowIcon, Typography } from '@databricks/design-system';
import { DatasetSourceTypes, RunDatasetWithTags } from '../../../../types';
import React from 'react';
import { FormattedMessage } from 'react-intl';

export interface DatasetLinkProps {
  datasetWithTags: RunDatasetWithTags;
  runTags: Record<string, { key: string; value: string }>;
}

export const ExperimentViewDatasetLink = ({ datasetWithTags, runTags }: DatasetLinkProps) => {
  const { dataset } = datasetWithTags;
  if (dataset.source_type === DatasetSourceTypes.EXTERNAL) {
    return (
      <Button>
        <NewWindowIcon />
        <FormattedMessage
          defaultMessage='Go to external location'
          description='Text for the external location link in the experiment run dataset drawer'
        />
      </Button>
    );
  }
  return null;
};
