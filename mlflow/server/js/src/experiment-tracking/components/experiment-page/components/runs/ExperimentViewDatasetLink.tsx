import { Button, CopyIcon, NewWindowIcon, Typography } from '@databricks/design-system';
import type { RunDatasetWithTags } from '../../../../types';
import { DatasetSourceTypes } from '../../../../types';
import React from 'react';
import { FormattedMessage } from 'react-intl';
import { getDatasetSourceUrl } from '../../../../utils/DatasetUtils';
import { CopyButton } from '../../../../../shared/building_blocks/CopyButton';

export interface DatasetLinkProps {
  datasetWithTags: RunDatasetWithTags;
  runTags: Record<string, { key: string; value: string }>;
}

export const ExperimentViewDatasetLink = ({ datasetWithTags, runTags }: DatasetLinkProps) => {
  const { dataset } = datasetWithTags;
  if (dataset.sourceType === DatasetSourceTypes.HTTP || dataset.sourceType === DatasetSourceTypes.HUGGING_FACE) {
    const url = getDatasetSourceUrl(datasetWithTags);
    if (url) {
      return (
        <Button
          type="primary"
          componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewdatasetlink.tsx_19_1"
          icon={<NewWindowIcon />}
          href={url}
          target="_blank"
        >
          <FormattedMessage
            defaultMessage="Open dataset"
            description="Text for the HTTP/HF location link in the experiment run dataset drawer"
          />
        </Button>
      );
    }
  }
  if (dataset.sourceType === DatasetSourceTypes.S3) {
    const url = getDatasetSourceUrl(datasetWithTags);
    if (url) {
      return (
        <CopyButton
          componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewdatasetlink.tsx_19_2"
          icon={<CopyIcon />}
          copyText={url}
        >
          <FormattedMessage
            defaultMessage="Copy S3 URI to clipboard"
            description="Text for the HTTP/HF location link in the experiment run dataset drawer"
          />
        </CopyButton>
      );
    }
  }
  if (dataset.sourceType === DatasetSourceTypes.EXTERNAL) {
    return (
      <Button
        componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewdatasetlink.tsx_19_3"
        icon={<NewWindowIcon />}
      >
        <FormattedMessage
          defaultMessage="Go to external location"
          description="Text for the external location link in the experiment run dataset drawer"
        />
      </Button>
    );
  }
  return null;
};
