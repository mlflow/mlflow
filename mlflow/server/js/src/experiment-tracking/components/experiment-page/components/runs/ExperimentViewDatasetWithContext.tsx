import { TableIcon, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { RunDatasetWithTags } from '../../../../types';
import React from 'react';
import { MLFLOW_RUN_DATASET_CONTEXT_TAG } from '../../../../constants';

export interface DatasetWithContextProps {
  datasetWithTags: RunDatasetWithTags;
  displayTextAsLink: boolean;
  className?: string;
}

export const ExperimentViewDatasetWithContext = ({
  datasetWithTags,
  displayTextAsLink,
  className,
}: DatasetWithContextProps) => {
  const { dataset, tags } = datasetWithTags;
  const { theme } = useDesignSystemTheme();

  const contextTag = tags?.find(({ key }) => key === MLFLOW_RUN_DATASET_CONTEXT_TAG)?.value;

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        marginTop: theme.spacing.xs,
        marginBottom: theme.spacing.xs,
      }}
      className={className}
    >
      <TableIcon css={{ marginRight: theme.spacing.xs, color: theme.colors.textSecondary }} />
      {displayTextAsLink ? (
        <div>
          {dataset.name} ({dataset.digest})
        </div>
      ) : (
        <Typography.Text size="md" css={{ marginBottom: 0 }}>
          {dataset.name} ({dataset.digest})
        </Typography.Text>
      )}
      {contextTag && (
        <Tag
          componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewdatasetwithcontext.tsx_41"
          css={{
            textTransform: 'capitalize',
            marginLeft: theme.spacing.xs,
            marginRight: theme.spacing.xs,
          }}
        >
          {contextTag}
        </Tag>
      )}
    </div>
  );
};
