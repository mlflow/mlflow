import {
  Header,
  TableIcon,
  useDesignSystemTheme,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableFilterInput,
  Spacer,
  Typography,
} from '@databricks/design-system';
import { ExperimentViewDatasetSchemaTable } from './ExperimentViewDatasetSchemaTable';
import type { RunDatasetWithTags } from '../../../../types';
import { DatasetSourceTypes } from '../../../../types';
import { useEffect, useMemo, useState } from 'react';
import React from 'react';
import { FormattedMessage } from 'react-intl';

export interface DatasetsCellRendererProps {
  datasetWithTags: RunDatasetWithTags;
}

export const ExperimentViewDatasetSchema = ({ datasetWithTags }: DatasetsCellRendererProps): JSX.Element => {
  const { theme } = useDesignSystemTheme();
  const { dataset } = datasetWithTags;
  const [filter, setFilter] = useState('');

  if (dataset.schema === null || dataset.schema === '') {
    return (
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          height: '100vh',
        }}
      >
        <div
          css={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'flex-start',
            alignContent: 'center',
          }}
        >
          <Header title={<div css={{ color: theme.colors.grey600 }}>No schema available</div>} />
        </div>
      </div>
    );
  }
  try {
    const schema = JSON.parse(dataset.schema);
    if ('mlflow_colspec' in schema) {
      // if the dataset schema is colspec
      return (
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            height: '100vh',
          }}
        >
          <div
            css={{
              marginTop: theme.spacing.sm,
              form: { width: '100%' },
            }}
          >
            <TableFilterInput
              componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewdatasetschema.tsx_92"
              value={filter}
              placeholder="Search fields"
              onChange={(e) => setFilter(e.target.value)}
              onClear={() => {
                setFilter('');
              }}
              css={{ width: '100%' }}
              containerProps={{ style: { width: 'auto' } }}
            />
          </div>
          <div
            css={{
              marginTop: theme.spacing.sm,
              overflow: 'hidden',
            }}
          >
            <ExperimentViewDatasetSchemaTable schema={schema.mlflow_colspec} filter={filter} />
          </div>
        </div>
      );
    } else if ('mlflow_tensorspec' in schema) {
      // if the dataset schema is tensorspec
      return (
        <div css={{ height: '100vh' }}>
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
            }}
          >
            <TableIcon css={{ fontSize: '56px', color: theme.colors.grey600 }} />
            <Header title={<div css={{ color: theme.colors.grey600 }}>Array Datasource</div>} />
            {/* @ts-expect-error Type 'string' is not assignable to type '"primary" | "secondary" | "info" | "error" | "success" | "warning" | undefined' */}
            <Typography.Text color={theme.colors.grey600} css={{ textAlign: 'center' }}>
              <FormattedMessage
                defaultMessage="The dataset is an array. To see a preview of the dataset, view the dataset in the training notebook."
                description="Notification when the dataset is an array data source in the experiment run dataset schema"
              />
            </Typography.Text>
          </div>
        </div>
      );
    } else {
      // if the dataset schema is not colspec or tensorspec
      return (
        <div css={{ marginLeft: theme.spacing.lg, marginTop: theme.spacing.md, width: '100%' }}>
          <div css={{ display: 'flex', flexDirection: 'column', justifyContent: 'flex-start' }}>
            <Header title={<div css={{ color: theme.colors.grey600 }}>Unrecognized Schema Format</div>} />
            {/* @ts-expect-error Type 'string' is not assignable to type '"primary" | "secondary" | "info" | "error" | "success" | "warning" | undefined' */}
            <Typography.Text color={theme.colors.grey600}>
              <FormattedMessage
                defaultMessage="Raw Schema JSON: "
                description="Label for the raw schema JSON in the experiment run dataset schema"
              />
              {JSON.stringify(schema)}
            </Typography.Text>
          </div>
        </div>
      );
    }
  } catch {
    return (
      <div css={{ marginLeft: theme.spacing.lg, marginTop: theme.spacing.md, width: '100%' }}>
        <div css={{ display: 'flex', flexDirection: 'column', justifyContent: 'flex-start' }}>
          <Header title={<div css={{ color: theme.colors.grey600 }}>No schema available</div>} />
        </div>
      </div>
    );
  }
};
