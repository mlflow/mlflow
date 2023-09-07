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
import { DatasetSourceTypes, RunDatasetWithTags } from '../../../../types';
import { useEffect, useMemo, useState } from 'react';
import React from 'react';
import { FormattedMessage } from 'react-intl';

export interface DatasetsCellRendererProps {
  datasetWithTags: RunDatasetWithTags;
}

export const ExperimentViewDatasetSchema = ({
  datasetWithTags,
}: DatasetsCellRendererProps): JSX.Element => {
  const { theme } = useDesignSystemTheme();
  const { dataset, tags } = datasetWithTags;
  const [value, setValue] = useState('');

  if (dataset.schema === null || dataset.schema === '') {
    return (
      <div css={{ marginLeft: theme.spacing.lg, marginTop: theme.spacing.md, width: '100%' }}>
        <div css={{ display: 'flex', flexDirection: 'column', justifyContent: 'flex-start' }}>
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
        <div css={{ marginLeft: theme.spacing.lg, marginTop: theme.spacing.md, width: '100%' }}>
          <div css={{ display: 'flex', flexDirection: 'column', justifyContent: 'flex-start' }}>
            <TableFilterInput
              value={value}
              placeholder='Search fields'
              onChange={(e) => setValue(e.target.value)}
              onClear={() => {
                setValue('');
              }}
              css={{ width: '100%' }}
            />
            <Spacer />
            <div css={{ display: 'flex', flexDirection: 'row' }}>
              <Table css={{ width: '100%' }}>
                <TableRow isHeader>
                  <TableHeader>
                    <FormattedMessage
                      defaultMessage='Name'
                      description='Header for "name" column in the experiment run dataset schema'
                    />
                  </TableHeader>
                  <TableHeader>
                    <FormattedMessage
                      defaultMessage='Type'
                      description='Header for "type" column in the experiment run dataset schema'
                    />
                  </TableHeader>
                </TableRow>
                {schema.mlflow_colspec.map((row: { name: string; type: string }, idx: number) => (
                  <TableRow key={`table-body-row-${idx}`}>
                    <TableCell>{row.name}</TableCell>
                    <TableCell>{row.type}</TableCell>
                  </TableRow>
                ))}
              </Table>
            </div>
          </div>
        </div>
      );
    } else if ('mlflow_tensorspec' in schema) {
      // if the dataset schema is tensorspec
      return (
        <div css={{ marginLeft: theme.spacing.lg, display: 'flex', alignItems: 'center' }}>
          <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <TableIcon css={{ fontSize: '56px', color: theme.colors.grey600 }} />
            <Header title={<div css={{ color: theme.colors.grey600 }}>Array Datasource</div>} />
            <Typography.Text color={theme.colors.grey600} css={{ textAlign: 'center' }}>
              <FormattedMessage
                defaultMessage='The dataset is an array. To see a preview of the dataset, view the dataset in the training notebook.'
                description='Notification when the dataset is an array data source in the experiment run dataset schema'
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
            <Header
              title={<div css={{ color: theme.colors.grey600 }}>Unrecognized Schema Format</div>}
            />
            <Typography.Text color={theme.colors.grey600}>
              <FormattedMessage
                defaultMessage='Raw Schema JSON: '
                description='Label for the raw schema JSON in the experiment run dataset schema'
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
