import { useMemo, useState } from 'react';

import type { LoggedModelProto } from '../../types';
import { Overflow, Spinner, TableIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { ExperimentLoggedModelDatasetButton } from './ExperimentLoggedModelDatasetButton';
import type { ColumnGroup } from '@ag-grid-community/core';
import { useExperimentLoggedModelOpenDatasetDetails } from './hooks/useExperimentLoggedModelOpenDatasetDetails';
import { FormattedMessage } from 'react-intl';

export const createLoggedModelDatasetColumnGroupId = (datasetName?: string, datasetDigest?: string, runId?: string) =>
  `metrics.${JSON.stringify([datasetName, datasetDigest, runId])}`;
const parseLoggedModelDatasetColumnGroupId = (groupId: string) => {
  try {
    const match = groupId.match(/metrics\.(.+)/);
    if (!match) {
      return null;
    }
    const datasetHash = match[1];
    const [datasetName, datasetDigest, runId] = JSON.parse(datasetHash);
    if (!datasetName || !datasetDigest) {
      return null;
    }
    return { datasetName, datasetDigest, runId };
  } catch {
    return null;
  }
};

export const ExperimentLoggedModelTableDatasetColHeader = ({ columnGroup }: { columnGroup: ColumnGroup }) => {
  const { onDatasetClicked } = useExperimentLoggedModelOpenDatasetDetails();
  const { theme } = useDesignSystemTheme();
  const [loading, setLoading] = useState(false);

  const datasetObject = useMemo(() => {
    try {
      const groupId = columnGroup.getGroupId();
      return groupId ? parseLoggedModelDatasetColumnGroupId(groupId) : null;
    } catch {
      return null;
    }
  }, [columnGroup]);
  if (!datasetObject) {
    return (
      <FormattedMessage
        defaultMessage="No dataset"
        description="Label for the metrics column group header that are not grouped by dataset"
      />
    );
  }
  return (
    <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, overflow: 'hidden' }}>
      Dataset:{' '}
      <Typography.Link
        css={{
          '.anticon': {
            fontSize: theme.general.iconFontSize,
          },
          fontSize: theme.typography.fontSizeBase,
          fontWeight: 'normal',
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
        }}
        role="button"
        componentId="mlflow.logged_model.list.metric_by_dataset_column_header"
        onClick={async () => {
          setLoading(true);
          try {
            await onDatasetClicked({
              datasetName: datasetObject.datasetName,
              datasetDigest: datasetObject.datasetDigest,
              runId: datasetObject.runId,
            });
          } finally {
            setLoading(false);
          }
        }}
      >
        {loading ? <Spinner size="small" /> : <TableIcon />}
        {datasetObject.datasetName} (#{datasetObject.datasetDigest})
      </Typography.Link>
    </span>
  );
};
