import type { CellContext, ColumnDefTemplate } from '@tanstack/react-table';
import type { ModelTraceInfoWithRunName } from './hooks/useExperimentTraces';
import { CheckCircleIcon, ClockIcon, XCircleIcon, useDesignSystemTheme } from '@databricks/design-system';
import { ExperimentViewTracesStatusLabels } from './TracesView.utils';
import { useIntl } from 'react-intl';
import type { Theme } from '@emotion/react';

const getIcon = (status: ModelTraceInfoWithRunName['status'], theme: Theme) => {
  if (status === 'IN_PROGRESS') {
    return <ClockIcon css={{ color: theme.colors.textValidationWarning }} />;
  }

  if (status === 'OK') {
    return <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess }} />;
  }

  if (status === 'ERROR') {
    return <XCircleIcon css={{ color: theme.colors.textValidationDanger }} />;
  }

  return null;
};

export const TracesViewTableStatusCell: ColumnDefTemplate<CellContext<ModelTraceInfoWithRunName, unknown>> = ({
  row: { original },
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const labelDescriptor = ExperimentViewTracesStatusLabels[original.status || 'UNSET'];

  return (
    <div css={{ display: 'flex', gap: theme.spacing.xs, alignItems: 'center' }}>
      {getIcon(original.status, theme)}
      {labelDescriptor ? intl.formatMessage(labelDescriptor) : ''}
    </div>
  );
};
