import type { Theme } from '@emotion/react';

import { CheckCircleIcon, ClockIcon, useDesignSystemTheme, XCircleIcon } from '@databricks/design-system';
import { useIntl, defineMessage } from '@databricks/i18n';

import { NullCell } from './NullCell';
import type { TraceInfoV3 } from '../types';

export const ExperimentViewTracesStatusLabels = {
  STATE_UNSPECIFIED: null,
  IN_PROGRESS: defineMessage({
    defaultMessage: 'In progress',
    description: 'Experiment page > traces table > status label > in progress',
  }),
  OK: defineMessage({
    defaultMessage: 'OK',
    description: 'Experiment page > traces table > status label > ok',
  }),
  ERROR: defineMessage({
    defaultMessage: 'Error',
    description: 'Experiment page > traces table > status label > error',
  }),
};

const getIcon = (state: TraceInfoV3['state'], theme: Theme) => {
  if (state === 'IN_PROGRESS') {
    return <ClockIcon css={{ color: theme.colors.textValidationWarning }} />;
  }

  if (state === 'OK') {
    return <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess }} />;
  }

  if (state === 'ERROR') {
    return <XCircleIcon css={{ color: theme.colors.textValidationDanger }} />;
  }

  return null;
};

export const StatusCellRenderer = ({
  original,
  isComparing,
}: {
  original: TraceInfoV3 | undefined;
  isComparing: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const labelDescriptor = ExperimentViewTracesStatusLabels[original?.state || 'STATE_UNSPECIFIED'];

  return labelDescriptor ? (
    <div css={{ display: 'flex', gap: theme.spacing.xs, alignItems: 'center' }}>
      {getIcon(original?.state || 'STATE_UNSPECIFIED', theme)}
      {labelDescriptor ? intl.formatMessage(labelDescriptor) : ''}
    </div>
  ) : (
    <NullCell isComparing={isComparing} />
  );
};
