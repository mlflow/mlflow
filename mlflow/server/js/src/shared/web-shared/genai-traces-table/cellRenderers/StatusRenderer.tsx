import type { Theme } from '@emotion/react';

import { CheckCircleIcon, ClockIcon, useDesignSystemTheme, XCircleIcon, Tag, type TagColors } from '@databricks/design-system';
import { useIntl, defineMessage } from '@databricks/i18n';

import { NullCell } from './NullCell';
import type { ModelTraceInfoV3 } from '../../model-trace-explorer';

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

const getIcon = (state: ModelTraceInfoV3['state'], theme: Theme) => {
  if (state === 'IN_PROGRESS') {
    return <ClockIcon css={{ color: theme.colors.textValidationWarning, width: 14, height: 14 }} />;
  }

  if (state === 'OK') {
    return <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess, width: 14, height: 14 }} />;
  }

  if (state === 'ERROR') {
    return <XCircleIcon css={{ color: theme.colors.textValidationDanger, width: 14, height: 14 }} />;
  }

  return null;
};

const getTagColor = (state: ModelTraceInfoV3['state']): TagColors | undefined => {
  if (state === 'IN_PROGRESS') {
    return 'lemon';
  }

  if (state === 'OK') {
    return 'teal';
  }

  if (state === 'ERROR') {
    return 'coral';
  }

  return undefined;
};

export const StatusCellRenderer = ({
  original,
  isComparing,
}: {
  original: ModelTraceInfoV3 | undefined;
  isComparing: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const labelDescriptor = ExperimentViewTracesStatusLabels[original?.state || 'STATE_UNSPECIFIED'];
  const state = original?.state || 'STATE_UNSPECIFIED';

  return labelDescriptor ? (
    <Tag
      color={getTagColor(state)}
      componentId="mlflow.genai-traces-table.status"
      css={{
        display: 'inline-flex',
        alignItems: 'center',
        width: 'fit-content',
      }}
    >
      {getIcon(state, theme)}
      <span css={{ marginLeft: theme.spacing.xs }}>
        {intl.formatMessage(labelDescriptor)}
      </span>
    </Tag>
  ) : (
    <NullCell isComparing={isComparing} />
  );
};
