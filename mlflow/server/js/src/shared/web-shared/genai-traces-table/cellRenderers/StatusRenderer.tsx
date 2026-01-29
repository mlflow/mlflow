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
  const iconStyle = { color: '', width: 14, height: 14 };
  
  if (state === 'IN_PROGRESS') {
    iconStyle.color = theme.colors.textValidationWarning;
    return <ClockIcon css={iconStyle} />;
  }

  if (state === 'OK') {
    iconStyle.color = theme.colors.textValidationSuccess;
    return <CheckCircleIcon css={iconStyle} />;
  }

  if (state === 'ERROR') {
    iconStyle.color = theme.colors.textValidationDanger;
    return <XCircleIcon css={iconStyle} />;
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
      css={{ width: 'fit-content', maxWidth: '100%' }}
      componentId="mlflow.genai-traces-table.status"
    >
      <span
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {getIcon(state, theme)}
        <span
          css={{
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {intl.formatMessage(labelDescriptor)}
        </span>
      </span>
    </Tag>
  ) : (
    <NullCell isComparing={isComparing} />
  );
};
